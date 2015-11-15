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

#ifndef UME_SIMD_TRAITS_H_
#define UME_SIMD_TRAITS_H_

namespace UME {
    namespace SIMD {

        /*******************************************************
        * SIMDTraits class gathers all types related to given SIMD_TYPE
        * This class is useful for function templates requiring additional
        * type information. Instead of passing multiple SIMD types as
        * template parameters, one can use SIMDTraits class to deduce
        * actual type at compile-time.
        *
        * SIMDTraits class needs to be explicitly specialized for any
        * SIMD type to carry type relation between different types.
        *
        * These classes are meant to be used in user code
        * and are distinct from trait classes defined inside plugins.
        *
        *******************************************************/

        template<typename SIMD_TYPE>
        class SIMDTraits {
            // Nothing here. This template should be specialized for any SIMD type.
        };

        template<>
        class SIMDTraits<SIMD1_8u>
        {
        public:
            typedef uint8_t      SCALAR_T;
            typedef SIMD1_8i     INT_VEC_T;
            typedef SIMDMask1    MASK_T;
            typedef SIMDSwizzle1 SWIZZLE_T;

            typedef SIMD2_8u     DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD2_8u>
        {
        public:
            typedef uint8_t      SCALAR_T;
            typedef SIMD2_8i     INT_VEC_T;
            typedef SIMDMask2    MASK_T;
            typedef SIMDSwizzle2 SWIZZLE_T;

            typedef SIMD1_8u     HALF_LEN_VEC_T;
            typedef SIMD4_8u     DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD4_8u>
        {
        public:
            typedef uint8_t      SCALAR_T;
            typedef SIMD4_8i     INT_VEC_T;
            typedef SIMDMask4    MASK_T;
            typedef SIMDSwizzle4 SWIZZLE_T;

            typedef SIMD2_8u     HALF_LEN_VEC_T;
            typedef SIMD8_8u     DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD8_8u>
        {
        public:
            typedef uint8_t      SCALAR_T;
            typedef SIMD8_8i     INT_VEC_T;
            typedef SIMDMask8    MASK_T;
            typedef SIMDSwizzle8 SWIZZLE_T;

            typedef SIMD4_8u     HALF_LEN_VEC_T;
            typedef SIMD16_8u    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD16_8u>
        {
        public:
            typedef uint8_t       SCALAR_T;
            typedef SIMD16_8i     INT_VEC_T;
            typedef SIMDMask16    MASK_T;
            typedef SIMDSwizzle16 SWIZZLE_T;

            typedef SIMD8_8u      HALF_LEN_VEC_T;
            typedef SIMD32_8u     DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD32_8u>
        {
        public:
            typedef uint8_t       SCALAR_T;
            typedef SIMD32_8i     INT_VEC_T;
            typedef SIMDMask32    MASK_T;
            typedef SIMDSwizzle32 SWIZZLE_T;

            typedef SIMD16_8u     HALF_LEN_VEC_T;
            typedef SIMD64_8u     DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD64_8u>
        {
        public:
            typedef uint8_t       SCALAR_T;
            typedef SIMD64_8i     INT_VEC_T;
            typedef SIMDMask64    MASK_T;
            typedef SIMDSwizzle64 SWIZZLE_T;

            typedef SIMD32_8u     HALF_LEN_VEC_T;
            typedef SIMD128_8u    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD128_8u>
        {
        public:
            typedef uint8_t        SCALAR_T;
            typedef SIMD128_8i     INT_VEC_T;
            typedef SIMDMask128    MASK_T;
            typedef SIMDSwizzle128 SWIZZLE_T;

            typedef SIMD64_8u      HALF_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD1_16u>
        {
        public:
            typedef uint16_t     SCALAR_T;
            typedef SIMD1_16i    INT_VEC_T;
            typedef SIMDMask1    MASK_T;
            typedef SIMDSwizzle1 SWIZZLE_T;

            typedef SIMD2_16u    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD2_16u>
        {
        public:
            typedef uint16_t     SCALAR_T;
            typedef SIMD2_16i    INT_VEC_T;
            typedef SIMDMask2    MASK_T;
            typedef SIMDSwizzle2 SWIZZLE_T;

            typedef SIMD1_16u    HALF_LEN_VEC_T;
            typedef SIMD4_16u    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD4_16u>
        {
        public:
            typedef uint16_t     SCALAR_T;
            typedef SIMD4_16i    INT_VEC_T;
            typedef SIMDMask4    MASK_T;
            typedef SIMDSwizzle4 SWIZZLE_T;

            typedef SIMD2_16u    HALF_LEN_VEC_T;
            typedef SIMD8_16u    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD8_16u>
        {
        public:
            typedef uint16_t     SCALAR_T;
            typedef SIMD8_16i    INT_VEC_T;
            typedef SIMDMask8    MASK_T;
            typedef SIMDSwizzle8 SWIZZLE_T;

            typedef SIMD4_16u    HALF_LEN_VEC_T;
            typedef SIMD16_16u   DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD16_16u>
        {
        public:
            typedef uint16_t      SCALAR_T;
            typedef SIMD16_16i    INT_VEC_T;
            typedef SIMDMask16    MASK_T;
            typedef SIMDSwizzle16 SWIZZLE_T;

            typedef SIMD8_16u     HALF_LEN_VEC_T;
            typedef SIMD32_16u    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD32_16u>
        {
        public:
            typedef uint16_t      SCALAR_T;
            typedef SIMD32_16i    INT_VEC_T;
            typedef SIMDMask32    MASK_T;
            typedef SIMDSwizzle32 SWIZZLE_T;

            typedef SIMD16_16u    HALF_LEN_VEC_T;
            typedef SIMD64_16u    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD64_16u>
        {
        public:
            typedef uint16_t      SCALAR_T;
            typedef SIMD64_16i    INT_VEC_T;
            typedef SIMDMask64    MASK_T;
            typedef SIMDSwizzle64 SWIZZLE_T;

            typedef SIMD32_16u    HALF_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD1_32u>
        {
        public:
            typedef uint32_t     SCALAR_T;
            typedef SIMD1_32i    INT_VEC_T;
            typedef SIMDMask1    MASK_T;
            typedef SIMDSwizzle1 SWIZZLE_T;

            typedef SIMD2_32u    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD2_32u>
        {
        public:
            typedef uint32_t     SCALAR_T;
            typedef SIMD2_32i    INT_VEC_T;
            typedef SIMDMask2    MASK_T;
            typedef SIMDSwizzle2 SWIZZLE_T;

            typedef SIMD1_32u    HALF_LEN_VEC_T;
            typedef SIMD4_32u    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD4_32u>
        {
        public:
            typedef uint32_t     SCALAR_T;
            typedef SIMD4_32i    INT_VEC_T;
            typedef SIMDMask4    MASK_T;
            typedef SIMDSwizzle4 SWIZZLE_T;

            typedef SIMD2_32u    HALF_LEN_VEC_T;
            typedef SIMD8_32u    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD8_32u>
        {
        public:
            typedef uint32_t     SCALAR_T;
            typedef SIMD8_32i    INT_VEC_T;
            typedef SIMDMask8    MASK_T;
            typedef SIMDSwizzle8 SWIZZLE_T;

            typedef SIMD4_32u    HALF_LEN_VEC_T;
            typedef SIMD16_32u   DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD16_32u>
        {
        public:
            typedef uint32_t      SCALAR_T;
            typedef SIMD16_32i    INT_VEC_T;
            typedef SIMDMask16    MASK_T;
            typedef SIMDSwizzle16 SWIZZLE_T;

            typedef SIMD8_32u     HALF_LEN_VEC_T;
            typedef SIMD16_32u    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD32_32u>
        {
        public:
            typedef uint32_t      SCALAR_T;
            typedef SIMD32_32i    INT_VEC_T;
            typedef SIMDMask32    MASK_T;
            typedef SIMDSwizzle32 SWIZZLE_T;

            typedef SIMD16_32u    HALF_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD1_64u>
        {
        public:
            typedef uint64_t     SCALAR_T;
            typedef SIMD1_64i    INT_VEC_T;
            typedef SIMDMask1    MASK_T;
            typedef SIMDSwizzle1 SWIZZLE_T;

            typedef SIMD2_64u    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD2_64u>
        {
        public:
            typedef uint64_t     SCALAR_T;
            typedef SIMD2_64i    INT_VEC_T;
            typedef SIMDMask2    MASK_T;
            typedef SIMDSwizzle2 SWIZZLE_T;

            typedef SIMD1_64u    HALF_LEN_VEC_T;
            typedef SIMD4_64u    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD4_64u>
        {
        public:
            typedef uint64_t     SCALAR_T;
            typedef SIMD4_64i    INT_VEC_T;
            typedef SIMDMask4    MASK_T;
            typedef SIMDSwizzle4 SWIZZLE_T;

            typedef SIMD2_64u    HALF_LEN_VEC_T;
            typedef SIMD8_64u    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD8_64u>
        {
        public:
            typedef uint64_t     SCALAR_T;
            typedef SIMD8_64i    INT_VEC_T;
            typedef SIMDMask8    MASK_T;
            typedef SIMDSwizzle8 SWIZZLE_T;

            typedef SIMD4_64u    HALF_LEN_VEC_T;
            typedef SIMD16_64u   DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD16_64u>
        {
        public:
            typedef uint64_t      SCALAR_T;
            typedef SIMD16_64i    INT_VEC_T;
            typedef SIMDMask16    MASK_T;
            typedef SIMDSwizzle16 SWIZZLE_T;

            typedef SIMD8_64u     HALF_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD1_8i>
        {
        public:
            typedef int8_t       SCALAR_T;
            typedef SIMD1_8u     UINT_VEC_T;
            typedef SIMDMask1    MASK_T;
            typedef SIMDSwizzle1 SWIZZLE_T;

            typedef SIMD2_8i    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD2_8i>
        {
        public:
            typedef int8_t       SCALAR_T;
            typedef SIMD2_8u     UINT_VEC_T;
            typedef SIMDMask2    MASK_T;
            typedef SIMDSwizzle2 SWIZZLE_T;

            typedef SIMD1_8i     HALF_LEN_VEC_T;
            typedef SIMD4_8i     DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD4_8i>
        {
        public:
            typedef int8_t       SCALAR_T;
            typedef SIMD4_8u     UINT_VEC_T;
            typedef SIMDMask4    MASK_T;
            typedef SIMDSwizzle4 SWIZZLE_T;

            typedef SIMD2_8i     HALF_LEN_VEC_T;
            typedef SIMD8_8i     DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD8_8i>
        {
        public:
            typedef int8_t       SCALAR_T;
            typedef SIMD8_8u     UINT_VEC_T;
            typedef SIMDMask8    MASK_T;
            typedef SIMDSwizzle8 SWIZZLE_T;

            typedef SIMD4_8i     HALF_LEN_VEC_T;
            typedef SIMD16_8i    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD16_8i>
        {
        public:
            typedef int8_t        SCALAR_T;
            typedef SIMD16_8u     UINT_VEC_T;
            typedef SIMDMask16    MASK_T;
            typedef SIMDSwizzle16 SWIZZLE_T;

            typedef SIMD8_8i      HALF_LEN_VEC_T;
            typedef SIMD32_8i     DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD32_8i>
        {
        public:
            typedef int8_t        SCALAR_T;
            typedef SIMD32_8u     UINT_VEC_T;
            typedef SIMDMask32    MASK_T;
            typedef SIMDSwizzle32 SWIZZLE_T;

            typedef SIMD16_8i     HALF_LEN_VEC_T;
            typedef SIMD64_8i     DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD64_8i>
        {
        public:
            typedef int8_t        SCALAR_T;
            typedef SIMD64_8u     UINT_VEC_T;
            typedef SIMDMask64    MASK_T;
            typedef SIMDSwizzle64 SWIZZLE_T;

            typedef SIMD32_8i     HALF_LEN_VEC_T;
            typedef SIMD128_8i    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD128_8i>
        {
        public:
            typedef int8_t         SCALAR_T;
            typedef SIMD128_8u     UINT_VEC_T;
            typedef SIMDMask128    MASK_T;
            typedef SIMDSwizzle128 SWIZZLE_T;

            typedef SIMD64_8i      HALF_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD1_16i>
        {
        public:
            typedef int16_t      SCALAR_T;
            typedef SIMD1_16u    UINT_VEC_T;
            typedef SIMDMask1    MASK_T;
            typedef SIMDSwizzle1 SWIZZLE_T;

            typedef SIMD2_16i   DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD2_16i>
        {
        public:
            typedef int16_t      SCALAR_T;
            typedef SIMD2_16u    UINT_VEC_T;
            typedef SIMDMask2    MASK_T;
            typedef SIMDSwizzle2 SWIZZLE_T;

            typedef SIMD1_16i    HALF_LEN_VEC_T;
            typedef SIMD4_16i    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD4_16i>
        {
        public:
            typedef int16_t      SCALAR_T;
            typedef SIMD4_16u    UINT_VEC_T;
            typedef SIMDMask4    MASK_T;
            typedef SIMDSwizzle4 SWIZZLE_T;

            typedef SIMD2_16i    HALF_LEN_VEC_T;
            typedef SIMD8_16i    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD8_16i>
        {
        public:
            typedef int16_t      SCALAR_T;
            typedef SIMD8_16u    UINT_VEC_T;
            typedef SIMDMask8    MASK_T;
            typedef SIMDSwizzle8 SWIZZLE_T;

            typedef SIMD4_16i    HALF_LEN_VEC_T;
            typedef SIMD16_16i   DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD16_16i>
        {
        public:
            typedef int16_t       SCALAR_T;
            typedef SIMD16_16u    UINT_VEC_T;
            typedef SIMDMask16    MASK_T;
            typedef SIMDSwizzle16 SWIZZLE_T;

            typedef SIMD8_16i     HALF_LEN_VEC_T;
            typedef SIMD32_16i    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD32_16i>
        {
        public:
            typedef int16_t       SCALAR_T;
            typedef SIMD32_16u    UINT_VEC_T;
            typedef SIMDMask32    MASK_T;
            typedef SIMDSwizzle32 SWIZZLE_T;

            typedef SIMD16_16i    HALF_LEN_VEC_T;
            typedef SIMD64_16i    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD64_16i>
        {
        public:
            typedef int16_t       SCALAR_T;
            typedef SIMD64_16u    UINT_VEC_T;
            typedef SIMDMask64    MASK_T;
            typedef SIMDSwizzle64 SWIZZLE_T;

            typedef SIMD32_16i    HALF_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD1_32i>
        {
        public:
            typedef int32_t      SCALAR_T;
            typedef SIMD1_32u    UINT_VEC_T;
            typedef SIMDMask1    MASK_T;
            typedef SIMDSwizzle1 SWIZZLE_T;

            typedef SIMD2_32i    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD2_32i>
        {
        public:
            typedef int32_t      SCALAR_T;
            typedef SIMD2_32u    UINT_VEC_T;
            typedef SIMDMask2    MASK_T;
            typedef SIMDSwizzle2 SWIZZLE_T;

            typedef SIMD1_32i    HALF_LEN_VEC_T;
            typedef SIMD4_32i    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD4_32i>
        {
        public:
            typedef int32_t      SCALAR_T;
            typedef SIMD4_32u    UINT_VEC_T;
            typedef SIMDMask4    MASK_T;
            typedef SIMDSwizzle4 SWIZZLE_T;

            typedef SIMD2_32i    HALF_LEN_VEC_T;
            typedef SIMD8_32i    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD8_32i>
        {
        public:
            typedef int32_t      SCALAR_T;
            typedef SIMD8_32u    UINT_VEC_T;
            typedef SIMDMask8    MASK_T;
            typedef SIMDSwizzle8 SWIZZLE_T;

            typedef SIMD4_32i    HALF_LEN_VEC_T;
            typedef SIMD16_32i   DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD16_32i>
        {
        public:
            typedef int32_t       SCALAR_T;
            typedef SIMD16_32u    UINT_VEC_T;
            typedef SIMDMask16    MASK_T;
            typedef SIMDSwizzle16 SWIZZLE_T;

            typedef SIMD8_32i     HALF_LEN_VEC_T;
            typedef SIMD32_32i    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD32_32i>
        {
        public:
            typedef int32_t       SCALAR_T;
            typedef SIMD32_32u    UINT_VEC_T;
            typedef SIMDMask32    MASK_T;
            typedef SIMDSwizzle32 SWIZZLE_T;

            typedef SIMD16_32i    HALF_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD1_64i>
        {
        public:
            typedef int64_t      SCALAR_T;
            typedef SIMD1_64u    UINT_VEC_T;
            typedef SIMDMask1    MASK_T;
            typedef SIMDSwizzle1 SWIZZLE_T;

            typedef SIMD2_64i    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD2_64i>
        {
        public:
            typedef int64_t      SCALAR_T;
            typedef SIMD2_64u    UINT_VEC_T;
            typedef SIMDMask2    MASK_T;
            typedef SIMDSwizzle2 SWIZZLE_T;

            typedef SIMD1_64i    HALF_LEN_VEC_T;
            typedef SIMD4_64i    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD4_64i>
        {
        public:
            typedef int64_t      SCALAR_T;
            typedef SIMD4_64u    UINT_VEC_T;
            typedef SIMDMask4    MASK_T;
            typedef SIMDSwizzle4 SWIZZLE_T;

            typedef SIMD2_64i    HALF_LEN_VEC_T;
            typedef SIMD8_64i    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD8_64i>
        {
        public:
            typedef int64_t      SCALAR_T;
            typedef SIMD8_64u    UINT_VEC_T;
            typedef SIMDMask8    MASK_T;
            typedef SIMDSwizzle8 SWIZZLE_T;

            typedef SIMD4_64i    HALF_LEN_VEC_T;
            typedef SIMD16_64i   DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD16_64i>
        {
        public:
            typedef int64_t       SCALAR_T;
            typedef SIMD16_64u    UINT_VEC_T;
            typedef SIMDMask16    MASK_T;
            typedef SIMDSwizzle16 SWIZZLE_T;

            typedef SIMD8_64i     HALF_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD1_32f>
        {
        public:
            typedef float        SCALAR_T;
            typedef int32_t      SCALAR_INT_T;
            typedef uint32_t     SCALAR_UINT_T;
            typedef SIMD1_32u    UINT_VEC_T;
            typedef SIMD1_32i    INT_VEC_T;
            typedef SIMDMask1    MASK_T;
            typedef SIMDSwizzle1 SWIZZLE_T;

            typedef SIMD2_32f    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD2_32f>
        {
        public:
            typedef float        SCALAR_T;
            typedef int32_t      SCALAR_INT_T;
            typedef uint32_t     SCALAR_UINT_T;
            typedef SIMD2_32u    UINT_VEC_T;
            typedef SIMD2_32i    INT_VEC_T;
            typedef SIMDMask2    MASK_T;
            typedef SIMDSwizzle2 SWIZZLE_T;

            typedef SIMD1_32f    HALF_LEN_VEC_T;
            typedef SIMD4_32f    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD4_32f>
        {
        public:
            typedef float        SCALAR_T;
            typedef int32_t      SCALAR_INT_T;
            typedef uint32_t     SCALAR_UINT_T;
            typedef SIMD4_32u    UINT_VEC_T;
            typedef SIMD4_32i    INT_VEC_T;
            typedef SIMDMask4    MASK_T;
            typedef SIMDSwizzle4 SWIZZLE_T;

            typedef SIMD2_32f    HALF_LEN_VEC_T;
            typedef SIMD8_32f    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD8_32f>
        {
        public:
            typedef float        SCALAR_T;
            typedef int32_t      SCALAR_INT_T;
            typedef uint32_t     SCALAR_UINT_T;
            typedef SIMD8_32u    UINT_VEC_T;
            typedef SIMD8_32i    INT_VEC_T;
            typedef SIMDMask8    MASK_T;
            typedef SIMDSwizzle8 SWIZZLE_T;

            typedef SIMD4_32f    HALF_LEN_VEC_T;
            typedef SIMD16_32f   DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD16_32f>
        {
        public:
            typedef float         SCALAR_T;
            typedef int32_t       SCALAR_INT_T;
            typedef uint32_t      SCALAR_UINT_T;
            typedef SIMD16_32u    UINT_VEC_T;
            typedef SIMD16_32i    INT_VEC_T;
            typedef SIMDMask16    MASK_T;
            typedef SIMDSwizzle16 SWIZZLE_T;

            typedef SIMD8_32f     HALF_LEN_VEC_T;
            typedef SIMD32_32f    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD32_32f>
        {
        public:
            typedef float         SCALAR_T;
            typedef int32_t       SCALAR_INT_T;
            typedef uint32_t      SCALAR_UINT_T;
            typedef SIMD32_32u    UINT_VEC_T;
            typedef SIMD32_32i    INT_VEC_T;
            typedef SIMDMask32    MASK_T;
            typedef SIMDSwizzle32 SWIZZLE_T;

            typedef SIMD16_32f    HALF_LEN_VEC_T;
        };


        template<>
        class SIMDTraits<SIMD1_64f>
        {
        public:
            typedef double       SCALAR_T;
            typedef int64_t      SCALAR_INT_T;
            typedef uint64_t     SCALAR_UINT_T;
            typedef SIMD1_64u    UINT_VEC_T;
            typedef SIMD1_64i    INT_VEC_T;
            typedef SIMDMask1    MASK_T;
            typedef SIMDSwizzle1 SWIZZLE_T;

            typedef SIMD2_64f    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD2_64f>
        {
        public:
            typedef double       SCALAR_T;
            typedef int64_t      SCALAR_INT_T;
            typedef uint64_t     SCALAR_UINT_T;
            typedef SIMD2_64u    UINT_VEC_T;
            typedef SIMD2_64i    INT_VEC_T;
            typedef SIMDMask2    MASK_T;
            typedef SIMDSwizzle2 SWIZZLE_T;

            typedef SIMD1_64f    HALF_LEN_VEC_T;
            typedef SIMD4_64f    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD4_64f>
        {
        public:
            typedef double       SCALAR_T;
            typedef int64_t      SCALAR_INT_T;
            typedef uint64_t     SCALAR_UINT_T;
            typedef SIMD4_64u    UINT_VEC_T;
            typedef SIMD4_64i    INT_VEC_T;
            typedef SIMDMask4    MASK_T;
            typedef SIMDSwizzle4 SWIZZLE_T;

            typedef SIMD2_64f    HALF_LEN_VEC_T;
            typedef SIMD8_64f    DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD8_64f>
        {
        public:
            typedef double       SCALAR_T;
            typedef int64_t      SCALAR_INT_T;
            typedef uint64_t     SCALAR_UINT_T;
            typedef SIMD8_64u    UINT_VEC_T;
            typedef SIMD8_64i    INT_VEC_T;
            typedef SIMDMask8    MASK_T;
            typedef SIMDSwizzle8 SWIZZLE_T;

            typedef SIMD4_64f    HALF_LEN_VEC_T;
            typedef SIMD16_64f   DOUBLE_LEN_VEC_T;
        };

        template<>
        class SIMDTraits<SIMD16_64f>
        {
        public:
            typedef double        SCALAR_T;
            typedef int64_t       SCALAR_INT_T;
            typedef uint64_t      SCALAR_UINT_T;
            typedef SIMD16_64u    UINT_VEC_T;
            typedef SIMD16_64i    INT_VEC_T;
            typedef SIMDMask16    MASK_T;
            typedef SIMDSwizzle16 SWIZZLE_T;

            typedef SIMD8_64f     HALF_LEN_VEC_T;
        };

    }
}

#endif
