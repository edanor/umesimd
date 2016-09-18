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

#ifndef UME_SIMD_VEC_UINT_PROTOTYPE_H_
#define UME_SIMD_VEC_UINT_PROTOTYPE_H_

#include <type_traits>

#include "../../../UMESimdInterface.h"

#include "../UMESimdMask.h"
#include "../UMESimdSwizzle.h"

namespace UME {
namespace SIMD {
    // ********************************************************************************************
    // UNSIGNED INTEGER VECTORS
    // ********************************************************************************************
    template<typename VEC_TYPE, uint32_t VEC_LEN>
    struct SIMDVec_u_traits {
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };

    // 8b vectors
    template<>
    struct SIMDVec_u_traits<uint8_t, 1> {
        typedef NullType<1>       HALF_LEN_VEC_TYPE;
        typedef int8_t            SCALAR_INT_TYPE;
        typedef NullType<2>       SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<1>    MASK_TYPE;
        typedef SIMDSwizzle<1>    SWIZZLE_MASK_TYPE;
        typedef NullType<3>       SCALAR_UINT_LOWER_PRECISION;
        typedef uint16_t          SCALAR_UINT_HIGHER_PRECISION;
    };

    // 16b vectors
    template<>
    struct SIMDVec_u_traits<uint8_t, 2> {
        typedef SIMDVec_u<uint8_t, 1> HALF_LEN_VEC_TYPE;
        typedef int8_t                SCALAR_INT_TYPE;
        typedef NullType<1>           SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<2>        MASK_TYPE;
        typedef SIMDSwizzle<2>        SWIZZLE_MASK_TYPE;
        typedef NullType<2>           SCALAR_UINT_LOWER_PRECISION;
        typedef uint16_t              SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint16_t, 1> {
        typedef NullType<1>         HALF_LEN_VEC_TYPE;
        typedef int16_t             SCALAR_INT_TYPE;
        typedef NullType<2>         SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<1>      MASK_TYPE;
        typedef SIMDSwizzle<1>      SWIZZLE_MASK_TYPE;
        typedef uint8_t             SCALAR_UINT_LOWER_PRECISION;
        typedef uint32_t            SCALAR_UINT_HIGHER_PRECISION;
    };

    // 32b vectors
    template<>
    struct SIMDVec_u_traits<uint8_t, 4> {
        typedef SIMDVec_u<uint8_t, 2> HALF_LEN_VEC_TYPE;
        typedef int8_t                SCALAR_INT_TYPE;
        typedef NullType<1>           SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<4>        MASK_TYPE;
        typedef SIMDSwizzle<4>        SWIZZLE_MASK_TYPE;
        typedef NullType<2>           SCALAR_UINT_LOWER_PRECISION;
        typedef uint16_t              SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint16_t, 2> {
        typedef SIMDVec_u<uint16_t, 1> HALF_LEN_VEC_TYPE;
        typedef int16_t                SCALAR_INT_TYPE;
        typedef NullType<1>            SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<2>         MASK_TYPE;
        typedef SIMDSwizzle<2>         SWIZZLE_MASK_TYPE;
        typedef uint8_t                SCALAR_UINT_LOWER_PRECISION;
        typedef uint32_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint32_t, 1> {
        typedef NullType<1>         HALF_LEN_VEC_TYPE;
        typedef int32_t             SCALAR_INT_TYPE;
        typedef float               SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<1>      MASK_TYPE;
        typedef SIMDSwizzle<1>      SWIZZLE_MASK_TYPE;
        typedef uint16_t            SCALAR_UINT_LOWER_PRECISION;
        typedef uint64_t            SCALAR_UINT_HIGHER_PRECISION;
    };

    // 64b vectors
    template<>
    struct SIMDVec_u_traits<uint8_t, 8> {
        typedef SIMDVec_u<uint8_t, 4> HALF_LEN_VEC_TYPE;
        typedef int8_t                SCALAR_INT_TYPE;
        typedef NullType<1>           SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<8>        MASK_TYPE;
        typedef SIMDSwizzle<8>        SWIZZLE_MASK_TYPE;
        typedef NullType<2>           SCALAR_UINT_LOWER_PRECISION;
        typedef uint16_t              SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint16_t, 4> {
        typedef SIMDVec_u<uint16_t, 2> HALF_LEN_VEC_TYPE;
        typedef int16_t                SCALAR_INT_TYPE;
        typedef NullType<1>            SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<4>         MASK_TYPE;
        typedef SIMDSwizzle<4>         SWIZZLE_MASK_TYPE;
        typedef uint8_t                SCALAR_UINT_LOWER_PRECISION;
        typedef uint32_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint32_t, 2> {
        typedef SIMDVec_u<uint32_t, 1> HALF_LEN_VEC_TYPE;
        typedef int32_t                SCALAR_INT_TYPE;
        typedef float                  SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<2>         MASK_TYPE;
        typedef SIMDSwizzle<2>         SWIZZLE_MASK_TYPE;
        typedef uint16_t               SCALAR_UINT_LOWER_PRECISION;
        typedef uint64_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint64_t, 1> {
        typedef NullType<1>         HALF_LEN_VEC_TYPE;
        typedef int64_t             SCALAR_INT_TYPE;
        typedef double              SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<1>      MASK_TYPE;
        typedef SIMDSwizzle<1>      SWIZZLE_MASK_TYPE;
        typedef uint32_t            SCALAR_UINT_LOWER_PRECISION;
        typedef NullType<2>         SCALAR_UINT_HIGHER_PRECISION;
    };

    // 128b vectors
    template<>
    struct SIMDVec_u_traits<uint8_t, 16> {
        typedef SIMDVec_u<uint8_t, 8> HALF_LEN_VEC_TYPE;
        typedef int8_t                SCALAR_INT_TYPE;
        typedef NullType<1>           SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<16>       MASK_TYPE;
        typedef SIMDSwizzle<16>       SWIZZLE_MASK_TYPE;
        typedef NullType<2>           SCALAR_UINT_LOWER_PRECISION;
        typedef uint16_t              SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint16_t, 8> {
        typedef SIMDVec_u<uint16_t, 4> HALF_LEN_VEC_TYPE;
        typedef int16_t                SCALAR_INT_TYPE;
        typedef NullType<1>            SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<8>         MASK_TYPE;
        typedef SIMDSwizzle<8>         SWIZZLE_MASK_TYPE;
        typedef uint8_t                SCALAR_UINT_LOWER_PRECISION;
        typedef uint32_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint32_t, 4> {
        typedef SIMDVec_u<uint32_t, 2> HALF_LEN_VEC_TYPE;
        typedef int32_t                SCALAR_INT_TYPE;
        typedef float                  SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<4>         MASK_TYPE;
        typedef SIMDSwizzle<4>         SWIZZLE_MASK_TYPE;
        typedef uint16_t               SCALAR_UINT_LOWER_PRECISION;
        typedef uint64_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint64_t, 2> {
        typedef SIMDVec_u<uint64_t, 1> HALF_LEN_VEC_TYPE;
        typedef int64_t                SCALAR_INT_TYPE;
        typedef double                 SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<2>         MASK_TYPE;
        typedef SIMDSwizzle<2>         SWIZZLE_MASK_TYPE;
        typedef uint32_t               SCALAR_UINT_LOWER_PRECISION;
        typedef NullType<1>            SCALAR_UINT_HIGHER_PRECISION;
    };

    // 256b vectors
    template<>
    struct SIMDVec_u_traits<uint8_t, 32> {
        typedef SIMDVec_u<uint8_t, 16> HALF_LEN_VEC_TYPE;
        typedef int8_t                 SCALAR_INT_TYPE;
        typedef NullType<1>            SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<32>        MASK_TYPE;
        typedef SIMDSwizzle<32>        SWIZZLE_MASK_TYPE;
        typedef NullType<2>            SCALAR_UINT_LOWER_PRECISION;
        typedef uint16_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint16_t, 16> {
        typedef SIMDVec_u<uint16_t, 8> HALF_LEN_VEC_TYPE;
        typedef int16_t                SCALAR_INT_TYPE;
        typedef NullType<1>            SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<16>        MASK_TYPE;
        typedef SIMDSwizzle<16>        SWIZZLE_MASK_TYPE;
        typedef uint8_t                SCALAR_UINT_LOWER_PRECISION;
        typedef uint32_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint32_t, 8> {
        typedef SIMDVec_u<uint32_t, 4> HALF_LEN_VEC_TYPE;
        typedef int32_t                SCALAR_INT_TYPE;
        typedef float                  SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<8>         MASK_TYPE;
        typedef SIMDSwizzle<8>         SWIZZLE_MASK_TYPE;
        typedef uint16_t               SCALAR_UINT_LOWER_PRECISION;
        typedef uint64_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint64_t, 4> {
        typedef SIMDVec_u<uint64_t, 2> HALF_LEN_VEC_TYPE;
        typedef int64_t                SCALAR_INT_TYPE;
        typedef double                 SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<4>         MASK_TYPE;
        typedef SIMDSwizzle<4>         SWIZZLE_MASK_TYPE;
        typedef uint32_t               SCALAR_UINT_LOWER_PRECISION;
        typedef NullType<2>            SCALAR_UINT_HIGHER_PRECISION;
    };

    // 512b vectors
    template<>
    struct SIMDVec_u_traits<uint8_t, 64> {
        typedef SIMDVec_u<uint8_t, 32> HALF_LEN_VEC_TYPE;
        typedef int8_t                 SCALAR_INT_TYPE;
        typedef NullType<1>            SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<64>        MASK_TYPE;
        typedef SIMDSwizzle<64>        SWIZZLE_MASK_TYPE;
        typedef NullType<2>            SCALAR_UINT_LOWER_PRECISION;
        typedef uint16_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint16_t, 32> {
        typedef SIMDVec_u<uint16_t, 16> HALF_LEN_VEC_TYPE;
        typedef int16_t                 SCALAR_INT_TYPE;
        typedef NullType<1>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<32>         MASK_TYPE;
        typedef SIMDSwizzle<32>         SWIZZLE_MASK_TYPE;
        typedef uint8_t                 SCALAR_UINT_LOWER_PRECISION;
        typedef uint32_t                SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint32_t, 16> {
        typedef SIMDVec_u<uint32_t, 8> HALF_LEN_VEC_TYPE;
        typedef int32_t                SCALAR_INT_TYPE;
        typedef float                  SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<16>        MASK_TYPE;
        typedef SIMDSwizzle<16>        SWIZZLE_MASK_TYPE;
        typedef uint16_t               SCALAR_UINT_LOWER_PRECISION;
        typedef uint64_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint64_t, 8> {
        typedef SIMDVec_u<uint64_t, 4> HALF_LEN_VEC_TYPE;
        typedef int64_t                SCALAR_INT_TYPE;
        typedef double                 SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<8>         MASK_TYPE;
        typedef SIMDSwizzle<8>         SWIZZLE_MASK_TYPE;
        typedef uint32_t               SCALAR_UINT_LOWER_PRECISION;
        typedef NullType<1>            SCALAR_UINT_HIGHER_PRECISION;
    };

    // 1024b vectors
    template<>
    struct SIMDVec_u_traits<uint8_t, 128> {
        typedef SIMDVec_u<uint8_t, 64> HALF_LEN_VEC_TYPE;
        typedef int8_t                 SCALAR_INT_TYPE;
        typedef NullType<1>            SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<128>       MASK_TYPE;
        typedef SIMDSwizzle<128>       SWIZZLE_MASK_TYPE;
        typedef NullType<2>            SCALAR_UINT_LOWER_PRECISION;
        typedef NullType<3>            SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint16_t, 64> {
        typedef SIMDVec_u<uint16_t, 32> HALF_LEN_VEC_TYPE;
        typedef int16_t                 SCALAR_INT_TYPE;
        typedef NullType<1>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<64>         MASK_TYPE;
        typedef SIMDSwizzle<64>         SWIZZLE_MASK_TYPE;
        typedef uint8_t                 SCALAR_UINT_LOWER_PRECISION;
        typedef NullType<1>             SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint32_t, 32> {
        typedef SIMDVec_u<uint32_t, 16> HALF_LEN_VEC_TYPE;
        typedef int32_t                 SCALAR_INT_TYPE;
        typedef float                   SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<32>         MASK_TYPE;
        typedef SIMDSwizzle<32>         SWIZZLE_MASK_TYPE;
        typedef uint16_t                SCALAR_UINT_LOWER_PRECISION;
        typedef NullType<1>             SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint64_t, 16> {
        typedef SIMDVec_u<uint64_t, 8> HALF_LEN_VEC_TYPE;
        typedef int64_t                SCALAR_INT_TYPE;
        typedef double                 SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<16>        MASK_TYPE;
        typedef SIMDSwizzle<16>        SWIZZLE_MASK_TYPE;
        typedef uint32_t               SCALAR_UINT_LOWER_PRECISION;
        typedef NullType<1>            SCALAR_UINT_HIGHER_PRECISION;
    };

    // ***************************************************************************
    // *
    // *    Implementation of unsigned integer SIMDx_8u, SIMDx_16u, SIMDx_32u, 
    // *    and SIMDx_64u.
    // *
    // *    This implementation uses scalar emulation available through to 
    // *    SIMDVecUnsignedInterface.
    // *
    // ***************************************************************************
    template<typename SCALAR_UINT_TYPE, uint32_t VEC_LEN>
    class SIMDVec_u :
        public SIMDVecUnsignedInterface<
            SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN>,
            SCALAR_UINT_TYPE,
            VEC_LEN,
            typename SIMDVec_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::MASK_TYPE,
            typename SIMDVec_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::SWIZZLE_MASK_TYPE>,
        public SIMDVecPackableInterface<
            SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN>,        
            typename SIMDVec_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::HALF_LEN_VEC_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_UINT_TYPE, VEC_LEN>   VEC_EMU_REG;

        typedef typename SIMDVec_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::SCALAR_INT_TYPE    SCALAR_INT_TYPE;
        typedef typename SIMDVec_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::SCALAR_FLOAT_TYPE  SCALAR_FLOAT_TYPE;
        typedef typename SIMDVec_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::MASK_TYPE          MASK_TYPE;

        typedef typename SIMDVec_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::SCALAR_UINT_LOWER_PRECISION  SCALAR_UINT_LOWER_PRECISION;
        typedef typename SIMDVec_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::SCALAR_UINT_HIGHER_PRECISION SCALAR_UINT_HIGHER_PRECISION;

        // Conversion operators require access to private members.
        friend class SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN>;
        friend class SIMDVec_f<SCALAR_FLOAT_TYPE, VEC_LEN>;

    public:
        constexpr static uint32_t alignment() { return VEC_LEN*sizeof(SCALAR_UINT_TYPE); }

    private:
        // This is the only data member and it is a low level representation of vector register.
        alignas(alignment()) SCALAR_UINT_TYPE mVec[VEC_LEN];

    public:
        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_u() : mVec() {};

        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_u(SCALAR_UINT_TYPE x) {
            SCALAR_UINT_TYPE *local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for (int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = x;
            }
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        inline SIMDVec_u(
            T i, 
            typename std::enable_if< (std::is_integral<T>::value) && 
                                    !std::is_same<T, SCALAR_UINT_TYPE>::value,
                                    void*>::type = nullptr)
        : SIMDVec_u(static_cast<SCALAR_UINT_TYPE>(i)) {}

        // LOAD-CONSTR - Construct by loading from memory
        UME_FORCE_INLINE explicit SIMDVec_u(SCALAR_UINT_TYPE const * p) { this->load(p); }

        UME_FORCE_INLINE SIMDVec_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1) {
            insert(0, i0);  insert(1, i1);
        }

        UME_FORCE_INLINE SIMDVec_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3) {
            insert(0, i0);  insert(1, i1);  insert(2, i2);  insert(3, i3);
        }

        UME_FORCE_INLINE SIMDVec_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7)
        {
            insert(0, i0);  insert(1, i1);  insert(2, i2);  insert(3, i3);
            insert(4, i4);  insert(5, i5);  insert(6, i6);  insert(7, i7);
        }

        UME_FORCE_INLINE SIMDVec_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7,
            SCALAR_UINT_TYPE i8, SCALAR_UINT_TYPE i9, SCALAR_UINT_TYPE i10, SCALAR_UINT_TYPE i11, SCALAR_UINT_TYPE i12, SCALAR_UINT_TYPE i13, SCALAR_UINT_TYPE i14, SCALAR_UINT_TYPE i15)
        {
            insert(0, i0);    insert(1, i1);    insert(2, i2);    insert(3, i3);
            insert(4, i4);    insert(5, i5);    insert(6, i6);    insert(7, i7);
            insert(8, i8);    insert(9, i9);    insert(10, i10);  insert(11, i11);
            insert(12, i12);  insert(13, i13);  insert(14, i14);  insert(15, i15);
        }

        UME_FORCE_INLINE SIMDVec_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7,
            SCALAR_UINT_TYPE i8, SCALAR_UINT_TYPE i9, SCALAR_UINT_TYPE i10, SCALAR_UINT_TYPE i11, SCALAR_UINT_TYPE i12, SCALAR_UINT_TYPE i13, SCALAR_UINT_TYPE i14, SCALAR_UINT_TYPE i15,
            SCALAR_UINT_TYPE i16, SCALAR_UINT_TYPE i17, SCALAR_UINT_TYPE i18, SCALAR_UINT_TYPE i19, SCALAR_UINT_TYPE i20, SCALAR_UINT_TYPE i21, SCALAR_UINT_TYPE i22, SCALAR_UINT_TYPE i23,
            SCALAR_UINT_TYPE i24, SCALAR_UINT_TYPE i25, SCALAR_UINT_TYPE i26, SCALAR_UINT_TYPE i27, SCALAR_UINT_TYPE i28, SCALAR_UINT_TYPE i29, SCALAR_UINT_TYPE i30, SCALAR_UINT_TYPE i31)
        {
            insert(0, i0);    insert(1, i1);    insert(2, i2);    insert(3, i3);
            insert(4, i4);    insert(5, i5);    insert(6, i6);    insert(7, i7);
            insert(8, i8);    insert(9, i9);    insert(10, i10);  insert(11, i11);
            insert(12, i12);  insert(13, i13);  insert(14, i14);  insert(15, i15);
            insert(16, i16);  insert(17, i17);  insert(18, i18);  insert(19, i19);
            insert(20, i20);  insert(21, i21);  insert(22, i22);  insert(23, i23);
            insert(24, i24);  insert(25, i25);  insert(26, i26);  insert(27, i27);
            insert(28, i28);  insert(29, i29);  insert(30, i30);  insert(31, i31);
        }

        // EXTRACT
        UME_FORCE_INLINE SCALAR_UINT_TYPE extract(uint32_t index) const {
            return mVec[index];
        }
        UME_FORCE_INLINE SCALAR_UINT_TYPE operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_u & insert(uint32_t index, SCALAR_UINT_TYPE value) {
            mVec[index] = value;
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_u, SCALAR_UINT_TYPE> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, SCALAR_UINT_TYPE>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, SCALAR_UINT_TYPE, MASK_TYPE> operator() (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVec_u, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, SCALAR_UINT_TYPE, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVec_u, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<SIMDVec_u &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVec_u const & src) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_src_ptr = &src.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = local_src_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (SIMDVec_u const & b) {
            return this->assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & src) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_src_ptr = &src.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_ptr[i] = local_src_ptr[i];
            }
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (SCALAR_UINT_TYPE b) {
            return this->assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_ptr[i] = b;
            }
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_u & load(SCALAR_UINT_TYPE const *p) {
            SCALAR_UINT_TYPE *local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const *local_p_ptr = &p[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = local_p_ptr[i];
            }
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_u & load(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE const *p) {
            SCALAR_UINT_TYPE *local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const *local_p_ptr = &p[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_ptr[i] = local_p_ptr[i];
            }
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_u & loada(SCALAR_UINT_TYPE const *p) {
            SCALAR_UINT_TYPE *local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const *local_p_ptr = &p[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = local_p_ptr[i];
            }
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_u & loada(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE const *p) {
            SCALAR_UINT_TYPE *local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const *local_p_ptr = &p[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_ptr[i] = local_p_ptr[i];
            }
            return *this;
        }
        // STORE
        UME_FORCE_INLINE SCALAR_UINT_TYPE* store(SCALAR_UINT_TYPE* p) const {
            SCALAR_UINT_TYPE const *local_ptr = &mVec[0];
            SCALAR_UINT_TYPE *local_p_ptr = &p[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_p_ptr[i] = local_ptr[i];
            }
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE SCALAR_UINT_TYPE* store(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE* p) const {
            SCALAR_UINT_TYPE const *local_ptr = &mVec[0];
            SCALAR_UINT_TYPE *local_p_ptr = &p[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_p_ptr[i] = local_ptr[i];
            }
            return p;
        }
        // STOREA
        UME_FORCE_INLINE SCALAR_UINT_TYPE* storea(SCALAR_UINT_TYPE* p) const {
            SCALAR_UINT_TYPE const *local_ptr = &mVec[0];
            SCALAR_UINT_TYPE *local_p_ptr = &p[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_p_ptr[i] = local_ptr[i];
            }
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE SCALAR_UINT_TYPE* storea(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE* p) const {
            SCALAR_UINT_TYPE const *local_ptr = &mVec[0];
            SCALAR_UINT_TYPE *local_p_ptr = &p[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_p_ptr[i] = local_ptr[i];
            }
            return p;
        }
        // BLENDV
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE *retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const *local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const *local_b_ptr = &b.mVec[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) retval_ptr[i] = local_b_ptr[i];
                else retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE const *local_ptr = &mVec[0];
            SCALAR_UINT_TYPE *retval_ptr = &retval.mVec[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) retval_ptr[i] = b;
                else retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] + local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (SIMDVec_u const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] + local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_u add(SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] + b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (SCALAR_UINT_TYPE b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] + b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] += local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (SIMDVec_u const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] += local_b_ptr[i];
            }
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] += b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (SCALAR_UINT_TYPE b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] += b;
            }
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
        UME_FORCE_INLINE SIMDVec_u postinc() {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i]++;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_u postinc(SIMDVecMask<VEC_LEN> const & mask) {
             SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i]++;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc() {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                ++local_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) ++local_ptr[i];
            }
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] - local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator- (SIMDVec_u const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] - local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_u sub(SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] - b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator- (SCALAR_UINT_TYPE b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] - b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] -= local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (SIMDVec_u const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] -= local_b_ptr[i];
            }
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] -= b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (SCALAR_UINT_TYPE b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] -= b;
            }
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
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_b_ptr[i] - local_ptr[i];
            }
            return retval;
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_b_ptr[i] - local_ptr[i];
                else local_retval_ptr[i] = local_b_ptr[i];
            }
            return retval;
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = b - local_ptr[i];
            }
            return retval;
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = b - local_ptr[i];
                else local_retval_ptr[i] = b;
            }
            return retval;
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] = local_b_ptr[i] - local_ptr[i];
            }
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = local_b_ptr[i] - local_ptr[i];
                else local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = b - local_ptr[i];
            }
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = b - local_ptr[i];
                else local_ptr[i] = b;
            }
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec() {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i]--;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec(SIMDVecMask<VEC_LEN> const & mask) {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i]--;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec() {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                --local_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) --local_ptr[i];
            }
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator* (SIMDVec_u const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_u mul(SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] * b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator* (SCALAR_UINT_TYPE b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] * b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] *= local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (SIMDVec_u const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] *= local_b_ptr[i];
            }
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_u & mula(SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] *= b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (SCALAR_UINT_TYPE b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] *= b;
            }
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_u div(SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] / local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator/ (SIMDVec_u const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_u div(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] / local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_u div(SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] / b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator/ (SCALAR_UINT_TYPE b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_u div(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] / b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_u & diva(SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] /= local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator/= (SIMDVec_u const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_u & diva(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] /= local_b_ptr[i];
            }
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_u & diva(SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] /= b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator/= (SCALAR_UINT_TYPE b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_u & diva(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] /= b;
            }
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_u rcp() const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = SCALAR_UINT_TYPE(1.0f) / local_ptr[i];
            }
            return retval;
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_u rcp(SIMDVecMask<VEC_LEN> const & mask) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = SCALAR_UINT_TYPE(1.0f) / local_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_u rcp(SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = b / local_ptr[i];
            }
            return retval;
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_u rcp(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = b / local_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_u & rcpa() {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] = SCALAR_UINT_TYPE(1.0f) / local_ptr[i];
            }
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_u & rcpa(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = SCALAR_UINT_TYPE(1.0f) / local_ptr[i];
            }
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_u & rcpa(SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = b / local_ptr[i];
            }
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_u & rcpa(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = b / local_ptr[i];
            }
            return *this;
        }
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpeq(SIMDVec_u const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] == local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator== (SIMDVec_u const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpeq(SCALAR_UINT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] == b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator== (SCALAR_UINT_TYPE b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpne(SIMDVec_u const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] != local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator!= (SIMDVec_u const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpne(SCALAR_UINT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] != b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator!= (SCALAR_UINT_TYPE b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpgt(SIMDVec_u const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] > local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator> (SIMDVec_u const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpgt(SCALAR_UINT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] > b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator> (SCALAR_UINT_TYPE b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmplt(SIMDVec_u const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] < local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator< (SIMDVec_u const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmplt(SCALAR_UINT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] < b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator< (SCALAR_UINT_TYPE b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpge(SIMDVec_u const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] >= local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator>= (SIMDVec_u const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpge(SCALAR_UINT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] >= b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator>= (SCALAR_UINT_TYPE b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmple(SIMDVec_u const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] <= local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator<= (SIMDVec_u const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmple(SCALAR_UINT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] <= b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator<= (SCALAR_UINT_TYPE b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_u const & b) const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool local_mask_ptr[VEC_LEN];
            bool retval = true;
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_mask_ptr[i] = local_ptr[i] == local_b_ptr[i];
            }
            #pragma omp simd reduction(&&:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval && local_mask_ptr[i];
            }
            return retval;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(SCALAR_UINT_TYPE b) const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool local_mask_ptr[VEC_LEN];
            bool retval = true;
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_mask_ptr[i] = local_ptr[i] == b;
            }
            #pragma omp simd reduction(&&:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval && local_mask_ptr[i];
            }
            return retval;
        }
        // UNIQUE
        // TODO

        // HADD
        UME_FORCE_INLINE SCALAR_UINT_TYPE hadd() const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE retval = SCALAR_UINT_TYPE(0.0f);
            #pragma omp simd reduction(+:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval + local_ptr[i];
            }
            return retval;
        }
        // MHADD
        UME_FORCE_INLINE SCALAR_UINT_TYPE hadd(SIMDVecMask<VEC_LEN> const & mask) const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_UINT_TYPE retval = SCALAR_UINT_TYPE(0.0f);
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_UINT_TYPE(0.0f);
            }
            #pragma omp simd reduction(+:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval + masked_copy[i];
            }
            return retval;
        }
        // HADDS
        UME_FORCE_INLINE SCALAR_UINT_TYPE hadd(SCALAR_UINT_TYPE b) const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE retval = b;
            #pragma omp simd reduction(+:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval + local_ptr[i];
            }
            return retval;
        }
        // MHADDS
        UME_FORCE_INLINE SCALAR_UINT_TYPE hadd(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_UINT_TYPE retval = b;
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_UINT_TYPE(0.0f);
            }
            #pragma omp simd reduction(+:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval + masked_copy[i];
            }
            return retval;
        }
        // HMUL
        UME_FORCE_INLINE SCALAR_UINT_TYPE hmul() const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE retval = SCALAR_UINT_TYPE(1.0f);
            #pragma omp simd reduction(*:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval * local_ptr[i];
            }
            return retval;        }
        // MHMUL
        UME_FORCE_INLINE SCALAR_UINT_TYPE hmul(SIMDVecMask<VEC_LEN> const & mask) const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_UINT_TYPE retval = SCALAR_UINT_TYPE(1.0f);
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_UINT_TYPE(1.0f);
            }
            #pragma omp simd reduction(*:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval * masked_copy[i];
            }
            return retval;
        }
        // HMULS
        UME_FORCE_INLINE SCALAR_UINT_TYPE hmul(SCALAR_UINT_TYPE b) const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE retval = b;
            #pragma omp simd reduction(*:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval * local_ptr[i];
            }
            return retval;
        }
        // MHMULS
        UME_FORCE_INLINE SCALAR_UINT_TYPE hmul(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_UINT_TYPE retval = b;
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_UINT_TYPE(1.0f);
            }
            #pragma omp simd reduction(*:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval * masked_copy[i];
            }
            return retval;
        }
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_UINT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i] + local_c_ptr[i];
            }
            return retval;
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_UINT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i] + local_c_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_UINT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i] - local_c_ptr[i];
            }
            return retval;
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_UINT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i] - local_c_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_UINT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = (local_ptr[i] + local_b_ptr[i]) * local_c_ptr[i];
            }
            return retval;
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_UINT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = (local_ptr[i] + local_b_ptr[i]) * local_c_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_UINT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = (local_ptr[i] - local_b_ptr[i]) * local_c_ptr[i];
            }
            return retval;
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_UINT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = (local_ptr[i] - local_b_ptr[i]) * local_c_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] > local_b_ptr[i]) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = local_b_ptr[i];
            }
            return retval;
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] > local_b_ptr[i];
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_retval_ptr[i] = local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_u max(SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] > b) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = b;
            }
            return retval;
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] > b;
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_retval_ptr[i] = b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] <= local_b_ptr[i]) local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] > local_b_ptr[i];
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] <= b) local_ptr[i] = b;
            }
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] > b;
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_ptr[i] = b;
            }
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] < local_b_ptr[i]) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = local_b_ptr[i];
            }
            return retval;
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] < local_b_ptr[i];
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_retval_ptr[i] = local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_u min(SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] < b) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = b;
            }
            return retval;
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] < b;
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_retval_ptr[i] = b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] > local_b_ptr[i]) local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] < local_b_ptr[i];
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_u & mina(SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] > b) local_ptr[i] = b;
            }
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] < b;
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_ptr[i] = b;
            }
            return *this;
        }

        // HMAX
        // MHMAX
        // IMAX
        // MIMAX
        // HMIN
        // MHMIN
        // IMIN
        // MIMIN

        // BANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] & local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator& (SIMDVec_u const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] & local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_u band(int32_t b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] & b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator& (int32_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<VEC_LEN> const & mask, int32_t b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] & b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] &= local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (SIMDVec_u const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] &= local_b_ptr[i];
            }
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(int32_t b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] &= b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<VEC_LEN> const & mask, int32_t b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] &= b;
            }
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] | local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator| (SIMDVec_u const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] | local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_u bor(int32_t b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] | b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator| (int32_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<VEC_LEN> const & mask, int32_t b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] | b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] |= local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (SIMDVec_u const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] |= local_b_ptr[i];
            }
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_u & bora(int32_t b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] |= b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (int32_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<VEC_LEN> const & mask, int32_t b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] |= b;
            }
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] ^ local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (SIMDVec_u const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] ^ local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_u bxor(int32_t b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] ^ b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (int32_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<VEC_LEN> const & mask, int32_t b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] ^ b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] ^= local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (SIMDVec_u const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] ^= local_b_ptr[i];
            }
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(int32_t b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] ^= b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (int32_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<VEC_LEN> const & mask, int32_t b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] ^= b;
            }
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_u bnot() const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = ~local_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_u bnot(SIMDVecMask<VEC_LEN> const & mask) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = ~local_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_u & bnota() {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = ~local_ptr[i];
            }
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_u & bnota(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = ~local_ptr[i];
            }
            return *this;
        }
        // HBAND
        UME_FORCE_INLINE SCALAR_UINT_TYPE hband() const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE retval = SCALAR_UINT_TYPE(0xFFFFFFFFFFFFFFFF);
            #pragma omp simd reduction(&:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval & local_ptr[i];
            }
            return retval;
        }
        // MHBAND
        UME_FORCE_INLINE SCALAR_UINT_TYPE hband(SIMDVecMask<VEC_LEN> const & mask) const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_UINT_TYPE retval = SCALAR_UINT_TYPE(0xFFFFFFFFFFFFFFFF);
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_UINT_TYPE(0xFFFFFFFFFFFFFFFF);
            }
            #pragma omp simd reduction(&:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval & masked_copy[i];
            }
            return retval;
        }
        // HBANDS
        UME_FORCE_INLINE SCALAR_UINT_TYPE hband(SCALAR_UINT_TYPE b) const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE retval = b;
            #pragma omp simd reduction(&:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval & local_ptr[i];
            }
            return retval;
        }
        // MHBANDS
        UME_FORCE_INLINE SCALAR_UINT_TYPE hband(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_UINT_TYPE retval = b;
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_UINT_TYPE(0xFFFFFFFFFFFFFFFF);
            }
            #pragma omp simd reduction(&:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval & masked_copy[i];
            }
            return retval;
        }
        // HBOR
        UME_FORCE_INLINE SCALAR_UINT_TYPE hbor() const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE retval = SCALAR_UINT_TYPE(0);
            #pragma omp simd reduction(|:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval | local_ptr[i];
            }
            return retval;
        }
        // MHBOR
        UME_FORCE_INLINE SCALAR_UINT_TYPE hbor(SIMDVecMask<VEC_LEN> const & mask) const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_UINT_TYPE retval = SCALAR_UINT_TYPE(0);
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_UINT_TYPE(0);
            }
            #pragma omp simd reduction(&:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval | masked_copy[i];
            }
            return retval;
        }
        // HBORS
        UME_FORCE_INLINE SCALAR_UINT_TYPE hbor(SCALAR_UINT_TYPE b) const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE retval = b;
            #pragma omp simd reduction(|:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval | local_ptr[i];
            }
            return retval;
        }
        // MHBORS
        UME_FORCE_INLINE SCALAR_UINT_TYPE hbor(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_UINT_TYPE retval = b;
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_UINT_TYPE(0);
            }
            #pragma omp simd reduction(|:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval | masked_copy[i];
            }
            return retval;
        }
        // HBXOR
        UME_FORCE_INLINE SCALAR_UINT_TYPE hbxor() const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE retval = SCALAR_UINT_TYPE(0);
            #pragma omp simd reduction(^:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval ^ local_ptr[i];
            }
            return retval;
        }
        // MHBXOR
        UME_FORCE_INLINE SCALAR_UINT_TYPE hbxor(SIMDVecMask<VEC_LEN> const & mask) const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_UINT_TYPE retval = SCALAR_UINT_TYPE(0);
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_UINT_TYPE(0);
            }
            #pragma omp simd reduction(^:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval ^ masked_copy[i];
            }
            return retval;
        }
        // HBXORS
        UME_FORCE_INLINE SCALAR_UINT_TYPE hbxor(SCALAR_UINT_TYPE b) const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE retval = b;
            #pragma omp simd reduction(^:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval ^ local_ptr[i];
            }
            return retval;
        }
        // MHBXORS
        UME_FORCE_INLINE SCALAR_UINT_TYPE hbxor(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_UINT_TYPE retval = b;
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_UINT_TYPE(0);
            }
            #pragma omp simd reduction(^:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval ^ masked_copy[i];
            }
            return retval;
        }
        // REMV
        UME_FORCE_INLINE SIMDVec_u rem(SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] % local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator% (SIMDVec_u const & b) const {
            return rem(b);
        }
        // MREMV
        UME_FORCE_INLINE SIMDVec_u rem(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] % local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // REMS
        UME_FORCE_INLINE SIMDVec_u rem(SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] % b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_u operator% (SCALAR_UINT_TYPE b) const {
            return rem(b);
        }
        // MREMS
        UME_FORCE_INLINE SIMDVec_u rem(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] % b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // REMVA
        UME_FORCE_INLINE SIMDVec_u & rema(SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] %= local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator%= (SIMDVec_u const & b) {
            return rema(b);
        }
        // MREMVA
        UME_FORCE_INLINE SIMDVec_u & rema(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] %= local_b_ptr[i];
            }
            return *this;
        }
        // REMSA
        UME_FORCE_INLINE SIMDVec_u & rema(SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] %= b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator%= (SCALAR_UINT_TYPE b) {
            return rema(b);
        }
        // MREMSA
        UME_FORCE_INLINE SIMDVec_u & rema(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] %= b;
            }
            return *this;
        }
        // LANDV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> land(SIMDVec_u const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            bool * local_retval_ptr = &retval.mMask[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] && local_b_ptr[i];
            }
            return retval;
        }
        // LANDS
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> land(SCALAR_UINT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            bool * local_retval_ptr = &retval.mMask[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] && b;
            }
            return retval;
        }
        // LORV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> lor(SIMDVec_u const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            bool * local_retval_ptr = &retval.mMask[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] || local_b_ptr[i];
            }
            return retval;
        }
        // LORS
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> lor(SCALAR_UINT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            bool * local_retval_ptr = &retval.mMask[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] || b;
            }
            return retval;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(SCALAR_UINT_TYPE * baseAddr, SCALAR_UINT_TYPE* indices) {
            for(int i = 0; i < VEC_LEN; i++)
            {
                mVec[i] = baseAddr[indices[i]];
            }
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE* baseAddr, SCALAR_UINT_TYPE* indices) {
            for(int i = 0; i < VEC_LEN; i++)
            {
                if(mask.mMask[i] == true) mVec[i] = baseAddr[indices[i]];
            }
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(SCALAR_UINT_TYPE * baseAddr, SIMDVec_u const & indices) {
            for(int i = 0; i < VEC_LEN; i++)
            {
                mVec[i] = baseAddr[indices.mVec[i]];
            }
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE* baseAddr, SIMDVec_u const & indices) {
            for(int i = 0; i < VEC_LEN; i++)
            {
                if(mask.mMask[i] == true) mVec[i] = baseAddr[indices.mVec[i]];
            }
            return *this;
        }
        // SCATTERS
        UME_FORCE_INLINE SCALAR_UINT_TYPE* scatter(SCALAR_UINT_TYPE* baseAddr, SCALAR_UINT_TYPE* indices) const {
            for(int i = 0; i < VEC_LEN; i++)
            {
                baseAddr[indices[i]] = mVec[i];
            }
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE SCALAR_UINT_TYPE* scatter(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE* baseAddr, SCALAR_UINT_TYPE* indices) const {
            for(int i = 0; i < VEC_LEN; i++)
            {
                if(mask.mMask[i]) baseAddr[indices[i]] = mVec[i];
            }
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE SCALAR_UINT_TYPE* scatter(SCALAR_UINT_TYPE* baseAddr, SIMDVec_u const & indices) const {
            for(int i = 0; i < VEC_LEN; i++)
            {
                baseAddr[indices.mVec[i]] = mVec[i];
            }
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE SCALAR_UINT_TYPE* scatter(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE* baseAddr, SIMDVec_u const & indices) const {
            for(int i = 0; i < VEC_LEN; i++)
            {
                if(mask.mMask[i]) baseAddr[indices.mVec[i]] = mVec[i];
            }
            return baseAddr;
        }
        // LSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] << local_b_ptr[i];
            }
            return retval;
        }
        // MLSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] << local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // LSHS
        UME_FORCE_INLINE SIMDVec_u lsh(SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] << b;
            }
            return retval;
        }
        // MLSHS
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] << b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // LSHVA
        UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] <<= local_b_ptr[i];
            }
            return *this;
        }
        // MLSHVA
        UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] <<= local_b_ptr[i];
            }
            return *this;
        }
        // LSHSA
        UME_FORCE_INLINE SIMDVec_u & lsha(SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] <<= b;
            }
            return *this;
        }
        // MLSHSA
        UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] <<= b;
            }
            return *this;
        }
        // RSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] >> local_b_ptr[i];
            }
            return retval;
        }
        // MRSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] >> local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // RSHS
        UME_FORCE_INLINE SIMDVec_u rsh(SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] >> b;
            }
            return retval;
        }
        // MRSHS
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SIMDVec_u retval;
            SCALAR_UINT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_UINT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] >> b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // RSHVA
        UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] >>= local_b_ptr[i];
            }
            return *this;
        }
        // MRSHVA
        UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u const & b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] >>= local_b_ptr[i];
            }
            return *this;
        }
        // RSHSA
        UME_FORCE_INLINE SIMDVec_u & rsha(SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] >>= b;
            }
            return *this;
        }
        // MRSHSA
        UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) {
            SCALAR_UINT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] >>= b;
            }
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
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_u<SCALAR_UINT_LOWER_PRECISION, VEC_LEN>() const;
        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_u<SCALAR_UINT_HIGHER_PRECISION, VEC_LEN>() const;

        // UTOI
        UME_FORCE_INLINE operator SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN>() const;
        // UTOF
        UME_FORCE_INLINE operator SIMDVec_f<SCALAR_FLOAT_TYPE, VEC_LEN>() const;
    };

    // SIMD NullTypes. These are used whenever a terminating
    // scalar type is used as a creator function for SIMD type.
    // These types cannot be instantiated, but are necessary for 
    // typeset to be consistent.
    template<>
    class SIMDVec_u<NullType<1>, 1>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<1>, 2>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<1>, 4>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<1>, 8>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<1>, 16>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<1>, 32>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<1>, 64>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<1>, 128>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<2>, 1>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<2>, 2>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<2>, 4>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<2>, 8>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<2>, 16>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<2>, 32>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<2>, 64>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<2>, 128>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<3>, 1>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<3>, 2>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<3>, 4>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<3>, 8>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<3>, 16>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<3>, 32>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<3>, 64>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<3>, 128>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };
}
}

#endif
