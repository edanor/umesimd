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

#ifndef UME_SIMD_PLUGIN_SCALAR_EMULATION_H_
#define UME_SIMD_PLUGIN_SCALAR_EMULATION_H_

#include <type_traits>

#include "UMESimdInterface.h"

namespace UME
{
namespace SIMD
{
    
    // forward declarations of simd types classes;
    template<typename SCALAR_TYPE, uint32_t VEC_LEN>       class SIMDVecScalarEmuMask;
    template<uint32_t SMASK_LEN>                           class SIMDVecScalarEmuSwizzleMask;
    template<typename SCALAR_UINT_TYPE, uint32_t VEC_LEN>  class SIMDVecScalarEmu_u;
    template<typename SCALAR_INT_TYPE, uint32_t VEC_LEN>   class SIMDVecScalarEmu_i;
    template<typename SCALAR_FLOAT_TYPE, uint32_t VEC_LEN> class SIMDVecScalarEmu_f;


    template<typename MASK_BASE_TYPE, uint32_t VEC_LEN>
    struct SIMDVecScalarEmuMask_traits {};

    template<>
    struct SIMDVecScalarEmuMask_traits<bool, 1> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecScalarEmuMask_traits<bool, 2> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecScalarEmuMask_traits<bool, 4> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecScalarEmuMask_traits<bool, 8> { 
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecScalarEmuMask_traits<bool, 16> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecScalarEmuMask_traits<bool, 32> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecScalarEmuMask_traits<bool, 64> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecScalarEmuMask_traits<bool, 128> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    
    // MASK_BASE_TYPE is the type of element that will represent single entry in
    //                mask register. This can be for examle a 'bool' or 'unsigned int' or 'float'
    //                The actual representation depends on how the underlying instruction
    //                set handles the masks/mask registers. For scalar emulation the mask vetor should
    //                be represented using a boolean values. Bool in C++ has one disadventage: it is possible
    //                for the compiler to implicitly cast it to integer. To forbid this casting operations from
    //                happening the default type has to be wrapped into a class. 
    template<typename MASK_BASE_TYPE, uint32_t VEC_LEN>
    class SIMDVecScalarEmuMask : public SIMDMaskBaseInterface< 
        SIMDVecScalarEmuMask<MASK_BASE_TYPE, VEC_LEN>,
        MASK_BASE_TYPE,
        VEC_LEN>
    {   
        typedef ScalarTypeWrapper<MASK_BASE_TYPE> MASK_SCALAR_TYPE; // Wrapp-up MASK_BASE_TYPE (int, float, bool) with a class
        typedef SIMDVecScalarEmuMask_traits<MASK_BASE_TYPE, VEC_LEN> MASK_TRAITS;
    private:
        MASK_SCALAR_TYPE mMask[VEC_LEN]; // each entry represents single mask element. For real SIMD vectors, mMask will be of mask intrinsic type.
    public:
        SIMDVecScalarEmuMask() {
            UME_EMULATION_WARNING();
            for(int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = MASK_SCALAR_TYPE(MASK_TRAITS::FALSE()); // Iniitialize MASK with FALSE value. False value depends on mask representation.
            }
        };

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        SIMDVecScalarEmuMask( bool m ) {
            UME_EMULATION_WARNING();
            for(int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = MASK_SCALAR_TYPE(m);
            }
        };
        
        // TODO: this should be handled using variadic templates, but unfortunatelly Visual Studio does not support this feature...
        SIMDVecScalarEmuMask( bool m0, bool m1 )
        {
            mMask[0] = MASK_SCALAR_TYPE(m0); 
            mMask[1] = MASK_SCALAR_TYPE(m1);
        };

        SIMDVecScalarEmuMask( bool m0, bool m1, bool m2, bool m3 )
        {
            mMask[0] = MASK_SCALAR_TYPE(m0); 
            mMask[1] = MASK_SCALAR_TYPE(m1); 
            mMask[2] = MASK_SCALAR_TYPE(m2); 
            mMask[3] = MASK_SCALAR_TYPE(m3);
        };

        SIMDVecScalarEmuMask( bool m0, bool m1, bool m2, bool m3,
                                bool m4, bool m5, bool m6, bool m7 )
        {
            mMask[0] = MASK_SCALAR_TYPE(m0); mMask[1] = MASK_SCALAR_TYPE(m1);
            mMask[2] = MASK_SCALAR_TYPE(m2); mMask[3] = MASK_SCALAR_TYPE(m3);
            mMask[4] = MASK_SCALAR_TYPE(m4); mMask[5] = MASK_SCALAR_TYPE(m5);
            mMask[6] = MASK_SCALAR_TYPE(m6); mMask[7] = MASK_SCALAR_TYPE(m7);
        };

        SIMDVecScalarEmuMask( bool m0,  bool m1,  bool m2,  bool m3,
                                bool m4,  bool m5,  bool m6,  bool m7,
                                bool m8,  bool m9,  bool m10, bool m11,
                                bool m12, bool m13, bool m14, bool m15 )
        {
            mMask[0] = MASK_SCALAR_TYPE(m0);  mMask[1] = MASK_SCALAR_TYPE(m1);
            mMask[2] = MASK_SCALAR_TYPE(m2);  mMask[3] = MASK_SCALAR_TYPE(m3);
            mMask[4] = MASK_SCALAR_TYPE(m4);  mMask[5] = MASK_SCALAR_TYPE(m5);
            mMask[6] = MASK_SCALAR_TYPE(m6);  mMask[7] = MASK_SCALAR_TYPE(m7);
            mMask[8] = MASK_SCALAR_TYPE(m8);  mMask[9] = MASK_SCALAR_TYPE(m9);
            mMask[10] = MASK_SCALAR_TYPE(m10); mMask[11] = MASK_SCALAR_TYPE(m11);
            mMask[12] = MASK_SCALAR_TYPE(m12); mMask[13] = MASK_SCALAR_TYPE(m13);
            mMask[14] = MASK_SCALAR_TYPE(m14); mMask[15] = MASK_SCALAR_TYPE(m15);
        };

        SIMDVecScalarEmuMask( bool m0,  bool m1,  bool m2,  bool m3,
                                bool m4,  bool m5,  bool m6,  bool m7,
                                bool m8,  bool m9,  bool m10, bool m11,
                                bool m12, bool m13, bool m14, bool m15,
                                bool m16, bool m17, bool m18, bool m19,
                                bool m20, bool m21, bool m22, bool m23,
                                bool m24, bool m25, bool m26, bool m27,
                                bool m28, bool m29, bool m30, bool m31)
        {
            mMask[0] = MASK_SCALAR_TYPE(m0);   mMask[1] = MASK_SCALAR_TYPE(m1);
            mMask[2] = MASK_SCALAR_TYPE(m2);   mMask[3] = MASK_SCALAR_TYPE(m3);
            mMask[4] = MASK_SCALAR_TYPE(m4);   mMask[5] = MASK_SCALAR_TYPE(m5);
            mMask[6] = MASK_SCALAR_TYPE(m6);   mMask[7] = MASK_SCALAR_TYPE(m7);
            mMask[8] = MASK_SCALAR_TYPE(m8);   mMask[9] = MASK_SCALAR_TYPE(m9);
            mMask[10] = MASK_SCALAR_TYPE(m10); mMask[11] = MASK_SCALAR_TYPE(m11);
            mMask[12] = MASK_SCALAR_TYPE(m12); mMask[13] = MASK_SCALAR_TYPE(m13);
            mMask[14] = MASK_SCALAR_TYPE(m14); mMask[15] = MASK_SCALAR_TYPE(m15);
            mMask[16] = MASK_SCALAR_TYPE(m16); mMask[17] = MASK_SCALAR_TYPE(m17);
            mMask[18] = MASK_SCALAR_TYPE(m18); mMask[19] = MASK_SCALAR_TYPE(m19);
            mMask[20] = MASK_SCALAR_TYPE(m20); mMask[21] = MASK_SCALAR_TYPE(m21);
            mMask[22] = MASK_SCALAR_TYPE(m22); mMask[23] = MASK_SCALAR_TYPE(m23);
            mMask[24] = MASK_SCALAR_TYPE(m24); mMask[25] = MASK_SCALAR_TYPE(m25);
            mMask[26] = MASK_SCALAR_TYPE(m26); mMask[27] = MASK_SCALAR_TYPE(m27);
            mMask[28] = MASK_SCALAR_TYPE(m28); mMask[29] = MASK_SCALAR_TYPE(m29);
            mMask[30] = MASK_SCALAR_TYPE(m30); mMask[31] = MASK_SCALAR_TYPE(m31);
        };

        // TODO: 64/128 element constructors
        
        inline bool extract(uint32_t index) const {
            return mMask[index];
        }

        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const { return MASK_SCALAR_TYPE(mMask[index]); }


        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) { 
            mMask[index] = MASK_SCALAR_TYPE(x);
        }

        SIMDVecScalarEmuMask(SIMDVecScalarEmuMask const & mask) {
            UME_EMULATION_WARNING();
            for(int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = mask.mMask[i];
            }
        }
    };

    template<uint32_t SMASK_LEN>
    class SIMDVecScalarEmuSwizzleMask : 
        public SIMDSwizzleMaskBaseInterface< 
            SIMDVecScalarEmuSwizzleMask<SMASK_LEN>,
            SMASK_LEN>
    {
    private:
        uint32_t mMaskElements[SMASK_LEN];
    public:
        SIMDVecScalarEmuSwizzleMask() { };

        explicit SIMDVecScalarEmuSwizzleMask(uint32_t m0) {
            UME_EMULATION_WARNING();
            for(int i = 0; i < SMASK_LEN; i++) {
                mMaskElements[i] = m0;
            }
        }

        explicit SIMDVecScalarEmuSwizzleMask(uint32_t *m) {
            UME_EMULATION_WARNING();
            for(int i = 0; i < SMASK_LEN; i++) {
                mMaskElements[i] = m[i];
            }
        }

        inline uint32_t extract(uint32_t index) const {
            UME_EMULATION_WARNING();
            return mMaskElements[index];
        }

        // A non-modifying element-wise access operator
        inline uint32_t operator[] (uint32_t index) const {
            UME_EMULATION_WARNING();
            return mMaskElements[index]; 
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, uint32_t x) { 
            UME_EMULATION_WARNING();
            mMaskElements[index] = x;
        }

        SIMDVecScalarEmuSwizzleMask(SIMDVecScalarEmuSwizzleMask const & mask) {
            UME_EMULATION_WARNING();
            for(int i = 0; i < SMASK_LEN; i++)
            {
                mMaskElements[i] = mask.mMaskElements[i];
            }
        }
    };

    template<typename VEC_TYPE, uint32_t VEC_LEN>
    struct SIMDVecScalarEmu_u_traits{
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };

    // 8b vectors
    template<>
    struct SIMDVecScalarEmu_u_traits<uint8_t, 1> {
        typedef int8_t                         SCALAR_INT_TYPE;
        typedef bool                           MASK_BASE_TYPE;
    };

    // 16b vectors
    template<>
    struct SIMDVecScalarEmu_u_traits<uint8_t, 2> {
        typedef SIMDVecScalarEmu_u<uint8_t, 1> HALF_LEN_VEC_TYPE;
        typedef int8_t   SCALAR_INT_TYPE;
        typedef bool     MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_u_traits<uint16_t, 1> {
        typedef int16_t  SCALAR_INT_TYPE;
        typedef bool     MASK_BASE_TYPE;
    };

    // 32b vectors
    template<>
    struct SIMDVecScalarEmu_u_traits<uint8_t, 4> {
        typedef SIMDVecScalarEmu_u<uint8_t, 2> HALF_LEN_VEC_TYPE;
        typedef int8_t                         SCALAR_INT_TYPE;
        typedef bool                           MASK_BASE_TYPE;
    };
    
    template<>
    struct SIMDVecScalarEmu_u_traits<uint16_t, 2> {
        typedef SIMDVecScalarEmu_u<uint16_t, 1> HALF_LEN_VEC_TYPE;
        typedef int16_t                         SCALAR_INT_TYPE;
        typedef bool                            MASK_BASE_TYPE;
    };
    
    template<>
    struct SIMDVecScalarEmu_u_traits<uint32_t, 1> {
        typedef int8_t   SCALAR_INT_TYPE;
        typedef bool     MASK_BASE_TYPE;
    };

    // 64b vectors
    template<>
    struct SIMDVecScalarEmu_u_traits<uint8_t, 8>{
        typedef SIMDVecScalarEmu_u<uint8_t, 4> HALF_LEN_VEC_TYPE;
        typedef int8_t                         SCALAR_INT_TYPE;
        typedef bool                           MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_u_traits<uint16_t, 4>{
        typedef SIMDVecScalarEmu_u<uint16_t, 2> HALF_LEN_VEC_TYPE;
        typedef int16_t                         SCALAR_INT_TYPE;
        typedef bool                            MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_u_traits<uint32_t, 2>{
        typedef SIMDVecScalarEmu_u<uint32_t, 1> HALF_LEN_VEC_TYPE;
        typedef int32_t                         SCALAR_INT_TYPE;
        typedef bool                            MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_u_traits<uint64_t, 1> {
        typedef int64_t   SCALAR_INT_TYPE;
        typedef bool      MASK_BASE_TYPE;
    };

    // 128b vectors
    template<>
    struct SIMDVecScalarEmu_u_traits<uint8_t, 16>{
        typedef SIMDVecScalarEmu_u<uint8_t, 8> HALF_LEN_VEC_TYPE;
        typedef int8_t                         SCALAR_INT_TYPE;
        typedef bool                           MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_u_traits<uint16_t, 8>{
        typedef SIMDVecScalarEmu_u<uint16_t, 4> HALF_LEN_VEC_TYPE;
        typedef int16_t                         SCALAR_INT_TYPE;
        typedef bool                            MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_u_traits<uint32_t, 4>{
        typedef SIMDVecScalarEmu_u<uint32_t, 2> HALF_LEN_VEC_TYPE;
        typedef int32_t                         SCALAR_INT_TYPE;
        typedef bool                            MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_u_traits<uint64_t, 2>{
        typedef SIMDVecScalarEmu_u<uint64_t, 1> HALF_LEN_VEC_TYPE;
        typedef int64_t                         SCALAR_INT_TYPE;
        typedef bool                            MASK_BASE_TYPE;
    };

    // 256b vectors
    template<>
    struct SIMDVecScalarEmu_u_traits<uint8_t, 32>{
        typedef SIMDVecScalarEmu_u<uint8_t, 16> HALF_LEN_VEC_TYPE;
        typedef int8_t                          SCALAR_INT_TYPE;
        typedef bool                            MASK_BASE_TYPE;
    };
    
    template<>
    struct SIMDVecScalarEmu_u_traits<uint16_t, 16>{
        typedef SIMDVecScalarEmu_u<uint16_t, 8> HALF_LEN_VEC_TYPE;
        typedef int16_t                         SCALAR_INT_TYPE;
        typedef bool                            MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_u_traits<uint32_t, 8>{
        typedef SIMDVecScalarEmu_u<uint32_t, 4> HALF_LEN_VEC_TYPE;
        typedef int32_t                         SCALAR_INT_TYPE;
        typedef bool                            MASK_BASE_TYPE;
    };
    
    template<>
    struct SIMDVecScalarEmu_u_traits<uint64_t, 4>{
        typedef SIMDVecScalarEmu_u<uint64_t, 2> HALF_LEN_VEC_TYPE;
        typedef int64_t                         SCALAR_INT_TYPE;
        typedef bool                            MASK_BASE_TYPE;
    };

    // 512b vectors
    template<>
    struct SIMDVecScalarEmu_u_traits<uint8_t, 64>{
        typedef SIMDVecScalarEmu_u<uint8_t, 32> HALF_LEN_VEC_TYPE;
        typedef int8_t                          SCALAR_INT_TYPE;
        typedef bool                            MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_u_traits<uint16_t, 32>{
        typedef SIMDVecScalarEmu_u<uint16_t, 16> HALF_LEN_VEC_TYPE;
        typedef int16_t                          SCALAR_INT_TYPE;
        typedef bool                             MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_u_traits<uint32_t, 16>{
        typedef SIMDVecScalarEmu_u<uint32_t, 8> HALF_LEN_VEC_TYPE;
        typedef int32_t                         SCALAR_INT_TYPE;
        typedef bool                            MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_u_traits<uint64_t, 8> {
        typedef SIMDVecScalarEmu_u<uint64_t, 4> HALF_LEN_VEC_TYPE;
        typedef int64_t                         SCALAR_INT_TYPE;
        typedef bool                            MASK_BASE_TYPE;
    };
    
    // 1024b vectors
    template<>
    struct SIMDVecScalarEmu_u_traits<uint8_t, 128> {
        typedef SIMDVecScalarEmu_u<uint8_t, 64> HALF_LEN_VEC_TYPE;
        typedef int8_t                          SCALAR_INT_TYPE;
        typedef bool                            MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_u_traits<uint16_t, 64> {
        typedef SIMDVecScalarEmu_u<uint16_t, 32> HALF_LEN_VEC_TYPE;
        typedef int16_t                          SCALAR_INT_TYPE;
        typedef bool                             MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_u_traits<uint32_t, 32> {
        typedef SIMDVecScalarEmu_u<uint32_t, 16> HALF_LEN_VEC_TYPE;
        typedef int32_t                          SCALAR_INT_TYPE;
        typedef bool                             MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_u_traits<uint64_t, 16> {
        typedef SIMDVecScalarEmu_u<uint64_t, 8> HALF_LEN_VEC_TYPE;
        typedef int64_t                         SCALAR_INT_TYPE;
        typedef bool                            MASK_BASE_TYPE;
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
    class SIMDVecScalarEmu_u final : 
        public SIMDVecUnsignedInterface< 
            SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, VEC_LEN>, // DERIVED_VEC_TYPE
            SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, VEC_LEN>, // DERIVED_VEC_UINT_TYPE
            SCALAR_UINT_TYPE,  // SCALAR_TYPE 
            SCALAR_UINT_TYPE,  // SCALAR_UINT_TYPE - in this case is the same as above
            VEC_LEN,
            SIMDVecScalarEmuMask<typename SIMDVecScalarEmu_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::MASK_BASE_TYPE, VEC_LEN>,
            SIMDVecScalarEmuSwizzleMask<VEC_LEN>>,
        public SIMDVecPackableInterface<
            SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, VEC_LEN>,        // DERIVED_VEC_TYPE
            typename SIMDVecScalarEmu_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::HALF_LEN_VEC_TYPE> // DERIVED_HALF_VEC_TYPE
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_UINT_TYPE, VEC_LEN>                                   VEC_EMU_REG;
            
        typedef typename SIMDVecScalarEmu_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::MASK_BASE_TYPE   MASK_BASE_TYPE;
        typedef typename SIMDVecScalarEmu_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::SCALAR_INT_TYPE  SCALAR_INT_TYPE;

        typedef SIMDVecScalarEmuMask<MASK_BASE_TYPE, VEC_LEN>   MASK_TYPE;

        // Conversion operators require access to private members.
        friend class SIMDVecScalarEmu_i<SCALAR_INT_TYPE, VEC_LEN>;

    private:
        // This is the only data member and it is a low level representation of vector register.
        VEC_EMU_REG mVec; 

    public:
        inline SIMDVecScalarEmu_u() : mVec() {};

        inline explicit SIMDVecScalarEmu_u(SCALAR_UINT_TYPE i) : mVec(i) {};

        inline SIMDVecScalarEmu_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3) {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
        }

        inline SIMDVecScalarEmu_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7) 
        {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
            mVec.insert(4, i4);  mVec.insert(5, i5);  mVec.insert(6, i6);  mVec.insert(7, i7);
        }

        inline SIMDVecScalarEmu_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7,
                            SCALAR_UINT_TYPE i8, SCALAR_UINT_TYPE i9, SCALAR_UINT_TYPE i10, SCALAR_UINT_TYPE i11, SCALAR_UINT_TYPE i12, SCALAR_UINT_TYPE i13, SCALAR_UINT_TYPE i14, SCALAR_UINT_TYPE i15)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15); 
        }

        inline SIMDVecScalarEmu_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7,
                            SCALAR_UINT_TYPE i8, SCALAR_UINT_TYPE i9, SCALAR_UINT_TYPE i10, SCALAR_UINT_TYPE i11, SCALAR_UINT_TYPE i12, SCALAR_UINT_TYPE i13, SCALAR_UINT_TYPE i14, SCALAR_UINT_TYPE i15,
                            SCALAR_UINT_TYPE i16, SCALAR_UINT_TYPE i17, SCALAR_UINT_TYPE i18, SCALAR_UINT_TYPE i19, SCALAR_UINT_TYPE i20, SCALAR_UINT_TYPE i21, SCALAR_UINT_TYPE i22, SCALAR_UINT_TYPE i23,
                            SCALAR_UINT_TYPE i24, SCALAR_UINT_TYPE i25, SCALAR_UINT_TYPE i26, SCALAR_UINT_TYPE i27, SCALAR_UINT_TYPE i28, SCALAR_UINT_TYPE i29, SCALAR_UINT_TYPE i30, SCALAR_UINT_TYPE i31)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15);     
            mVec.insert(16, i16);  mVec.insert(17, i17);  mVec.insert(18, i18);  mVec.insert(19, i19);
            mVec.insert(20, i20);  mVec.insert(21, i21);  mVec.insert(22, i22);  mVec.insert(23, i23);
            mVec.insert(24, i24);  mVec.insert(25, i25);  mVec.insert(26, i26);  mVec.insert(27, i27);
            mVec.insert(28, i28);  mVec.insert(29, i29);  mVec.insert(30, i30);  mVec.insert(31, i31);
        }
            
        // Override Access operators
        inline SCALAR_UINT_TYPE operator[] (uint32_t index) const {
            SCALAR_UINT_TYPE temp = mVec[index];
            return temp;
        }
                
        // insert[] (scalar)
        inline SIMDVecScalarEmu_u & insert(uint32_t index, SCALAR_UINT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline operator SIMDVecScalarEmu_i<SCALAR_INT_TYPE, VEC_LEN>() const {
            SIMDVecScalarEmu_i<SCALAR_INT_TYPE, VEC_LEN> retval;
            for(uint32_t i = 0; i < VEC_LEN; i++) {
                retval.insert(i, (SCALAR_INT_TYPE)mVec[i]);
            }
            return retval;
        }
    };
    
    // ***************************************************************************
    // *
    // *    Partial specialization of unsigned integer SIMD for VEC_LEN == 1.
    // *    This specialization is necessary to eliminate PACK operations from
    // *    being used on SIMD1 types.
    // *
    // ***************************************************************************
    template<typename SCALAR_UINT_TYPE>
    class SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, 1> : 
        public SIMDVecUnsignedInterface< 
            SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, 1>, // DERIVED_VEC_TYPE
            SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, 1>, // DERIVED_VEC_UINT_TYPE
            SCALAR_UINT_TYPE,  // SCALAR_TYPE 
            SCALAR_UINT_TYPE,  // SCALAR_UINT_TYPE - in this case is the same as above
            1,
            SIMDVecScalarEmuMask<typename SIMDVecScalarEmu_u_traits<SCALAR_UINT_TYPE, 1>::MASK_BASE_TYPE, 1>,
            SIMDVecScalarEmuSwizzleMask<1>>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_UINT_TYPE, 1>                                   VEC_EMU_REG;
            
        typedef typename SIMDVecScalarEmu_u_traits<SCALAR_UINT_TYPE, 1>::MASK_BASE_TYPE   MASK_BASE_TYPE;
        typedef typename SIMDVecScalarEmu_u_traits<SCALAR_UINT_TYPE, 1>::SCALAR_INT_TYPE  SCALAR_INT_TYPE;

        typedef SIMDVecScalarEmuMask<MASK_BASE_TYPE, 1>   MASK_TYPE;

        // Conversion operators require access to private members.
        friend class SIMDVecScalarEmu_i<SCALAR_INT_TYPE, 1>;

    private:
        // This is the only data member and it is a low level representation of vector register.
        VEC_EMU_REG mVec; 

    public:
        inline SIMDVecScalarEmu_u() : mVec() {};

        inline explicit SIMDVecScalarEmu_u(SCALAR_UINT_TYPE i) : mVec(i) {};

        // Override Access operators
        inline SCALAR_UINT_TYPE operator[] (uint32_t index) const {
            SCALAR_UINT_TYPE temp = mVec[index];
            return temp;
        }
                
        // insert[] (scalar)
        inline SIMDVecScalarEmu_u & insert(uint32_t index, SCALAR_UINT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline operator SIMDVecScalarEmu_i<SCALAR_INT_TYPE, 1>() const {
            SIMDVecScalarEmu_i<SCALAR_INT_TYPE, 1> retval(mVec[0]);
            return retval;
        }
    };

    template<typename SCALAR_INT_TYPE, uint32_t VEC_LEN>
    struct SIMDVecScalarEmu_i_traits{
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };

    // 8b vectors
    template<>
    struct SIMDVecScalarEmu_i_traits<int8_t, 1>{
        typedef SIMDVecScalarEmu_u<uint8_t, 1> VEC_UINT;
        typedef uint8_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };

    // 16b vectors
    template<>
    struct SIMDVecScalarEmu_i_traits<int8_t, 2>{
        typedef SIMDVecScalarEmu_i<int8_t, 1> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint8_t, 2> VEC_UINT;
        typedef uint8_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };
    
    template<>
    struct SIMDVecScalarEmu_i_traits<int16_t, 1>{
        typedef SIMDVecScalarEmu_u<uint16_t, 1> VEC_UINT;
        typedef uint16_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };

    // 32b vectors
    template<>
    struct SIMDVecScalarEmu_i_traits<int8_t, 4>{
        typedef SIMDVecScalarEmu_i<int8_t, 2> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint8_t, 4> VEC_UINT;
        typedef uint8_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_i_traits<int16_t, 2>{
        typedef SIMDVecScalarEmu_i<int16_t, 1> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint16_t, 2> VEC_UINT;
        typedef uint16_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };
    
    template<>
    struct SIMDVecScalarEmu_i_traits<int32_t, 1>{
        typedef SIMDVecScalarEmu_u<uint32_t, 1> VEC_UINT;
        typedef uint32_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };

    // 64b vectors
    template<>
    struct SIMDVecScalarEmu_i_traits<int8_t, 8>{
        typedef SIMDVecScalarEmu_i<int8_t, 4> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint8_t, 8> VEC_UINT;
        typedef uint8_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };
    
    template<>
    struct SIMDVecScalarEmu_i_traits<int16_t, 4>{
        typedef SIMDVecScalarEmu_i<int16_t, 2> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint16_t, 4> VEC_UINT;
        typedef uint16_t  SCALAR_UINT_TYPE;
        typedef bool      MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_i_traits<int32_t, 2>{
        typedef SIMDVecScalarEmu_i<int32_t, 1> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint32_t, 2> VEC_UINT;
        typedef uint32_t SCALAR_UINT_TYPE;
        typedef bool     MASK_BASE_TYPE;
    };
    
    template<>
    struct SIMDVecScalarEmu_i_traits<int64_t, 1>{
        typedef SIMDVecScalarEmu_u<uint64_t, 1> VEC_UINT;
        typedef uint64_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };

    // 128b vectors
    template<>
    struct SIMDVecScalarEmu_i_traits<int8_t, 16>{
        typedef SIMDVecScalarEmu_i<int8_t, 8> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint8_t, 16> VEC_UINT;
        typedef uint8_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_i_traits<int16_t, 8>{
        typedef SIMDVecScalarEmu_i<int16_t, 4> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint16_t, 8> VEC_UINT;
        typedef uint16_t SCALAR_UINT_TYPE;
        typedef bool     MASK_BASE_TYPE;
    };
            
    template<>
    struct SIMDVecScalarEmu_i_traits<int32_t, 4>{
        typedef SIMDVecScalarEmu_i<int32_t, 2> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint32_t, 4> VEC_UINT;
        typedef uint32_t SCALAR_UINT_TYPE;
        typedef bool     MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_i_traits<int64_t, 2>{
        typedef SIMDVecScalarEmu_i<int64_t, 1> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint64_t, 2> VEC_UINT;
        typedef uint64_t SCALAR_UINT_TYPE;
        typedef bool     MASK_BASE_TYPE;
    };

    // 256b vectors
    template<>
    struct SIMDVecScalarEmu_i_traits<int8_t, 32>{
        typedef SIMDVecScalarEmu_i<int8_t, 16> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint8_t, 32> VEC_UINT;
        typedef uint8_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };
    
    template<>
    struct SIMDVecScalarEmu_i_traits<int16_t, 16>{
        typedef SIMDVecScalarEmu_i<int16_t, 8> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint16_t, 16> VEC_UINT;
        typedef uint16_t SCALAR_UINT_TYPE;
        typedef bool     MASK_BASE_TYPE;
    };
    
    template<>
    struct SIMDVecScalarEmu_i_traits<int32_t, 8>{
        typedef SIMDVecScalarEmu_i<int32_t, 4> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint32_t, 8> VEC_UINT;
        typedef uint32_t SCALAR_UINT_TYPE;
        typedef bool     MASK_BASE_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_i_traits<int64_t, 4>{
        typedef SIMDVecScalarEmu_i<int64_t, 2> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint64_t, 4> VEC_UINT;
        typedef uint64_t SCALAR_UINT_TYPE;
        typedef bool     MASK_BASE_TYPE;
    };

    // 512b vectors
    template<>
    struct SIMDVecScalarEmu_i_traits<int8_t, 64>{
        typedef SIMDVecScalarEmu_i<int8_t, 32> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint8_t, 64> VEC_UINT;
        typedef uint8_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };
    
    template<>
    struct SIMDVecScalarEmu_i_traits<int16_t, 32>{
        typedef SIMDVecScalarEmu_i<int16_t, 16> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint16_t, 32> VEC_UINT;
        typedef uint16_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };
    
    template<>
    struct SIMDVecScalarEmu_i_traits<int32_t, 16>{
        typedef SIMDVecScalarEmu_i<int32_t, 8> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint32_t, 16> VEC_UINT;
        typedef uint32_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };
    
    template<>
    struct SIMDVecScalarEmu_i_traits<int64_t, 8>{
        typedef SIMDVecScalarEmu_i<int64_t, 4> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint64_t, 8> VEC_UINT;
        typedef uint64_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };  

    // 1024b vectors
    template<>
    struct SIMDVecScalarEmu_i_traits<int8_t, 128>{
        typedef SIMDVecScalarEmu_i<int8_t, 64> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint8_t, 128> VEC_UINT;
        typedef uint8_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };
    
    template<>
    struct SIMDVecScalarEmu_i_traits<int16_t, 64>{
        typedef SIMDVecScalarEmu_i<int16_t, 32> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint16_t, 64> VEC_UINT;
        typedef uint16_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };
    
    template<>
    struct SIMDVecScalarEmu_i_traits<int32_t, 32>{
        typedef SIMDVecScalarEmu_i<int32_t, 16> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint32_t, 32> VEC_UINT;
        typedef uint32_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };
    
    template<>
    struct SIMDVecScalarEmu_i_traits<int64_t, 16>{
        typedef SIMDVecScalarEmu_i<int64_t, 8> HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint64_t, 16> VEC_UINT;
        typedef uint64_t SCALAR_UINT_TYPE;
        typedef bool    MASK_BASE_TYPE;
    };  

    // ***************************************************************************
    // *
    // *    Implementation of signed integer SIMDx_8i, SIMDx_16i, SIMDx_32i, 
    // *    and SIMDx_64i.
    // *
    // *    This implementation uses scalar emulation available through to 
    // *    SIMDVecSignedInterface.
    // *
    // ***************************************************************************
    template<typename SCALAR_INT_TYPE, uint32_t VEC_LEN>
    class SIMDVecScalarEmu_i final : 
        public SIMDVecSignedInterface<
            SIMDVecScalarEmu_i<SCALAR_INT_TYPE, VEC_LEN>, 
            typename SIMDVecScalarEmu_i_traits<SCALAR_INT_TYPE, VEC_LEN>::VEC_UINT,
            SCALAR_INT_TYPE, 
            VEC_LEN,
            typename SIMDVecScalarEmu_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE,
            SIMDVecScalarEmuMask<
                typename SIMDVecScalarEmu_i_traits<
                    SCALAR_INT_TYPE, 
                    VEC_LEN>::MASK_BASE_TYPE, 
                    VEC_LEN>,
            SIMDVecScalarEmuSwizzleMask<VEC_LEN>>,
        public SIMDVecPackableInterface<
            SIMDVecScalarEmu_i<SCALAR_INT_TYPE, VEC_LEN>,
            typename SIMDVecScalarEmu_i_traits<SCALAR_INT_TYPE, VEC_LEN>::HALF_LEN_VEC_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_INT_TYPE, VEC_LEN>                            VEC_EMU_REG;
            
        typedef typename SIMDVecScalarEmu_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE     SCALAR_UINT_TYPE;
        typedef typename SIMDVecScalarEmu_i_traits<SCALAR_INT_TYPE, VEC_LEN>::VEC_UINT             VEC_UINT;
        typedef typename SIMDVecScalarEmu_i_traits<SCALAR_INT_TYPE, VEC_LEN>::MASK_BASE_TYPE       MASK_BASE_TYPE;

        typedef SIMDVecScalarEmuMask<MASK_BASE_TYPE, VEC_LEN> MASK_TYPE;


        friend class SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, VEC_LEN>;
    private:
        VEC_EMU_REG mVec;

    public:
        inline SIMDVecScalarEmu_i() : mVec() {};

        inline explicit SIMDVecScalarEmu_i(SCALAR_INT_TYPE i) : mVec(i) {};

        inline SIMDVecScalarEmu_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3) {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
        }

        inline SIMDVecScalarEmu_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7) 
        {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
            mVec.insert(4, i4);  mVec.insert(5, i5);  mVec.insert(6, i6);  mVec.insert(7, i7);
        }

        inline SIMDVecScalarEmu_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7,
                            SCALAR_INT_TYPE i8, SCALAR_INT_TYPE i9, SCALAR_INT_TYPE i10, SCALAR_INT_TYPE i11, SCALAR_INT_TYPE i12, SCALAR_INT_TYPE i13, SCALAR_INT_TYPE i14, SCALAR_INT_TYPE i15)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15); 
        }

        inline SIMDVecScalarEmu_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7,
                            SCALAR_INT_TYPE i8, SCALAR_INT_TYPE i9, SCALAR_INT_TYPE i10, SCALAR_INT_TYPE i11, SCALAR_INT_TYPE i12, SCALAR_INT_TYPE i13, SCALAR_INT_TYPE i14, SCALAR_INT_TYPE i15,
                            SCALAR_INT_TYPE i16, SCALAR_INT_TYPE i17, SCALAR_INT_TYPE i18, SCALAR_INT_TYPE i19, SCALAR_INT_TYPE i20, SCALAR_INT_TYPE i21, SCALAR_INT_TYPE i22, SCALAR_INT_TYPE i23,
                            SCALAR_INT_TYPE i24, SCALAR_INT_TYPE i25, SCALAR_INT_TYPE i26, SCALAR_INT_TYPE i27, SCALAR_INT_TYPE i28, SCALAR_INT_TYPE i29, SCALAR_INT_TYPE i30, SCALAR_INT_TYPE i31)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15);     
            mVec.insert(16, i16);  mVec.insert(17, i17);  mVec.insert(18, i18);  mVec.insert(19, i19);
            mVec.insert(20, i20);  mVec.insert(21, i21);  mVec.insert(22, i22);  mVec.insert(23, i23);
            mVec.insert(24, i24);  mVec.insert(25, i25);  mVec.insert(26, i26);  mVec.insert(27, i27);
            mVec.insert(28, i28);  mVec.insert(29, i29);  mVec.insert(30, i30);  mVec.insert(31, i31);
        }
            
        // Override Access operators
        inline SCALAR_INT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }
                
        // insert[] (scalar)
        inline SIMDVecScalarEmu_i & insert(uint32_t index, SCALAR_INT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        // Conversion to unsigned integer
        /*operator VEC_UINT() const { 
            VEC_UINT retvec;
            for(uint32_t i = 0; i < VEC_LEN; i++) {
                retvec.mVec.insert(i, (SCALAR_UINT_TYPE) mVec[i]);
            }
            return retvec;
        }*/

        inline  operator SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, VEC_LEN>() const {
            SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, VEC_LEN> retval;
            for(uint32_t i = 0; i < VEC_LEN; i++) {
                retval.insert(i, (SCALAR_UINT_TYPE)mVec[i]);
            }
            return retval;
        }
    };
    
    // ***************************************************************************
    // *
    // *    Partial specialization of signed integer SIMD for VEC_LEN == 1.
    // *    This specialization is necessary to eliminate PACK operations from
    // *    being used on SIMD1 types.
    // *
    // ***************************************************************************
    template<typename SCALAR_INT_TYPE>
    class SIMDVecScalarEmu_i<SCALAR_INT_TYPE, 1> : 
        public SIMDVecSignedInterface<
            SIMDVecScalarEmu_i<SCALAR_INT_TYPE, 1>, 
            typename SIMDVecScalarEmu_i_traits<SCALAR_INT_TYPE, 1>::VEC_UINT,
            SCALAR_INT_TYPE, 
            1,
            typename SIMDVecScalarEmu_i_traits<SCALAR_INT_TYPE, 1>::SCALAR_UINT_TYPE,
            SIMDVecScalarEmuMask<
                typename SIMDVecScalarEmu_i_traits<SCALAR_INT_TYPE, 1>::MASK_BASE_TYPE, 
                1>,
            SIMDVecScalarEmuSwizzleMask<1>
        >
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_INT_TYPE, 1>                            VEC_EMU_REG;
            
        typedef typename SIMDVecScalarEmu_i_traits<SCALAR_INT_TYPE, 1>::SCALAR_UINT_TYPE     SCALAR_UINT_TYPE;
        typedef typename SIMDVecScalarEmu_i_traits<SCALAR_INT_TYPE, 1>::VEC_UINT             VEC_UINT;
        typedef typename SIMDVecScalarEmu_i_traits<SCALAR_INT_TYPE, 1>::MASK_BASE_TYPE       MASK_BASE_TYPE;

        typedef SIMDVecScalarEmuMask<MASK_BASE_TYPE, 1> MASK_TYPE;


        friend class SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, 1>;
    private:
        VEC_EMU_REG mVec;

    public:
        inline SIMDVecScalarEmu_i() : mVec() {};

        inline explicit SIMDVecScalarEmu_i(SCALAR_INT_TYPE i) : mVec(i) {};

        // Override Access operators
        inline SCALAR_INT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }
                
        // insert[] (scalar)
        inline SIMDVecScalarEmu_i & insert(uint32_t index, SCALAR_INT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline  operator SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, 1>() const {
            SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, 1> retval(mVec[0]);
            return retval;
        }
    };

    template<typename SCALAR_FLOAT_TYPE, uint32_t VEC_LEN>
    struct SIMDVecScalarEmu_f_traits{
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };
    
    // 32b vectors
    template<>
    struct SIMDVecScalarEmu_f_traits<float, 1> {
        typedef SIMDVecScalarEmu_u<uint32_t, 1> VEC_UINT_TYPE;
        typedef SIMDVecScalarEmu_i<int32_t, 1>  VEC_INT_TYPE;
        typedef SIMDVecScalarEmuMask<bool, 1>   MASK_TYPE;
        typedef int32_t                         SCALAR_INT_TYPE;
        typedef uint32_t                        SCALAR_UINT_TYPE;
    };

    // 64b vectors
    template<>
    struct SIMDVecScalarEmu_f_traits<float, 2> {
        typedef SIMDVecScalarEmu_f<float, 1>    HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint32_t, 2> VEC_UINT_TYPE;
        typedef SIMDVecScalarEmu_i<int32_t, 2>  VEC_INT_TYPE;
        typedef SIMDVecScalarEmuMask<bool, 2>   MASK_TYPE;
        typedef int32_t                         SCALAR_INT_TYPE;
        typedef uint32_t                        SCALAR_UINT_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_f_traits<double, 1> {
        typedef SIMDVecScalarEmu_u<uint64_t, 1> VEC_UINT_TYPE;
        typedef SIMDVecScalarEmu_i<int64_t, 1>  VEC_INT_TYPE;
        typedef SIMDVecScalarEmuMask<bool, 1>   MASK_TYPE;
        typedef int32_t                         SCALAR_INT_TYPE;
        typedef uint32_t                        SCALAR_UINT_TYPE;
    };

    // 128b vectors
    template<>
    struct SIMDVecScalarEmu_f_traits<float, 4>{
        typedef SIMDVecScalarEmu_f<float, 2>    HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint32_t, 4> VEC_UINT_TYPE;
        typedef SIMDVecScalarEmu_i<int32_t, 4>  VEC_INT_TYPE;
        typedef SIMDVecScalarEmuMask<bool, 4>   MASK_TYPE;
        typedef int32_t                         SCALAR_INT_TYPE;
        typedef uint32_t                        SCALAR_UINT_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_f_traits<double, 2>{
        typedef SIMDVecScalarEmu_f<double, 1>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint64_t, 2> VEC_UINT_TYPE;
        typedef SIMDVecScalarEmu_i<int64_t, 2>  VEC_INT_TYPE;
        typedef SIMDVecScalarEmuMask<bool, 2>   MASK_TYPE;
        typedef int64_t                         SCALAR_INT_TYPE;
        typedef uint64_t                        SCALAR_UINT_TYPE;
    };

    // 256b vectors
    template<>
    struct SIMDVecScalarEmu_f_traits<float, 8>{
        typedef SIMDVecScalarEmu_f<float, 4>    HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint32_t, 8> VEC_UINT_TYPE;
        typedef SIMDVecScalarEmu_i<int32_t, 8>  VEC_INT_TYPE;
        typedef SIMDVecScalarEmuMask<bool, 8>   MASK_TYPE;
        typedef int32_t                         SCALAR_INT_TYPE;
        typedef uint32_t                        SCALAR_UINT_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_f_traits<double, 4>{
        typedef SIMDVecScalarEmu_f<double, 2>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint64_t, 4> VEC_UINT_TYPE;
        typedef SIMDVecScalarEmu_i<int64_t, 4>  VEC_INT_TYPE;
        typedef SIMDVecScalarEmuMask<bool, 4>   MASK_TYPE;
        typedef int64_t                         SCALAR_INT_TYPE;
        typedef uint64_t                        SCALAR_UINT_TYPE;
    };
    
    // 512b vectors
    template<>
    struct SIMDVecScalarEmu_f_traits<float, 16>{
        typedef SIMDVecScalarEmu_f<float, 8>     HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint32_t, 16> VEC_UINT_TYPE;
        typedef SIMDVecScalarEmu_i<int32_t, 16>  VEC_INT_TYPE;
        typedef SIMDVecScalarEmuMask<bool, 16>   MASK_TYPE;
        typedef int32_t                          SCALAR_INT_TYPE;
        typedef uint32_t                         SCALAR_UINT_TYPE;
    };
    
    template<>
    struct SIMDVecScalarEmu_f_traits<double, 8>{
        typedef SIMDVecScalarEmu_f<double, 4>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint64_t, 8> VEC_UINT_TYPE;
        typedef SIMDVecScalarEmu_i<int64_t, 8>  VEC_INT_TYPE;
        typedef SIMDVecScalarEmuMask<bool, 8>   MASK_TYPE;
        typedef int64_t                         SCALAR_INT_TYPE;
        typedef uint64_t                        SCALAR_UINT_TYPE;
    };
    
    // 1024b vectors
    template<>
    struct SIMDVecScalarEmu_f_traits<float, 32>{
        typedef SIMDVecScalarEmu_f<float, 16>    HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint32_t, 32> VEC_UINT_TYPE;
        typedef SIMDVecScalarEmu_i<int32_t, 32>  VEC_INT_TYPE;
        typedef SIMDVecScalarEmuMask<bool, 32>   MASK_TYPE;
        typedef int32_t                          SCALAR_INT_TYPE;
        typedef uint32_t                         SCALAR_UINT_TYPE;
    };

    template<>
    struct SIMDVecScalarEmu_f_traits<double, 16>{
        typedef SIMDVecScalarEmu_f<double, 8>    HALF_LEN_VEC_TYPE;
        typedef SIMDVecScalarEmu_u<uint64_t, 16> VEC_UINT_TYPE;
        typedef SIMDVecScalarEmu_i<int64_t, 16>  VEC_INT_TYPE;
        typedef SIMDVecScalarEmuMask<bool, 16>   MASK_TYPE;
        typedef int64_t                          SCALAR_INT_TYPE;
        typedef uint64_t                         SCALAR_UINT_TYPE;
    };
    
    // ***************************************************************************
    // *
    // *    Implementation of floating point types SIMDx_32f and SIMDx_64f.
    // *
    // *    This implementation uses scalar emulation available through to 
    // *    SIMDVecFloatInterface.
    // *
    // ***************************************************************************
    template<typename SCALAR_FLOAT_TYPE, uint32_t VEC_LEN>
    class SIMDVecScalarEmu_f : 
        public SIMDVecFloatInterface<
            SIMDVecScalarEmu_f<SCALAR_FLOAT_TYPE, VEC_LEN>, 
            typename SIMDVecScalarEmu_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_UINT_TYPE,
            typename SIMDVecScalarEmu_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_INT_TYPE,
            SCALAR_FLOAT_TYPE, 
            VEC_LEN,
            typename SIMDVecScalarEmu_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE,
            typename SIMDVecScalarEmu_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::MASK_TYPE,
            SIMDVecScalarEmuSwizzleMask<VEC_LEN>>,
        public SIMDVecPackableInterface<
            SIMDVecScalarEmu_f<SCALAR_FLOAT_TYPE, VEC_LEN>, 
            typename SIMDVecScalarEmu_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::HALF_LEN_VEC_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_FLOAT_TYPE, VEC_LEN> VEC_EMU_REG;
    private:
        VEC_EMU_REG mVec;

    public:
        inline SIMDVecScalarEmu_f() : mVec() {}

        inline explicit SIMDVecScalarEmu_f(SCALAR_FLOAT_TYPE f) : mVec(f) {}
        
        inline explicit SIMDVecScalarEmu_f(SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1) {
            mVec.insert(0, f0); mVec.insert(1, f1);
        }

        inline SIMDVecScalarEmu_f(SCALAR_FLOAT_TYPE i0, SCALAR_FLOAT_TYPE i1, SCALAR_FLOAT_TYPE i2, SCALAR_FLOAT_TYPE i3) {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
        }

        inline SIMDVecScalarEmu_f(SCALAR_FLOAT_TYPE i0, SCALAR_FLOAT_TYPE i1, SCALAR_FLOAT_TYPE i2, SCALAR_FLOAT_TYPE i3, SCALAR_FLOAT_TYPE i4, SCALAR_FLOAT_TYPE i5, SCALAR_FLOAT_TYPE i6, SCALAR_FLOAT_TYPE i7) 
        {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
            mVec.insert(4, i4);  mVec.insert(5, i5);  mVec.insert(6, i6);  mVec.insert(7, i7);
        }

        inline SIMDVecScalarEmu_f(SCALAR_FLOAT_TYPE i0, SCALAR_FLOAT_TYPE i1, SCALAR_FLOAT_TYPE i2, SCALAR_FLOAT_TYPE i3, SCALAR_FLOAT_TYPE i4, SCALAR_FLOAT_TYPE i5, SCALAR_FLOAT_TYPE i6, SCALAR_FLOAT_TYPE i7,
                            SCALAR_FLOAT_TYPE i8, SCALAR_FLOAT_TYPE i9, SCALAR_FLOAT_TYPE i10, SCALAR_FLOAT_TYPE i11, SCALAR_FLOAT_TYPE i12, SCALAR_FLOAT_TYPE i13, SCALAR_FLOAT_TYPE i14, SCALAR_FLOAT_TYPE i15)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15); 
        }

        inline SIMDVecScalarEmu_f(SCALAR_FLOAT_TYPE i0, SCALAR_FLOAT_TYPE i1, SCALAR_FLOAT_TYPE i2, SCALAR_FLOAT_TYPE i3, SCALAR_FLOAT_TYPE i4, SCALAR_FLOAT_TYPE i5, SCALAR_FLOAT_TYPE i6, SCALAR_FLOAT_TYPE i7,
                            SCALAR_FLOAT_TYPE i8, SCALAR_FLOAT_TYPE i9, SCALAR_FLOAT_TYPE i10, SCALAR_FLOAT_TYPE i11, SCALAR_FLOAT_TYPE i12, SCALAR_FLOAT_TYPE i13, SCALAR_FLOAT_TYPE i14, SCALAR_FLOAT_TYPE i15,
                            SCALAR_FLOAT_TYPE i16, SCALAR_FLOAT_TYPE i17, SCALAR_FLOAT_TYPE i18, SCALAR_FLOAT_TYPE i19, SCALAR_FLOAT_TYPE i20, SCALAR_FLOAT_TYPE i21, SCALAR_FLOAT_TYPE i22, SCALAR_FLOAT_TYPE i23,
                            SCALAR_FLOAT_TYPE i24, SCALAR_FLOAT_TYPE i25, SCALAR_FLOAT_TYPE i26, SCALAR_FLOAT_TYPE i27, SCALAR_FLOAT_TYPE i28, SCALAR_FLOAT_TYPE i29, SCALAR_FLOAT_TYPE i30, SCALAR_FLOAT_TYPE i31)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15);     
            mVec.insert(16, i16);  mVec.insert(17, i17);  mVec.insert(18, i18);  mVec.insert(19, i19);
            mVec.insert(20, i20);  mVec.insert(21, i21);  mVec.insert(22, i22);  mVec.insert(23, i23);
            mVec.insert(24, i24);  mVec.insert(25, i25);  mVec.insert(26, i26);  mVec.insert(27, i27);
            mVec.insert(28, i28);  mVec.insert(29, i29);  mVec.insert(30, i30);  mVec.insert(31, i31);
        }
            
        // Override Access operators
        inline SCALAR_FLOAT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }
                
        // insert[] (scalar)
        inline SIMDVecScalarEmu_f & insert(uint32_t index, SCALAR_FLOAT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }
    };

    // ***************************************************************************
    // *
    // *    Partial specialization of floating point SIMD for VEC_LEN == 1.
    // *    This specialization is necessary to eliminate PACK operations from
    // *    being used on SIMD1 types.
    // *
    // ***************************************************************************
    template<typename SCALAR_FLOAT_TYPE>
    class SIMDVecScalarEmu_f<SCALAR_FLOAT_TYPE, 1> : public SIMDVecFloatInterface<
        SIMDVecScalarEmu_f<SCALAR_FLOAT_TYPE, 1>, 
        typename SIMDVecScalarEmu_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_UINT_TYPE,
        typename SIMDVecScalarEmu_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_INT_TYPE,
        SCALAR_FLOAT_TYPE, 
        1,
        typename SIMDVecScalarEmu_f_traits<SCALAR_FLOAT_TYPE, 1>::SCALAR_UINT_TYPE,
        typename SIMDVecScalarEmu_f_traits<SCALAR_FLOAT_TYPE, 1>::MASK_TYPE,
        SIMDVecScalarEmuSwizzleMask<1>>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_FLOAT_TYPE, 1> VEC_EMU_REG;
    private:
        VEC_EMU_REG mVec;

    public:
        inline SIMDVecScalarEmu_f() : mVec() {};

        inline explicit SIMDVecScalarEmu_f(SCALAR_FLOAT_TYPE i) : mVec(i) {};

        // Override Access operators
        inline SCALAR_FLOAT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }

        // insert[] (scalar)
        inline SIMDVecScalarEmu_f & insert(uint32_t index, SCALAR_FLOAT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }
    };


#if defined USE_EMULATED_TYPES
    // mask vectors
    typedef SIMDVecScalarEmuMask<bool, 1>   SIMDMask1;
    typedef SIMDVecScalarEmuMask<bool, 2>   SIMDMask2;
    typedef SIMDVecScalarEmuMask<bool, 4>   SIMDMask4;
    typedef SIMDVecScalarEmuMask<bool, 8>   SIMDMask8;
    typedef SIMDVecScalarEmuMask<bool, 16>  SIMDMask16;
    typedef SIMDVecScalarEmuMask<bool, 32>  SIMDMask32;
    typedef SIMDVecScalarEmuMask<bool, 64>  SIMDMask64;
    typedef SIMDVecScalarEmuMask<bool, 128> SIMDMask128;     

    typedef SIMDVecScalarEmuSwizzleMask<1>   SIMDSwizzle1;
    typedef SIMDVecScalarEmuSwizzleMask<2>   SIMDSwizzle2;
    typedef SIMDVecScalarEmuSwizzleMask<4>   SIMDSwizzle4;
    typedef SIMDVecScalarEmuSwizzleMask<8>   SIMDSwizzle8;
    typedef SIMDVecScalarEmuSwizzleMask<16>  SIMDSwizzle16;
    typedef SIMDVecScalarEmuSwizzleMask<32>  SIMDSwizzle32;
    typedef SIMDVecScalarEmuSwizzleMask<64>  SIMDSwizzle64;
    typedef SIMDVecScalarEmuSwizzleMask<128> SIMDSwizzle128;

    // 8b uint vectors
    typedef SIMDVecScalarEmu_u<uint8_t, 1>      SIMD1_8u;

    // 16b uint vectors
    typedef SIMDVecScalarEmu_u<uint8_t, 2>      SIMD2_8u;
    typedef SIMDVecScalarEmu_u<uint16_t, 1>     SIMD1_16u;

    // 32b uint vectors
    typedef SIMDVecScalarEmu_u<uint8_t, 4>      SIMD4_8u;
    typedef SIMDVecScalarEmu_u<uint16_t, 2>     SIMD2_16u;
    typedef SIMDVecScalarEmu_u<uint32_t, 1>     SIMD1_32u;

    // 64b uint vectors
    typedef SIMDVecScalarEmu_u<uint8_t,  8>     SIMD8_8u;
    typedef SIMDVecScalarEmu_u<uint16_t, 4>     SIMD4_16u;
    typedef SIMDVecScalarEmu_u<uint32_t, 2>     SIMD2_32u; 
    typedef SIMDVecScalarEmu_u<uint64_t, 1>     SIMD1_64u;
    
    // 128b uint vectors
    typedef SIMDVecScalarEmu_u<uint8_t,  16>    SIMD16_8u;
    typedef SIMDVecScalarEmu_u<uint16_t, 8>     SIMD8_16u;
    typedef SIMDVecScalarEmu_u<uint32_t, 4>     SIMD4_32u;
    typedef SIMDVecScalarEmu_u<uint64_t, 2>     SIMD2_64u;
    
    // 256b uint vectors
    typedef SIMDVecScalarEmu_u<uint8_t,  32>    SIMD32_8u;
    typedef SIMDVecScalarEmu_u<uint16_t, 16>    SIMD16_16u;
    typedef SIMDVecScalarEmu_u<uint32_t, 8>     SIMD8_32u;
    typedef SIMDVecScalarEmu_u<uint64_t, 4>     SIMD4_64u;
    
    // 512b uint vectors
    typedef SIMDVecScalarEmu_u<uint8_t,  64>    SIMD64_8u;
    typedef SIMDVecScalarEmu_u<uint16_t, 32>    SIMD32_16u;
    typedef SIMDVecScalarEmu_u<uint32_t, 16>    SIMD16_32u;
    typedef SIMDVecScalarEmu_u<uint64_t, 8>     SIMD8_64u;

    // 1024 uint vectors
    typedef SIMDVecScalarEmu_u<uint8_t,  128>   SIMD128_8u;
    typedef SIMDVecScalarEmu_u<uint16_t, 64>    SIMD64_16u;
    typedef SIMDVecScalarEmu_u<uint32_t, 32>    SIMD32_32u;
    typedef SIMDVecScalarEmu_u<uint64_t, 16>    SIMD16_64u;
    
    // 8b int vectors
    typedef SIMDVecScalarEmu_i<int8_t,   1>     SIMD1_8i; 
    
    // 16b int vectors
    typedef SIMDVecScalarEmu_i<int8_t,   2>     SIMD2_8i;
    typedef SIMDVecScalarEmu_i<int16_t,  1>     SIMD1_16i;
    
    // 32b int vectors
    typedef SIMDVecScalarEmu_i<int8_t,   4>     SIMD4_8i;
    typedef SIMDVecScalarEmu_i<int16_t,  2>     SIMD2_16i;
    typedef SIMDVecScalarEmu_i<int32_t,  1>     SIMD1_32i;

    // 64b int vectors
    typedef SIMDVecScalarEmu_i<int8_t,   8>     SIMD8_8i; 
    typedef SIMDVecScalarEmu_i<int16_t,  4>     SIMD4_16i;
    typedef SIMDVecScalarEmu_i<int32_t,  2>     SIMD2_32i;
    typedef SIMDVecScalarEmu_i<int64_t,  1>     SIMD1_64i;

    // 128b int vectors
    typedef SIMDVecScalarEmu_i<int8_t,   16>    SIMD16_8i; 
    typedef SIMDVecScalarEmu_i<int16_t,  8>     SIMD8_16i;
    typedef SIMDVecScalarEmu_i<int32_t,  4>     SIMD4_32i;
    typedef SIMDVecScalarEmu_i<int64_t,  2>     SIMD2_64i;

    // 256b int vectors
    typedef SIMDVecScalarEmu_i<int8_t,   32>    SIMD32_8i;
    typedef SIMDVecScalarEmu_i<int16_t,  16>    SIMD16_16i;
    typedef SIMDVecScalarEmu_i<int32_t,  8>     SIMD8_32i;
    typedef SIMDVecScalarEmu_i<int64_t,  4>     SIMD4_64i;
    
    // 512b int vectors
    typedef SIMDVecScalarEmu_i<int8_t,   64>    SIMD64_8i;
    typedef SIMDVecScalarEmu_i<int16_t,  32>    SIMD32_16i;
    typedef SIMDVecScalarEmu_i<int32_t,  16>    SIMD16_32i;
    typedef SIMDVecScalarEmu_i<int64_t,  8>     SIMD8_64i;

    // 1024 int vectors
    typedef SIMDVecScalarEmu_i<int8_t, 128>     SIMD128_8i;
    typedef SIMDVecScalarEmu_i<int16_t, 64>     SIMD64_16i;
    typedef SIMDVecScalarEmu_i<int32_t, 32>     SIMD32_32i;
    typedef SIMDVecScalarEmu_i<int64_t, 16>     SIMD16_64i;

    // 32b float vectors
    typedef SIMDVecScalarEmu_f<float,  1>       SIMD1_32f;

    // 64b float vectors
    typedef SIMDVecScalarEmu_f<float,  2>       SIMD2_32f;
    typedef SIMDVecScalarEmu_f<double, 1>       SIMD1_64f;
    
    // 128b float vectors
    typedef SIMDVecScalarEmu_f<float,  4>       SIMD4_32f;
    typedef SIMDVecScalarEmu_f<double, 2>       SIMD2_64f;
    
    // 256b float vectors
    typedef SIMDVecScalarEmu_f<float,  8>       SIMD8_32f;
    typedef SIMDVecScalarEmu_f<double, 4>       SIMD4_64f;

    // 512b float vectors
    typedef SIMDVecScalarEmu_f<float,  16>      SIMD16_32f;
    typedef SIMDVecScalarEmu_f<double, 8>       SIMD8_64f;

    // 1024b float vectors
    typedef SIMDVecScalarEmu_f<float,  32>      SIMD32_32f;
    typedef SIMDVecScalarEmu_f<double, 16>      SIMD16_64f;
#endif

} // SIMD
} // UME

#endif
