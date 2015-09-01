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
//  “ICE-DIP is a European Industrial Doctorate project funded by the European Community’s 
//  7th Framework programme Marie Curie Actions under grant PITN-GA-2012-316596”.
//

#ifndef UME_SIMD_PLUGIN_NATIVE_KNC_H_
#define UME_SIMD_PLUGIN_NATIVE_KNC_H_


#include <type_traits>

#include "UMESimdInterface.h"
#include "UMESimdPluginScalarEmulation.h"

#include <immintrin.h>
namespace UME
{
namespace SIMD
{   

    // forward declarations of simd types classes;
    template<typename SCALAR_TYPE, uint32_t VEC_LEN>       class SIMDVecKNCMask;
    template<typename SCALAR_UINT_TYPE, uint32_t VEC_LEN>  class SIMDVecKNC_u;
    template<typename SCALAR_INT_TYPE, uint32_t VEC_LEN>   class SIMDVecKNC_i;
    template<typename SCALAR_FLOAT_TYPE, uint32_t VEC_LEN> class SIMDVecKNC_f;

    // ********************************************************************************************
    // MASK VECTORS
    // ********************************************************************************************
    template<typename MASK_BASE_TYPE, uint32_t VEC_LEN>
    struct SIMDVecKNCMask_traits {};

    template<>
    struct SIMDVecKNCMask_traits<bool, 1> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecKNCMask_traits<bool, 2> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecKNCMask_traits<bool, 4> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecKNCMask_traits<bool, 8> { 
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecKNCMask_traits<bool, 16> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecKNCMask_traits<bool, 32> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecKNCMask_traits<bool, 64> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecKNCMask_traits<bool, 128> {
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
    class SIMDVecKNCMask final : public SIMDMaskBaseInterface< SIMDVecKNCMask<MASK_BASE_TYPE, VEC_LEN>,
                                                               MASK_BASE_TYPE,
                                                               VEC_LEN>
    {   
        typedef ScalarTypeWrapper<MASK_BASE_TYPE> MASK_SCALAR_TYPE; // Wrapp-up MASK_BASE_TYPE (int, float, bool) with a class
        typedef SIMDVecKNCMask_traits<MASK_BASE_TYPE, VEC_LEN> MASK_TRAITS;
    private:
        MASK_SCALAR_TYPE mMask[VEC_LEN]; // each entry represents single mask element. For real SIMD vectors, mMask will be of mask intrinsic type.
    public:
        SIMDVecKNCMask() {
            UME_EMULATION_WARNING();
            for(int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = MASK_SCALAR_TYPE(MASK_TRAITS::FALSE()); // Iniitialize MASK with FALSE value. False value depends on mask representation.
            }
        }

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        SIMDVecKNCMask( bool m ) {
            UME_EMULATION_WARNING();
            for(int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = MASK_SCALAR_TYPE(m);
            }
        }
        
        // TODO: this should be handled using variadic templates, but unfortunatelly Visual Studio does not support this feature...
        SIMDVecKNCMask( bool m0, bool m1 )
        {
            mMask[0] = MASK_SCALAR_TYPE(m0); 
            mMask[1] = MASK_SCALAR_TYPE(m1);
        }

        SIMDVecKNCMask( bool m0, bool m1, bool m2, bool m3 )
        {
            mMask[0] = MASK_SCALAR_TYPE(m0); 
            mMask[1] = MASK_SCALAR_TYPE(m1); 
            mMask[2] = MASK_SCALAR_TYPE(m2); 
            mMask[3] = MASK_SCALAR_TYPE(m3);
        };

        SIMDVecKNCMask( bool m0, bool m1, bool m2, bool m3,
                                bool m4, bool m5, bool m6, bool m7 )
        {
            mMask[0] = MASK_SCALAR_TYPE(m0); mMask[1] = MASK_SCALAR_TYPE(m1);
            mMask[2] = MASK_SCALAR_TYPE(m2); mMask[3] = MASK_SCALAR_TYPE(m3);
            mMask[4] = MASK_SCALAR_TYPE(m4); mMask[5] = MASK_SCALAR_TYPE(m5);
            mMask[6] = MASK_SCALAR_TYPE(m6); mMask[7] = MASK_SCALAR_TYPE(m7);
        }

        SIMDVecKNCMask( bool m0,  bool m1,  bool m2,  bool m3,
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
        }

        SIMDVecKNCMask( bool m0,  bool m1,  bool m2,  bool m3,
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
        }

        // A non-modifying element-wise access operator
        inline MASK_SCALAR_TYPE operator[] (uint32_t index) const { return MASK_SCALAR_TYPE(mMask[index]); }

        inline MASK_BASE_TYPE extract(uint32_t index)
        {
            return mMask[index];
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) { 
            mMask[index] = MASK_SCALAR_TYPE(x);
        }

        SIMDVecKNCMask(SIMDVecKNCMask const & mask) {
            UME_EMULATION_WARNING();
            for(int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = mask.mMask[i];
            }
        }
    };
    
    // Mask vectors. Mask vectors with bool base type will resolve into scalar emulation.
    typedef SIMDVecKNCMask<bool, 1>     SIMDMask1;
    typedef SIMDVecKNCMask<bool, 2>     SIMDMask2;
    typedef SIMDVecKNCMask<bool, 4>     SIMDMask4;
    typedef SIMDVecKNCMask<bool, 8>     SIMDMask8;
    typedef SIMDVecKNCMask<bool, 16>    SIMDMask16;
    typedef SIMDVecKNCMask<bool, 32>    SIMDMask32;
    typedef SIMDVecKNCMask<bool, 64>    SIMDMask64;
    typedef SIMDVecKNCMask<bool, 128>   SIMDMask128;    


    // ********************************************************************************************
    // UNSIGNED INTEGER VECTORS
    // ********************************************************************************************
    template<typename VEC_TYPE, uint32_t VEC_LEN>
    struct SIMDVecKNC_u_traits{
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };

    // 64b vectors
    template<>
    struct SIMDVecKNC_u_traits<uint8_t, 8>{
        typedef int8_t    SCALAR_INT_TYPE;
        typedef SIMDMask8 MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_u_traits<uint16_t, 4>{
        typedef int16_t   SCALAR_INT_TYPE;
        typedef SIMDMask4 MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_u_traits<uint32_t, 2>{
        typedef int32_t   SCALAR_INT_TYPE;
        typedef SIMDMask2 MASK_TYPE;
    };

    // 128b vectors
    template<>
    struct SIMDVecKNC_u_traits<uint8_t, 16>{
        typedef int8_t     SCALAR_INT_TYPE;
        typedef SIMDMask16 MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_u_traits<uint16_t, 8>{
        typedef int16_t   SCALAR_INT_TYPE;
        typedef SIMDMask8 MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_u_traits<uint32_t, 4>{
        typedef int32_t   SCALAR_INT_TYPE;
        typedef SIMDMask4 MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_u_traits<uint64_t, 2>{
        typedef int64_t   SCALAR_INT_TYPE;
        typedef SIMDMask2 MASK_TYPE;
    };

    // 256b vectors
    template<>
    struct SIMDVecKNC_u_traits<uint8_t, 32>{
        typedef int8_t   SCALAR_INT_TYPE;
        typedef SIMDMask32 MASK_TYPE;
    };
    
    template<>
    struct SIMDVecKNC_u_traits<uint16_t, 16>{
        typedef int16_t   SCALAR_INT_TYPE;
        typedef SIMDMask16 MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_u_traits<uint32_t, 8>{
        typedef int32_t   SCALAR_INT_TYPE;
        typedef SIMDMask8 MASK_TYPE;
    };
    
    template<>
    struct SIMDVecKNC_u_traits<uint64_t, 4>{
        typedef int64_t   SCALAR_INT_TYPE;
        typedef SIMDMask4 MASK_TYPE;
    };

    // 512b vectors
    template<>
    struct SIMDVecKNC_u_traits<uint8_t, 64>{
        typedef int8_t   SCALAR_INT_TYPE;
        typedef SIMDMask64 MASK_TYPE;
    };
    
    template<>
    struct SIMDVecKNC_u_traits<uint16_t, 32>{
        typedef int16_t   SCALAR_INT_TYPE;
        typedef SIMDMask32 MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_u_traits<uint32_t, 16>{
        typedef int32_t   SCALAR_INT_TYPE;
        typedef SIMDMask16 MASK_TYPE;
    };
    
    template<>
    struct SIMDVecKNC_u_traits<uint64_t, 8>{
        typedef int64_t   SCALAR_INT_TYPE;
        typedef SIMDMask8 MASK_TYPE;
    };
    
    // 1024b vectors
    template<>
    struct SIMDVecKNC_u_traits<uint8_t, 128>{
        typedef int8_t   SCALAR_INT_TYPE;
        typedef SIMDMask128 MASK_TYPE;
    };
    
    template<>
    struct SIMDVecKNC_u_traits<uint16_t, 64>{
        typedef int16_t   SCALAR_INT_TYPE;
        typedef SIMDMask64 MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_u_traits<uint32_t, 32>{
        typedef int32_t   SCALAR_INT_TYPE;
        typedef SIMDMask32 MASK_TYPE;
    };
    
    template<>
    struct SIMDVecKNC_u_traits<uint64_t, 16>{
        typedef int64_t   SCALAR_INT_TYPE;
        typedef SIMDMask16 MASK_TYPE;
    };

    template<typename SCALAR_UINT_TYPE, uint32_t VEC_LEN>
    class SIMDVecKNC_u final : public SIMDVecUnsignedInterface<
        SIMDVecKNC_u<SCALAR_UINT_TYPE, VEC_LEN>, // DERIVED_VEC_TYPE
        SIMDVecKNC_u<SCALAR_UINT_TYPE, VEC_LEN>, // DERIVED_VEC_UINT_TYPE
        SCALAR_UINT_TYPE,                        // SCALAR_TYPE
        SCALAR_UINT_TYPE,                        // SCALAR_UINT_TYPE
        VEC_LEN,
        typename SIMDVecKNC_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::MASK_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_UINT_TYPE, VEC_LEN>                                   VEC_EMU_REG;
            
        typedef typename SIMDVecKNC_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::SCALAR_INT_TYPE  SCALAR_INT_TYPE;
        
        // Conversion operators require access to private members.
        friend class SIMDVecKNC_i<SCALAR_INT_TYPE, VEC_LEN>;

    private:
        // This is the only data member and it is a low level representation of vector register.
        VEC_EMU_REG mVec; 

    public:
        inline SIMDVecKNC_u() : mVec() {};

        inline explicit SIMDVecKNC_u(SCALAR_UINT_TYPE i) : mVec(i) {};

        inline SIMDVecKNC_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3) {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
        }

        inline SIMDVecKNC_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7) 
        {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
            mVec.insert(4, i4);  mVec.insert(5, i5);  mVec.insert(6, i6);  mVec.insert(7, i7);
        }

        inline SIMDVecKNC_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7,
                            SCALAR_UINT_TYPE i8, SCALAR_UINT_TYPE i9, SCALAR_UINT_TYPE i10, SCALAR_UINT_TYPE i11, SCALAR_UINT_TYPE i12, SCALAR_UINT_TYPE i13, SCALAR_UINT_TYPE i14, SCALAR_UINT_TYPE i15)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15); 
        }

        inline SIMDVecKNC_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7,
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
            return mVec[index];
        }
                
        // insert[] (scalar)
        inline SIMDVecKNC_u & insert(uint32_t index, SCALAR_UINT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline  operator SIMDVecKNC_i<SCALAR_INT_TYPE, VEC_LEN>() const {
            SIMDVecKNC_i<SCALAR_INT_TYPE, VEC_LEN> retval;
            for(uint32_t i = 0; i < VEC_LEN; i++) {
                retval.insert(i, (SCALAR_INT_TYPE)mVec[i]);
            }
            return retval;
        }
    };

                        
    // ********************************************************************************************
    // UNSIGNED INTEGER VECTORS specialization
    // ********************************************************************************************
    template<>
    class SIMDVecKNC_u<uint32_t, 16> : public SIMDVecUnsignedInterface< 
        SIMDVecKNC_u<uint32_t, 16>, 
        SIMDVecKNC_u<uint32_t, 16>,
        uint32_t, 
        uint32_t,
        16,
        SIMDMask16>
    {
    public:            
        // Conversion operators require access to private members.
        friend class SIMDVecKNC_i<int32_t, 16>;

    private:
        __m512i mVec;

        inline SIMDVecKNC_u(__m512i & x) { this->mVec = x; }
    public:
        inline SIMDVecKNC_u() { 
        }

        inline explicit SIMDVecKNC_u(uint32_t i) {
            mVec = _mm512_set1_epi32(i);
        }

        inline SIMDVecKNC_u(uint32_t i0,  uint32_t i1,  uint32_t i2,  uint32_t i3, 
                            uint32_t i4,  uint32_t i5,  uint32_t i6,  uint32_t i7,
                            uint32_t i8,  uint32_t i9,  uint32_t i10, uint32_t i11,
                            uint32_t i12, uint32_t i13, uint32_t i14, uint32_t i15) 
        {
           mVec = _mm512_setr_epi32(i0, i1, i2,  i3,  i4,  i5,  i6,  i7, 
                                    i8, i9, i10, i11, i12, i13, i14, i15);
        }

        inline uint32_t extract (uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING(); // This routine can be optimized
            alignas(32) uint32_t raw[8];
            _mm512_store_epi32(raw, mVec);
            return raw[index];
        }            

        // Override Access operators
        inline uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }
                
        // insert[] (scalar)
        inline SIMDVecKNC_u & insert(uint32_t index, uint32_t value) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) uint32_t raw[8];
            _mm512_store_epi32 (raw, mVec);
            raw[index] = value;
            _mm512_load_epi32(raw);
            return *this;
        }
        // assign (VEC) -> VEC        // assign (MASK, VEC) -> VEC
        // assign (VEC, scalar) -> VEC
        // assign (MASK, scalar) -> VEC
        // load (addr) -> VEC
        // load (MASK, addr) -> VEC
        // loadAligned (addrAligned) -> VEC
        // loadAligned (MASK, addrAligned) -> VEC
        // store (addr) -> scalar*
        // store (MASK, addr) -> scalar*
        // storeAligned (addrAligned) -> scalar*
        // storeAligned(MASK, addrAligned) -> scalar*
        // add (VEC) -> VEC
        // add (MASK, VEC) -> VEC
        // add (scalar) -> VEC
        // add (MASK, scalar) -> VEC
        // addAssign (VEC) -> VEC
        // addAssign (MASK, VEC) -> VEC
        // addAssign (scalar) -> VEC
        // addAssign (MASK, scalar) -> VEC
        // postfixIncrement () -> VEC
        // postfixIncrement (MASK) -> VEC
        // prefixIncrement () -> VEC
        // prefixIncrement (MASK) -> VEC
        // sub (VEC) -> VEC
        // sub(MASK, VEC) -> VEC
        // sub(scalar) -> VEC
        // sub(MASK, scalar) -> VEC
        // unaryMinus() -> VEC
        // unaryMinus(MASK) -> VEC
        // subAssign(VEC) -> VEC
        // subAssign (MASK, VEC) -> VEC
        // subAssign (scalar) -> VEC
        // subAssign(MASK, scalar) -> VEC
        // postfixDecrement() -> VEC
        // postfixDecrement (MASK) -> VEC
        // prefixDecrement() -> VEC
        // postfixDecrement (MASK, VEC) -> VEC
        // mult(VEC) -> VEC
        // mult(MASK, VEC) -> VEC
        // mult(scalar) -> VEC
        // mult(MASK, scalar) -> VEC
        // multAssign(VEC) -> VEC
        // multAssign (MASK, VEC) -> VEC
        // multAssign(scalar) -> VEC
        // div(VEC) -> VEC
        // div (MASK, VEC) -> VEC
        // div(scalar) -> VEC
        // div (MASK, scalar) -> VEC
        // divAssign (VEC, VEC) -> VEC
        // divAssign (MASK, VEC, VEC) -> VEC
        // divAssign (VEC, scalar) -> VEC
        // divAssign (MASK, VEC, scalar) -> VEC
        // reciprocal() -> VEC (implied nominator)
        // reciprocal(scalar) -> VEC
        // reciprocalAssign(scalar) -> VEC
        // isEqual (VEC) -> MASK
        // isEqual (scalar) -> MASK
        // isNotEqual (VEC) -> MASK
        // isNotEqual (scalar) -> MASK
        // isGreater (VEC) -> MASK
        // isGreater (scalar) -> MASK
        // isLesser (VEC) -> MASK
        // isLesser (scalar) -> MASK
        // isGreaterEqual (VEC) -> MASK
        // isGreaterEqual (scalar) -> MASK
        // isLesserEqual (VEC) -> MASK
        // isLesserEqual (scalar) -> MASK
        // binaryAnd(VEC) -> VEC
        // binaryAnd (MASK, VEC) -> VEC
        // binaryAnd (scalar) -> VEC
        // binaryAnd (MASK, scalar) -> VEC
        // binaryAndAssign (VEC) -> VEC
        // binaryAndAssign (MASK, VEC) -> VEC
        // binaryAndAssign (scalar) -> VEC
        // binaryAndAssign (MASK, scalar) -> VEC
        // binaryOr (VEC) -> VEC
        // binaryOr (MASK, VEC) -> VEC
        // binaryOr (scalar) -> VEC
        // binaryOr (MASK, scalar) -> VEC
        // binaryOrAssign (VEC) -> VEC
        // binaryOrAssign (MASK, VEC) -> VEC
        // binaryOrAssign (scalar) -> VEC
        // binaryOrAssign (MASK, scalar) -> VEC
        // binaryXor (VEC) -> VEC
        // binaryXor (MASK, VEC) -> VEC
        // binaryXor (scalar) -> VEC
        // binaryXor (MASK, scalar) -> VEC
        // binaryXorAssign (VEC) -> VEC
        // binaryXorAssign (MASK, VEC) -> VEC
        // binaryXorAssign (scalar) -> VEC
        // binaryXorAssign (MASK, scalar) -> VEC
        // binaryNot (VEC) -> VEC
        // binaryNot (MASK, VEC) -> VEC
        // binaryNotAssign () -> VEC
        // binaryNotAssign (MASK) -> VEC
        // blend(MASK, VEC) -> VEC
        // blend(MASK, scalar) -> VEC
        // blendAssign(MASK, VEC) -> VEC
        // blendAssign(MASK, scalar) -> VEC
        // reduceAdd () -> VEC
        // reduceAdd (MASK) -> scalar
        // reduceMult () -> scalar
        // reduceMult (MASK) -> scalar
        // reduceMult (scalar) -> scalar
        // reduceMult (MASK, scalar) -> scalar
        // reduceBinaryOr () -> scalar
        // reduceBinaryOr (MASK) -> scalar
        // reduceBinaryOr (scalar) -> scalar
        // reduceBinaryOr (MASK, scalar) -> scalar
        // reduceBinaryAnd () -> scalar
        // reduceBinaryAnd (MASK) -> scalar
        // reduceBinaryAnd (scalar) -> scalar
        // reduceBinaryAnd (MASK, scalar) -> scalar
        // gather (scalar_uint*, uint64_t*) -> VEC     
        // gather (MASK, scalar_uint*, uint64_t*) -> VEC
        // gather (scalar_uint*, VEC) -> VEC
        // gather (MASK, SCALAR_UINT_TYPE*, VEC) -> VEC
        // scatter (scalar_uint*, uint64_t*) -> scalar *
        // scatter (MASK, scalar_uint*, uint64_t*) -> scalar *
        // scatter (scalar_uint*, VEC_UINT) -> scalar *
        // scatter (MASK, scalar_uint*, VEC_UINT) -> scalar *
        // shiftBitsLeft (VEC_UINT, VEC_UINT) -> VEC
        // shiftBitsLeft (MASK, VEC_UINT, VEC_UINT) -> VEC
        // shiftBitsLeft (VEC_UINT, scalar_uint) -> VEC
        // shiftBitsLeft (MASK, VEC_UINT, scalar_uint) -> VEC
        // shiftBitsLeftAssign (VEC_UINT, VEC_UINT) -> VEC
        // shiftBitsLeftAssign (MASK, VEC_UINT, VEC_UINT) -> VEC
        // shiftBitsLeftAssign (VEC_UINT, scalar_uint) -> VEC
        // shiftBitsLeftAssign (MASK, VEC_UINT, scalar_uint) -> VEC
        // shiftBitsRight (VEC_UINT, VEC_UINT) -> VEC
        // shiftBitsRight (MASK, VEC_UINT, VEC_UINT) -> VEC
        // shiftBitsRight (VEC_UINT, scalar_uint) -> VEC
        // shiftBitsRight (MASK, VEC_UINT, scalar_uint) -> VEC
        // shiftBitsRightAssign (VEC_UINT, VEC_UINT) -> VEC
        // shiftBitsRightAssign (MASK, VEC_UINT, VEC_UINT) -> VEC
        // shiftBitsRightAssign (VEC_UINT, scalar_uint) -> VEC
        // shiftBitsRightAssign (MASK, VEC_UINT, scalar_uint) -> VEC
        // *****************************************************************************
        // rotateBitsLeft (UINT_VEC) -> VEC
        // rotateBitsLeft(MASK, VEC_UINT) -> VEC
        // rotateBitsLeftScalar(scalar_uint) -> VEC
        // rotateBitsLeftScalar(MASK, scalar_uint) -> 
        // rotateBitsLeft(scalar, VEC_UINT) -> VEC
        // rotateBitsLeft(MASK, scalar, VEC_UINT) -> VEC
        // rotateBitsLeftAssign(VEC_UINT) -> VEC
        // rotateBitsLeftAssign(MASK, VEC_UINT) -> VEC
        // rotateBitsLeftAssign(scalar_uint) -> VEC
        // rotateBitsLeftAssign(MASK, scalar_uint) -> VEC


        // ******************************************************************
        // * Additional math functions
        // ******************************************************************

        // max (VEC) -> VEC
        // max (MASK, VEC) -> VEC
        // max (scalar) -> VEC
        // max (MASK, scalar) -> VEC
        // maxReduce () -> VEC
        // maxReduce (MASK) -> VEC
        // maxReduce (scalar) -> VEC
        // maxReduce (MASK, scalar) -> VEC
        // min (VEC) -> VEC
        // min (MASK, VEC) -> VEC
        // min (scalar) -> VEC
        // min (MASK, scalar) -> VEC
        // abs () -> VEC
        // abs (MASK) -> VEC
        // absAssign () -> VEC
        // absAssign (MASK) -> VEC  

        // truncToInt () -> VEC_INT
        //inline DERIVED_VEC_INT_TYPE truncToInt() 
        // truncToInt (MASK) -> VEC_INT
        // sqrt () -> VEC
        // inline DERIVED_VEC_TYPE sqrt()
        // sqrt (MASK) -> VEC
        // sin () -> VEC
        // inline DERIVED_VEC_TYPE sin()
        // sin (MASK) -> VEC
        // inline DERIVED_VEC_TYPE sin(MASK_TYPE const & mask) 
        // cos () -> VEC
        // inline DERIVED_VEC_TYPE cos() 
        // cos (MASK) -> VEC
        // inline DERIVED_VEC_TYPE cos(MASK_TYPE const & mask)
        
        inline  operator SIMDVecKNC_i<int32_t, 16> const ();
    };
                        
    // ********************************************************************************************
    // SIGNED INTEGER VECTORS
    // ********************************************************************************************
    template<typename SCALAR_INT_TYPE, uint32_t VEC_LEN>
    struct SIMDVecKNC_i_traits{
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };

    // 64b vectors
    template<>
    struct SIMDVecKNC_i_traits<int8_t, 8> {
        typedef SIMDVecKNC_u<uint8_t, 8> VEC_UINT;
        typedef uint8_t SCALAR_UINT_TYPE;
        typedef SIMDMask8 MASK_TYPE;
    };
    
    template<>
    struct SIMDVecKNC_i_traits<int16_t, 4>{
        typedef SIMDVecKNC_u<uint16_t, 4> VEC_UINT;
        typedef uint16_t  SCALAR_UINT_TYPE;
        typedef SIMDMask4 MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int32_t, 2>{
        typedef SIMDVecKNC_u<uint32_t, 2> VEC_UINT;
        typedef uint32_t SCALAR_UINT_TYPE;
        typedef SIMDMask2 MASK_TYPE;
    };

    // 128b vectors
    template<>
    struct SIMDVecKNC_i_traits<int8_t, 16>{
        typedef SIMDVecKNC_u<uint8_t, 16> VEC_UINT;
        typedef uint8_t SCALAR_UINT_TYPE;
        typedef SIMDMask16 MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int16_t, 8>{
        typedef SIMDVecKNC_u<uint16_t, 8> VEC_UINT;
        typedef uint16_t SCALAR_UINT_TYPE;
        typedef SIMDMask8 MASK_TYPE;
    };
            
    template<>
    struct SIMDVecKNC_i_traits<int32_t, 4>{
        typedef SIMDVecKNC_u<uint32_t, 4> VEC_UINT;
        typedef uint32_t SCALAR_UINT_TYPE;
        typedef SIMDMask4 MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int64_t, 2>{
        typedef SIMDVecKNC_u<uint64_t, 2> VEC_UINT;
        typedef uint64_t SCALAR_UINT_TYPE;
        typedef SIMDMask2 MASK_TYPE;
    };

    // 256b vectors
    template<>
    struct SIMDVecKNC_i_traits<int8_t, 32>{
        typedef SIMDVecKNC_u<uint8_t, 32> VEC_UINT;
        typedef uint8_t SCALAR_UINT_TYPE;
        typedef SIMDMask32 MASK_TYPE;
    };
    
    template<>
    struct SIMDVecKNC_i_traits<int16_t, 16>{
        typedef SIMDVecKNC_u<uint16_t, 16> VEC_UINT;
        typedef uint16_t SCALAR_UINT_TYPE;
        typedef SIMDMask16 MASK_TYPE;
    };
    
    template<>
    struct SIMDVecKNC_i_traits<int32_t, 8>{
        typedef SIMDVecKNC_u<uint32_t, 8> VEC_UINT;
        typedef uint32_t SCALAR_UINT_TYPE;
        typedef SIMDMask8 MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int64_t, 4>{
        typedef SIMDVecKNC_u<uint64_t, 4> VEC_UINT;
        typedef uint64_t SCALAR_UINT_TYPE;
        typedef SIMDMask4 MASK_TYPE;
    };

    // 512b vectors
    template<>
    struct SIMDVecKNC_i_traits<int8_t, 64>{
        typedef SIMDVecKNC_u<uint8_t, 64> VEC_UINT;
        typedef uint8_t SCALAR_UINT_TYPE;
        typedef SIMDMask64 MASK_TYPE;
    };
    
    template<>
    struct SIMDVecKNC_i_traits<int16_t, 32>{
        typedef SIMDVecKNC_u<uint16_t, 32> VEC_UINT;
        typedef uint16_t SCALAR_UINT_TYPE;
        typedef SIMDMask32 MASK_TYPE;
    };
    
    template<>
    struct SIMDVecKNC_i_traits<int32_t, 16>{
        typedef SIMDVecKNC_u<uint32_t, 16> VEC_UINT;
        typedef uint32_t SCALAR_UINT_TYPE;
        typedef SIMDMask16 MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int64_t, 8>{
        typedef SIMDVecKNC_u<uint64_t, 8> VEC_UINT;
        typedef uint64_t SCALAR_UINT_TYPE;
        typedef SIMDMask8 MASK_TYPE;
    };

    // 1024b vectors
    template<>
    struct SIMDVecKNC_i_traits<int8_t, 128>{
        typedef SIMDVecKNC_u<uint8_t, 128> VEC_UINT;
        typedef uint8_t SCALAR_UINT_TYPE;
        typedef SIMDMask128 MASK_TYPE;
    };
    
    template<>
    struct SIMDVecKNC_i_traits<int16_t, 64>{
        typedef SIMDVecKNC_u<uint16_t, 64> VEC_UINT;
        typedef uint16_t SCALAR_UINT_TYPE;
        typedef SIMDMask64 MASK_TYPE;
    };
    
    template<>
    struct SIMDVecKNC_i_traits<int32_t, 32>{
        typedef SIMDVecKNC_u<uint32_t, 32> VEC_UINT;
        typedef uint32_t SCALAR_UINT_TYPE;
        typedef SIMDMask32 MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int64_t, 16>{
        typedef SIMDVecKNC_u<uint64_t, 16> VEC_UINT;
        typedef uint64_t SCALAR_UINT_TYPE;
        typedef SIMDMask16 MASK_TYPE;
    };

    template<typename SCALAR_INT_TYPE, uint32_t VEC_LEN>
    class SIMDVecKNC_i final : public SIMDVecSignedInterface< 
        SIMDVecKNC_i<SCALAR_INT_TYPE, VEC_LEN>, 
        typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, VEC_LEN>::VEC_UINT,
        SCALAR_INT_TYPE, 
        VEC_LEN,
        typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE,
        typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, VEC_LEN>::MASK_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_INT_TYPE, VEC_LEN>                            VEC_EMU_REG;
            
        typedef typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE     SCALAR_UINT_TYPE;
        typedef typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, VEC_LEN>::VEC_UINT             VEC_UINT;
        
        friend class SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, VEC_LEN>;
    private:
        VEC_EMU_REG mVec;

    public:
        inline SIMDVecKNC_i() : mVec() {};

        inline explicit SIMDVecKNC_i(SCALAR_INT_TYPE i) : mVec(i) {};

        inline SIMDVecKNC_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3) {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
        }

        inline SIMDVecKNC_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7) 
        {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
            mVec.insert(4, i4);  mVec.insert(5, i5);  mVec.insert(6, i6);  mVec.insert(7, i7);
        }

        inline SIMDVecKNC_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7,
                            SCALAR_INT_TYPE i8, SCALAR_INT_TYPE i9, SCALAR_INT_TYPE i10, SCALAR_INT_TYPE i11, SCALAR_INT_TYPE i12, SCALAR_INT_TYPE i13, SCALAR_INT_TYPE i14, SCALAR_INT_TYPE i15)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15); 
        }

        inline SIMDVecKNC_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7,
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
        inline SIMDVecKNC_i & insert(uint32_t index, SCALAR_INT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline  operator SIMDVecKNC_u<SCALAR_UINT_TYPE, VEC_LEN>() const {
            SIMDVecKNC_u<SCALAR_UINT_TYPE, VEC_LEN> retval;
            for(uint32_t i = 0; i < VEC_LEN; i++) {
                retval.insert(i, (SCALAR_UINT_TYPE)mVec[i]);
            }
            return retval;
        }
    };
    // ********************************************************************************************
    // SIGNED INTEGER VECTOR specializations
    // ********************************************************************************************

    template<>
    class SIMDVecKNC_i<int32_t, 16>: public SIMDVecSignedInterface<
        SIMDVecKNC_i<int32_t, 16>, 
        SIMDVecKNC_u<uint32_t, 16>,
        int32_t, 
        16,
        uint32_t,
        SIMDMask16>
    {
        friend class SIMDVecKNC_u<uint32_t, 16>;
        friend class SIMDVecKNC_f<float, 16>;
        friend class SIMDVecKNC_f<double, 16>;

    private:
        __m512i mVec;

        inline explicit SIMDVecKNC_i(__m512i & x) {
            this->mVec = x;
        }
    public:
        inline SIMDVecKNC_i() {};

        inline explicit SIMDVecKNC_i(int32_t i) {
            mVec = _mm512_set1_epi32(i);
        }

        inline SIMDVecKNC_i(int32_t i0,  int32_t i1,  int32_t i2,  int32_t i3, 
                            int32_t i4,  int32_t i5,  int32_t i6,  int32_t i7,
                            int32_t i8,  int32_t i9,  int32_t i10, int32_t i11,
                            int32_t i12, int32_t i13, int32_t i14, int32_t i15) 
        {
            mVec = _mm512_setr_epi32(i0, i1, i2,  i3,  i4,  i5,  i6,  i7, 
                                     i8, i9, i10, i11, i12, i13, i14, i15);
        }

        inline int32_t extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) int32_t raw[8];
            _mm512_store_si512(raw, mVec);
            return raw[index];
        }
            
        // Override Access operators
        inline int32_t operator[] (uint32_t index) const {
            return extract(index);
        }
                
        // insert[] (scalar)
        inline SIMDVecKNC_i & insert(uint32_t index, int32_t value) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
            alignas(32) int32_t raw[16];
            _mm512_store_si512(raw, mVec);
            raw[index] = value;
            mVec = _mm512_load_si512(raw);
            return *this;
        }

        inline  operator SIMDVecKNC_u<uint32_t, 16> const ();

        // assign (VEC) -> VEC        
        // assign (MASK, VEC) -> VEC
        // assign (VEC, scalar) -> VEC
        // assign (MASK, scalar) -> VEC
        // load (addr) -> VEC
        // load (MASK, addr) -> VEC
        // loadAligned (addrAligned) -> VEC
        // loadAligned (MASK, addrAligned) -> VEC
        // store(addr) -> scalar*
        // store(MASK, addr) -> scalar*
        // storeAligned(addrAligned) -> scalar*
        // storeAligned(MASK, addrAligned) -> scalar*
        // add(VEC) -> VEC
        // add (MASK, VEC) -> VEC
        // add (scalar) -> VEC
        // add (MASK, scalar) -> VEC
        // addAssign (VEC) -> VEC
        // addAssign (MASK, VEC) -> VEC
        // addAssign (scalar) -> VEC
        // addAssign (MASK, scalar) -> VEC
        // postfixIncrement () -> VEC
        // postfixIncrement (MASK) -> VEC
        // prefixIncrement () -> VEC
        // prefixIncrement (MASK) -> VEC
        // sub (VEC) -> VEC
        // sub(MASK, VEC) -> VEC
        // sub(scalar) -> VEC
        // sub(MASK, scalar) -> VEC
        // unaryMinus() -> VEC
        // unaryMinus(MASK) -> VEC
        // subAssign(VEC) -> VEC
        // subAssign (MASK, VEC) -> VEC
        // subAssign (scalar) -> VEC
        // subAssign(MASK, scalar) -> VEC
        // postfixDecrement() -> VEC
        // postfixDecrement (MASK) -> VEC
        // prefixDecrement() -> VEC
        // postfixDecrement (MASK, VEC) -> VEC
        // mult(VEC) -> VEC
        // mult(MASK, VEC) -> VEC
        // mult(scalar) -> VEC
        // mult(MASK, scalar) -> VEC
        // multAssign(VEC) -> VEC
        // multAssign (MASK, VEC) -> VEC
        // multAssign(scalar) -> VEC
        // div(VEC) -> VEC
        // div (MASK, VEC) -> VEC
        // div(scalar) -> VEC
        // div (MASK, scalar) -> VEC
        // divAssign (VEC, VEC) -> VEC
        // divAssign (MASK, VEC, VEC) -> VEC
        // divAssign (VEC, scalar) -> VEC
        // divAssign (MASK, VEC, scalar) -> VEC
        // reciprocal() -> VEC (implied nominator)
        // reciprocal(scalar) -> VEC
        // reciprocalAssign(scalar) -> VEC
        // isEqual (VEC, VEC) -> MASK
        // isEqual (VEC, scalar) -> MASK
        // isNotEqual (VEC, VEC) -> MASK
        // isNotEqual (VEC, scalar) -> MASK
        // isGreater (VEC, VEC) -> MASK
        // isGreater (VEC, scalar) -> MASK
        // isLesser (VEC, VEC) -> MASK
        // isLesser (VEC, scalar) -> MASK
        // isGreaterEqual (VEC, VEC) -> MASK
        // isGreaterEqual (VEC, scalar) -> MASK
        // isLesserEqual (VEC, VEC) -> MASK
        // isLesserEqual (VEC, scalar) -> MASK
        // binaryAnd(VEC) -> VEC
        // binaryAnd (MASK, VEC) -> VEC
        // binaryAnd (scalar) -> VEC
        // binaryAnd (MASK, scalar) -> VEC
        // binaryAndAssign (VEC) -> VEC
        // binaryAndAssign (MASK, VEC) -> VEC
        // binaryAndAssign (scalar) -> VEC
        // binaryAndAssign (MASK, scalar) -> VEC
        // binaryOr (VEC) -> VEC
        // binaryOr (MASK, VEC) -> VEC
        // binaryOr (scalar) -> VEC
        // binaryOr (MASK, scalar) -> VEC
        // binaryOrAssign (VEC) -> VEC
        // binaryOrAssign (MASK, VEC) -> VEC
        // binaryOrAssign (scalar) -> VEC
        // binaryOrAssign (MASK, scalar) -> VEC
        // binaryXor (VEC) -> VEC
        // binaryXor (MASK, VEC) -> VEC
        // binaryXor (scalar) -> VEC
        // binaryXor (MASK, scalar) -> VEC
        // binaryXorAssign (VEC) -> VEC
        // binaryXorAssign (MASK, VEC) -> VEC
        // binaryXorAssign (scalar) -> VEC
        // binaryXorAssign (MASK, scalar) -> VEC
        // binaryNot (VEC) -> VEC
        // binaryNot (MASK, VEC) -> VEC
        // binaryNotAssign () -> VEC
        // binaryNotAssign (MASK) -> VEC
        // blend(MASK, VEC) -> VEC
        // blend(MASK, scalar) -> VEC
        // blendAssign(MASK, VEC) -> VEC
        // blendAssign(MASK, scalar) -> VEC
        // reduceAdd () -> VEC
        // reduceAdd (MASK) -> scalar
        // reduceMult () -> scalar
        // reduceMult (MASK) -> scalar
        // reduceMult (scalar) -> scalar
        // reduceMult (MASK, scalar) -> scalar
        // reduceBinaryOr () -> scalar
        // reduceBinaryOr (MASK) -> scalar
        // reduceBinaryOr (scalar) -> scalar
        // reduceBinaryOr (MASK, scalar) -> scalar
        // reduceBinaryAnd () -> scalar
        // reduceBinaryAnd (MASK) -> scalar
        // reduceBinaryAnd (scalar) -> scalar
        // reduceBinaryAnd (MASK, scalar) -> scalar

        // gather (SCALAR_INT_TYPE*, uint64_t*) -> VEC
        // gather (MASK, SCALAR_INT_TYPE*, uint64_t*) -> VEC
        // gather (SCALAR_INT_TYPE*, VEC) -> VEC
        // gather (MASK, SCALAR_INT_TYPE*, VEC) -> VEC
        // scatter (SCALAR_INT_TYPE*, uint64_t*) -> scalar*
        // scatter (MASK, SCALAR_INT_TYPE*, uint64_t*) -> scalar*
        // scatter (SCALAR_INT_TYPE*, uint64_t*) -> scalar*
        // scatter (MASK, SCALAR_INT_TYPE*, uint64_t*) -> scalar*
        // shiftBitsLeft (VEC_UINT) -> VEC
        // shiftBitsLeft (MASK, VEC_UINT) -> VEC
        // shiftBitsLeft (scalar_uint) -> VEC
        // shiftBitsLeft (MASK, scalar_uint) -> VEC
        // shiftBitsLeftAssign (VEC_UINT) -> VEC
        // shiftBitsLeftAssign (MASK, VEC_UINT) -> VEC
        // shiftBitsLeftAssign (scalar_uint) -> VEC
        // shiftBitsLeftAssign (MASK, scalar_uint) -> VEC
        // shiftBitsRight (VEC_UINT) -> VEC
        // shiftBitsRight (MASK, VEC_INT, VEC_UINT) -> VEC
        // shiftBitsRight (scalar_uint) -> VEC
        // shiftBitsRight (MASK, scalar_uint) -> VEC
        // shiftBitsRightAssign (VEC_UINT) -> VEC
        // shiftBitsRightAssign (MASK, VEC_UINT) -> VEC
        // shiftBitsRightAssign (scalar_uint) -> VEC
        // shiftBitsRightAssign (MASK, scalar_uint) -> VEC
        // rotateBitsLeft (VEC_UINT) -> VEC
        // rotateBitsLeft(MASK, VEC_UINT) -> VEC
        // rotateBitsLeftScalar(scalar_uint) -> VEC
        // rotateBitsLeftScalar(MASK, scalar_uint) -> VEC
        // rotateBitsLeftAssign(VEC_UINT) -> VEC
        // rotateBitsLeftAssign(MASK, VEC_UINT) -> VEC
        // rotateBitsLeftAssign(scalar_uint) -> VEC
        // rotateBitsLeftAssign(MASK, scalar_uint) -> VEC

        // rotateBitsLeft (UINT_VEC) -> VEC
        // rotateBitsLeft(MASK, VEC_UINT) -> VEC
        // rotateBitsLeftScalar(scalar_uint) -> VEC
        // rotateBitsLeftScalar(MASK, scalar_uint) -> 
        // rotateBitsLeft(scalar, VEC_UINT) -> VEC
        // rotateBitsLeft(MASK, scalar, VEC_UINT) -> VEC
        // rotateBitsLeftAssign(VEC_UINT) -> VEC
        // rotateBitsLeftAssign(MASK, VEC_UINT) -> VEC
        // rotateBitsLeftAssign(scalar_uint) -> VEC
        // rotateBitsLeftAssign(MASK, scalar_uint) -> VEC

        // ******************************************************************
        // * Additional math functions
        // ******************************************************************

        // max (VEC) -> VEC
        // max (MASK, VEC) -> VEC
        // max (scalar) -> VEC
        // max (MASK, scalar) -> VEC
        // maxReduce () -> VEC
        // maxReduce (MASK) -> VEC
        // maxReduce (scalar) -> VEC
        // maxReduce (MASK, scalar) -> VEC
        // min (VEC) -> VEC
        // min (MASK, VEC) -> VEC
        // min (scalar) -> VEC
        // min (MASK, scalar) -> VEC
        // abs () -> VEC
        // abs (MASK) -> VEC
        // absAssign () -> VEC
        // absAssign (MASK) -> VEC  
    };

    inline SIMDVecKNC_i<int32_t, 16>::operator const SIMDVecKNC_u<uint32_t, 16>() {
        return SIMDVecKNC_u<uint32_t, 16>(this->mVec);
    }

    inline SIMDVecKNC_u<uint32_t, 16>::operator const SIMDVecKNC_i<int32_t, 16>() {
        return SIMDVecKNC_i<int32_t, 16>(this->mVec);
    }

    // ********************************************************************************************
    // FLOATING POINT VECTORS
    // ********************************************************************************************

    template<typename SCALAR_FLOAT_TYPE, uint32_t VEC_LEN>
    struct SIMDVecKNC_f_traits{
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };

    // 64b vectors
    template<>
    struct SIMDVecKNC_f_traits<float, 2>{
        typedef SIMDVecKNC_u<uint32_t, 2> VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int32_t, 2>  VEC_INT_TYPE;
        typedef int32_t                      SCALAR_INT_TYPE;
        typedef uint32_t                     SCALAR_UINT_TYPE;
        typedef float*                       SCALAR_TYPE_PTR;
        typedef SIMDMask2                    MASK_TYPE;
    };
    
    // 128b vectors
    template<>
    struct SIMDVecKNC_f_traits<float, 4>{
        typedef SIMDVecKNC_u<uint32_t, 4>  VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int32_t, 4>  VEC_INT_TYPE;
        typedef int32_t                   SCALAR_INT_TYPE;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef float*                    SCALAR_TYPE_PTR;
        typedef SIMDMask4                 MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_f_traits<double, 2>{
        typedef SIMDVecKNC_u<uint64_t, 2>  VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int64_t, 2>  VEC_INT_TYPE;
        typedef int64_t                   SCALAR_INT_TYPE;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef double*                   SCALAR_TYPE_PTR;
        typedef SIMDMask2                 MASK_TYPE;
    };

    // 256b vectors
    template<>
    struct SIMDVecKNC_f_traits<float, 8>{
        typedef SIMDVecKNC_u<uint32_t, 8>  VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int32_t, 8>  VEC_INT_TYPE;
        typedef int32_t                   SCALAR_INT_TYPE;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef float*                    SCALAR_TYPE_PTR;
        typedef SIMDMask8                 MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_f_traits<double, 4>{
        typedef SIMDVecKNC_u<uint64_t, 4>  VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int64_t, 4>  VEC_INT_TYPE;
        typedef int64_t                   SCALAR_INT_TYPE;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef double*                   SCALAR_TYPE_PTR;
        typedef SIMDMask4                 MASK_TYPE;
    };
    
    // 512b vectors
    template<>
    struct SIMDVecKNC_f_traits<float, 16>{
        typedef SIMDVecKNC_u<uint32_t, 16>  VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int32_t, 16> VEC_INT_TYPE;
        typedef int32_t                   SCALAR_INT_TYPE;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef float*                    SCALAR_TYPE_PTR;
        typedef SIMDMask16                MASK_TYPE;
    };
    
    template<>
    struct SIMDVecKNC_f_traits<double, 8>{
        typedef SIMDVecKNC_u<uint64_t, 8>  VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int64_t, 8>  VEC_INT_TYPE;
        typedef int64_t                   SCALAR_INT_TYPE;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef double*                   SCALAR_TYPE_PTR;
        typedef SIMDMask8                 MASK_TYPE;
    };

    // 1024b vectors
    template<>
    struct SIMDVecKNC_f_traits<float, 32>{
        typedef SIMDVecKNC_u<uint32_t,32>     VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int32_t, 32>     VEC_INT_TYPE;
        typedef int32_t                       SCALAR_INT_TYPE;
        typedef uint32_t                      SCALAR_UINT_TYPE;
        typedef float*                        SCALAR_TYPE_PTR;
        typedef SIMDMask32                    MASK_TYPE;
    };
    
    template<>
    struct SIMDVecKNC_f_traits<double, 16>{
        typedef SIMDVecKNC_u<uint64_t, 16>    VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int64_t, 16>     VEC_INT_TYPE;
        typedef int64_t                       SCALAR_INT_TYPE;
        typedef uint64_t                      SCALAR_UINT_TYPE;
        typedef double*                       SCALAR_TYPE_PTR;
        typedef SIMDMask16                    MASK_TYPE;
    };


    template<typename SCALAR_FLOAT_TYPE, uint32_t VEC_LEN>
    class SIMDVecKNC_f final : public SIMDVecFloatInterface<
        SIMDVecKNC_f<SCALAR_FLOAT_TYPE, VEC_LEN>, 
        typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_UINT_TYPE,
        typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_INT_TYPE,
        SCALAR_FLOAT_TYPE, 
        VEC_LEN,
        typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE,
        typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::MASK_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_FLOAT_TYPE, VEC_LEN>                            VEC_EMU_REG;
        typedef typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::MASK_TYPE       MASK_TYPE;
        
        typedef SIMDVecKNC_f VEC_TYPE;
    private:
        VEC_EMU_REG mVec;

    public:
        inline SIMDVecKNC_f() : mVec() {};

        inline explicit SIMDVecKNC_f(SCALAR_FLOAT_TYPE i) : mVec(i) {};

        inline SIMDVecKNC_f(SCALAR_FLOAT_TYPE i0, SCALAR_FLOAT_TYPE i1, SCALAR_FLOAT_TYPE i2, SCALAR_FLOAT_TYPE i3) {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
        }

        inline SIMDVecKNC_f(SCALAR_FLOAT_TYPE i0, SCALAR_FLOAT_TYPE i1, SCALAR_FLOAT_TYPE i2, SCALAR_FLOAT_TYPE i3, SCALAR_FLOAT_TYPE i4, SCALAR_FLOAT_TYPE i5, SCALAR_FLOAT_TYPE i6, SCALAR_FLOAT_TYPE i7) 
        {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
            mVec.insert(4, i4);  mVec.insert(5, i5);  mVec.insert(6, i6);  mVec.insert(7, i7);
        }

        inline SIMDVecKNC_f(SCALAR_FLOAT_TYPE i0, SCALAR_FLOAT_TYPE i1, SCALAR_FLOAT_TYPE i2, SCALAR_FLOAT_TYPE i3, SCALAR_FLOAT_TYPE i4, SCALAR_FLOAT_TYPE i5, SCALAR_FLOAT_TYPE i6, SCALAR_FLOAT_TYPE i7,
                            SCALAR_FLOAT_TYPE i8, SCALAR_FLOAT_TYPE i9, SCALAR_FLOAT_TYPE i10, SCALAR_FLOAT_TYPE i11, SCALAR_FLOAT_TYPE i12, SCALAR_FLOAT_TYPE i13, SCALAR_FLOAT_TYPE i14, SCALAR_FLOAT_TYPE i15)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15); 
        }

        inline SIMDVecKNC_f(SCALAR_FLOAT_TYPE i0, SCALAR_FLOAT_TYPE i1, SCALAR_FLOAT_TYPE i2, SCALAR_FLOAT_TYPE i3, SCALAR_FLOAT_TYPE i4, SCALAR_FLOAT_TYPE i5, SCALAR_FLOAT_TYPE i6, SCALAR_FLOAT_TYPE i7,
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
        inline SIMDVecKNC_f & insert(uint32_t index, SCALAR_FLOAT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

    };

    // ********************************************************************************************
    // FLOATING POINT VECTOR specializations
    // ********************************************************************************************

    template<>
    class SIMDVecKNC_f<float, 16> : public SIMDVecFloatInterface<
        SIMDVecKNC_f<float, 16>, 
        SIMDVecKNC_u<uint32_t, 16>,
        SIMDVecKNC_i<int32_t, 16>,
        float, 
        16,
        uint32_t,
        SIMDMask16>
    {
    private:
        __m512 mVec;

        inline SIMDVecKNC_f(__m512 & x) {
            this->mVec = x;
        }

    public:
        inline SIMDVecKNC_f() {
            mVec = _mm512_setzero_ps();
        }

        inline explicit SIMDVecKNC_f(float f) {
            mVec = _mm512_set1_ps(f);
        }

        inline SIMDVecKNC_f(float f0, float f1, float f2,  float f3,  float f4,  float f5,  float f6,  float f7,
                            float f8, float f9, float f10, float f11, float f12, float f13, float f14, float f15) {
            mVec = _mm512_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15);
        }

        inline float extract (uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) float raw[8];
            _mm512_store_ps(raw, mVec);
            return raw[index];
        }

        // Override Access operators
        inline float operator[] (uint32_t index) const {

            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }
                
        // insert[] (scalar)
        inline SIMDVecKNC_f & insert(uint32_t index, float value) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) float raw[8];
            //_mm512_store_ps(raw, mVec);
            raw[index] = value;
            //mVec = _mm512_load_ps(raw);
            return *this;
        }
        
        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        // assign (VEC) -> VEC
        // assign (MASK, VEC) -> VEC
        // assign (VEC, scalar) -> VEC
        // assign (MASK, scalar) -> VEC
        // load (addr) -> VEC
        // load (MASK, addr) -> VEC
        // loadAligned (addrAligned) -> VEC
        // loadAligned (MASK, addrAligned) -> VEC
        // store (addr) -> scalar*
        // store (MASK, addr) -> scalar*
        // storeAligned (addrAligned) -> scalar*
        // storeAligned (MASK, addrAligned) -> scalar*
        // add (VEC) -> VEC
        // add (MASK, VEC) -> VEC
        // add (scalar) -> VEC
        // add (MASK, scalar) -> VEC
        // addAssign (VEC) -> VEC
        // addAssign (MASK, VEC) -> VEC
        // addAssign (scalar) -> VEC
        // addAssign (MASK, scalar) -> VEC
        // postfixIncrement () -> VEC
        // postfixIncrement (MASK) -> VEC
        // prefixIncrement () -> VEC
        // prefixIncrement (MASK) -> VEC
        // sub (VEC) -> VEC
        // sub (MASK, VEC) -> VEC
        // sub (scalar) -> VEC
        // sub (MASK, scalar) -> VEC
        // unaryMinus () -> VEC
        // unaryMinus (MASK) -> VEC
        // subAssign (VEC) -> VEC
        // subAssign (MASK, VEC) -> VEC
        // subAssign (scalar) -> VEC
        // subAssign (MASK, scalar) -> VEC
        // postfixDecrement () -> VEC
        // postfixDecrement (MASK) -> VEC
        // prefixDecrement () -> VEC
        // postfixDecrement (MASK, VEC) -> VEC
        // mult (VEC) -> VEC
        // mult(MASK, VEC) -> VEC
        // mult(scalar) -> VEC
        // mult(MASK, scalar) -> VEC
        // multAssign(VEC) -> VEC
        // multAssign (MASK, VEC) -> VEC
        // multAssign(scalar) -> VEC
        // div(VEC) -> VEC
        // div (MASK, VEC) -> VEC
        // div(scalar) -> VEC
        // div (MASK, scalar) -> VEC
        // divAssign (VEC, VEC) -> VEC
        // divAssign (MASK, VEC, VEC) -> VEC
        // divAssign (VEC, scalar) -> VEC
        // divAssign (MASK, VEC, scalar) -> VEC
        // reciprocal() -> VEC (implied nominator)
        // reciprocal(scalar) -> VEC
        // reciprocalAssign(scalar) -> VEC
        // isEqual (VEC, VEC) -> MASK
        // isEqual (VEC, scalar) -> MASK
        // isNotEqual (VEC, VEC) -> MASK
        // isNotEqual (VEC, scalar) -> MASK
        // isGreater (VEC, VEC) -> MASK
        // isGreater (VEC, scalar) -> MASK
        // isLesser (VEC, VEC) -> MASK
        // isLesser (VEC, scalar) -> MASK
        // isGreaterEqual (VEC, VEC) -> MASK
        // isGreaterEqual (VEC, scalar) -> MASK
        // isLesserEqual (VEC, VEC) -> MASK
        // isLesserEqual (VEC, scalar) -> MASK
        // binaryAnd(VEC) -> VEC
        // binaryAnd (MASK, VEC) -> VEC
        // binaryAnd (scalar) -> VEC
        // binaryAnd (MASK, scalar) -> VEC
        // binaryAndAssign (VEC) -> VEC
        // binaryAndAssign (MASK, VEC) -> VEC
        // binaryAndAssign (scalar) -> VEC
        // binaryAndAssign (MASK, scalar) -> VEC
        // binaryOr (VEC) -> VEC
        // binaryOr (MASK, VEC) -> VEC
        // binaryOr (scalar) -> VEC
        // binaryOr (MASK, scalar) -> VEC
        // binaryOrAssign (VEC) -> VEC
        // binaryOrAssign (MASK, VEC) -> VEC
        // binaryOrAssign (scalar) -> VEC
        // binaryOrAssign (MASK, scalar) -> VEC
        // binaryXor (VEC) -> VEC
        // binaryXor (MASK, VEC) -> VEC
        // binaryXor (scalar) -> VEC
        // binaryXor (MASK, scalar) -> VEC
        // binaryXorAssign (VEC) -> VEC
        // binaryXorAssign (MASK, VEC) -> VEC
        // binaryXorAssign (scalar) -> VEC
        // binaryXorAssign (MASK, scalar) -> VEC
        // binaryNot (VEC) -> VEC
        // binaryNot (MASK, VEC) -> VEC
        // binaryNotAssign () -> VEC
        // binaryNotAssign (MASK) -> VEC
        // blend(MASK, VEC) -> VEC
        // blend(MASK, scalar) -> VEC
        // blendAssign(MASK, VEC) -> VEC
        // blendAssign(MASK, scalar) -> VEC
        // reduceAdd () -> VEC
        // reduceAdd (MASK) -> scalar
        // reduceMult () -> scalar
        // reduceMult (MASK) -> scalar
        // reduceMult (scalar) -> scalar
        // reduceMult (MASK, scalar) -> scalar
        // reduceBinaryOr () -> scalar
        // reduceBinaryOr (MASK) -> scalar
        // reduceBinaryOr (scalar) -> scalar
        // reduceBinaryOr (MASK, scalar) -> scalar
        // reduceBinaryAnd () -> scalar
        // reduceBinaryAnd (MASK) -> scalar
        // reduceBinaryAnd (scalar) -> scalar
        // reduceBinaryAnd (MASK, scalar) -> scalar
        
        // ******************************************************************
        // * Additional math functions
        // ******************************************************************

        // max (VEC) -> VEC
        // max (MASK, VEC) -> VEC
        // max (scalar) -> VEC
        // max (MASK, scalar) -> VEC
        // maxReduce () -> VEC
        // maxReduce (MASK) -> VEC
        // maxReduce (scalar) -> VEC
        // maxReduce (MASK, scalar) -> VEC
        // min (VEC) -> VEC
        // min (MASK, VEC) -> VEC
        // min (scalar) -> VEC
        // min (MASK, scalar) -> VEC
        // abs () -> VEC
        // abs (MASK) -> VEC
        // absAssign () -> VEC
        // absAssign (MASK) -> VEC  

        // *******************************************************************
        // * Additional math functions for FLOATING vectors
        // *******************************************************************

        // truncToInt() -> VEC_INT
        // sqrt() -> VEC
        // sqrt(MASK) -> VEC
        // sqrtAssign() -> VEC 
        // sqrtAssign(MASK) -> VEC
        // sin() -> VEC
        // sin(MASK) -> VEC
        // sinAssign() -> VEC
        // sinAssign(MASK) -> VEC
        // cos() -> VEC
        // cos(MASK) -> VEC
        // cosAssign() -> VEC
        // cosAssign(MASK) -> VEC
    };

    // 64b uint vectors
    typedef SIMDVecKNC_u<uint8_t,  8>   SIMD8_8u;
    typedef SIMDVecKNC_u<uint16_t, 4>   SIMD4_16u;
    typedef SIMDVecKNC_u<uint32_t, 2>   SIMD2_32u; 

    // 128b uint vectors
    typedef SIMDVecKNC_u<uint8_t,  16>  SIMD16_8u;
    typedef SIMDVecKNC_u<uint16_t, 8>   SIMD8_16u;
    typedef SIMDVecKNC_u<uint32_t, 4>   SIMD4_32u;
    typedef SIMDVecKNC_u<uint64_t, 2>   SIMD2_64u;
    
    // 256b uint vectors
    typedef SIMDVecKNC_u<uint8_t,  32>  SIMD32_8u;
    typedef SIMDVecKNC_u<uint16_t, 16>  SIMD16_16u;
    typedef SIMDVecKNC_u<uint32_t, 8>   SIMD8_32u;
    typedef SIMDVecKNC_u<uint64_t, 4>   SIMD4_64u;
    
    // 512b uint vectors
    typedef SIMDVecKNC_u<uint8_t,  64>  SIMD64_8u;
    typedef SIMDVecKNC_u<uint16_t, 32>  SIMD32_16u;
    typedef SIMDVecKNC_u<uint32_t, 16>  SIMD16_32u;
    typedef SIMDVecKNC_u<uint64_t, 8>   SIMD8_64u;

    // 1024b uint vectors
    typedef SIMDVecKNC_u<uint8_t,  128> SIMD128_8u;
    typedef SIMDVecKNC_u<uint16_t, 64>  SIMD64_16u;
    typedef SIMDVecKNC_u<uint32_t, 32>  SIMD32_32u;
    typedef SIMDVecKNC_u<uint64_t, 16>  SIMD16_64u;

    // 64b int vectors
    typedef SIMDVecKNC_i<int8_t,   8>   SIMD8_8i; 
    typedef SIMDVecKNC_i<int16_t,  4>   SIMD4_16i;
    typedef SIMDVecKNC_i<int32_t,  2>   SIMD2_32i;

    // 128b int vectors
    typedef SIMDVecKNC_i<int8_t,   16>  SIMD16_8i; 
    typedef SIMDVecKNC_i<int16_t,  8>   SIMD8_16i;
    typedef SIMDVecKNC_i<int32_t,  4>   SIMD4_32i;
    typedef SIMDVecKNC_i<int64_t,  2>   SIMD2_64i;

    // 256b int vectors
    typedef SIMDVecKNC_i<int8_t,   32>  SIMD32_8i;
    typedef SIMDVecKNC_i<int16_t,  16>  SIMD16_16i;
    typedef SIMDVecKNC_i<int32_t,  8>   SIMD8_32i;
    typedef SIMDVecKNC_i<int64_t,  4>   SIMD4_64i;
    
    // 512b int vectors
    typedef SIMDVecKNC_i<int8_t,   64>  SIMD64_8i;
    typedef SIMDVecKNC_i<int16_t,  32>  SIMD32_16i;
    typedef SIMDVecKNC_i<int32_t,  16>  SIMD16_32i;
    typedef SIMDVecKNC_i<int64_t,  8>   SIMD8_64i;

    // 1024b int vectors
    typedef SIMDVecKNC_i<int8_t,   128> SIMD128_8i;
    typedef SIMDVecKNC_i<int16_t,  64>  SIMD64_16i;
    typedef SIMDVecKNC_i<int32_t,  32>  SIMD32_32i;
    typedef SIMDVecKNC_i<int64_t,  16>  SIMD16_64i;


    // 64b float vectors
    typedef SIMDVecKNC_f<float, 2>      SIMD2_32f;

    // 128b float vectors
    typedef SIMDVecKNC_f<float,  4>     SIMD4_32f;
    typedef SIMDVecKNC_f<double, 2>     SIMD2_64f;

    // 256b float vectors
    typedef SIMDVecKNC_f<float,  8>     SIMD8_32f;
    typedef SIMDVecKNC_f<double, 4>     SIMD4_64f;

    // 512b float vectors
    typedef SIMDVecKNC_f<float,  16>    SIMD16_32f;
    typedef SIMDVecKNC_f<double, 8>     SIMD8_64f;

    // 1024b float vectors
    typedef SIMDVecKNC_f<float, 32>     SIMD32_32f;
    typedef SIMDVecKNC_f<double, 16>    SIMD16_64f;
} // SIMD
} // UME

#endif // UME_SIMD_PLUGIN_NATIVE_KNC_H_
