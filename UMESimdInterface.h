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

#ifndef UME_SIMD_INTERFACE_H_
#define UME_SIMD_INTERFACE_H_

#include <cmath>
#include <limits>

#include "UMEBasicTypes.h"

#if defined (_MSC_VER)
// WORKAROUND: Visual studio 2012 does not provide implementation for c++ 11 std::trunc, but VS2013 already has it.
#if _MSC_VER < 1800
namespace std
{
    inline float       trunc( float f ) { return (f>0) ? floor(f) : ceil(f); }
    inline double      trunc( double d ) { return (d>0) ? floor(d) : ceil(d); }
    inline long double trunc( long double ld ) { return (ld>0) ? floor(ld) : ceil(ld); }
    float round(float d) { return static_cast<float>(static_cast<int>(d + 0.5f)); }
    double round(double d) { return static_cast<double>(static_cast<long>(d + 0.5)); }
    //double      trunc( Integral arg );
    inline bool       isfinite( float f) { return _finite((double)f) != 0 ? true : false; }
    inline bool       isfinite( double f) { return _finite(f) != 0 ? true : false; }
}
#endif
#endif

/*
#if defined (_MSC_VER)
    #pragma loop(ivdep)
#elif defined (__GNUC__)
    #pragma GCC ivdep
#elif defined (__ICC) || defined(__INTEL_COMPILER)
    #pragma ivdep
#endif
    */

namespace UME
{
namespace SIMD
{

    // All functions in this namespace will have one purpose: emulation of single function in different backends.
    //   Scalar emulation plugin has to emulate all of these features using scalar values either way. Spliting
    //   Functionality implementation from class implementation will allow re-use of the operator functions for 
    //   other backends. This will decrease overall amount of code, and remove potential, repeated errors in plugins.
    namespace EMULATED_FUNCTIONS
    {
        // assign (VEC, VEC) -> VEC
        template<typename VEC_TYPE>
        inline VEC_TYPE & assign(VEC_TYPE & dst, VEC_TYPE const & src) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, src[i]);
            }
            return dst;
        }

        // assign (mask, VEC, VEC) -> VEC
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & assign(MASK_TYPE const & mask, VEC_TYPE & dst, VEC_TYPE const & src) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) dst.insert(i, src[i]);
            }
            return dst;
        }

        // assign (VEC, scalar) -> VEC
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE & assign(VEC_TYPE & dst, SCALAR_TYPE src) {
            UME_EMULATION_WARNING();
            for( uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, src);
            }
            return dst;
        }

        // assign (MASK, VEC, scalar) -> VEC
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & assign(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE src) {
            UME_EMULATION_WARNING();
            for( uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) dst.insert(i, src);
            }
            return dst;
        }

        // LOAD
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE & load(VEC_TYPE & dst, SCALAR_TYPE const * p) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, p[i]);
            }
            return dst;
        }
            
        // MLOAD
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & load(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE const * p) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++)
            {
                if(mask[i] == true) dst.insert(i, p[i]);
            }
            return dst;
        }
        
        // LOADA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE & loadAligned(VEC_TYPE & dst, SCALAR_TYPE const * p) {
            UME_ALIGNMENT_CHECK(p, VEC_TYPE::alignment());
            return EMULATED_FUNCTIONS::load<VEC_TYPE, SCALAR_TYPE>(dst, p);
        }
        
        // MLOADA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & loadAligned(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE const * p) {
            UME_ALIGNMENT_CHECK(p, VEC_TYPE::alignment());
            return EMULATED_FUNCTIONS::load<VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, dst, p);
        }

        // STORE
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline SCALAR_TYPE* store(VEC_TYPE & src, SCALAR_TYPE * p) {
            typedef decltype(src[0])* SCALAR_TYPE_PTR;
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++)
            {
                ((SCALAR_TYPE_PTR)p)[i] = src[i];
            }
            return p;
        }

        // MSTORE
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline SCALAR_TYPE* store(MASK_TYPE const & mask, VEC_TYPE & src, SCALAR_TYPE * p) {
            typedef decltype(src[0])* SCALAR_TYPE_PTR;
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++)
            {
                if(mask[i] == true) ((SCALAR_TYPE_PTR)p)[i] = src[i];
            }
            return p;
        }

        // STOREA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline SCALAR_TYPE* storeAligned(VEC_TYPE & src, SCALAR_TYPE *p) {
            UME_ALIGNMENT_CHECK(p, VEC_TYPE::alignment());
            return store<VEC_TYPE, SCALAR_TYPE>(src, p); 
        }

        // MSTOREA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline SCALAR_TYPE* storeAligned(MASK_TYPE const & mask, VEC_TYPE & src, SCALAR_TYPE *p) {
            UME_ALIGNMENT_CHECK(p, VEC_TYPE::alignment());
            return store<MASK_TYPE, VEC_TYPE, SCALAR_TYPE>(mask, src, p);
        }

        // GATHERS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE & gather(VEC_TYPE & dst, SCALAR_TYPE* base, uint64_t* indices) {
            UME_EMULATION_WARNING();
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert( i, base[indices[i]]);
            }
            return dst;
        }
        
        // MGATHERS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & gather(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE* base, uint64_t* indices) {
            UME_EMULATION_WARNING();
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) dst.insert( i, base[indices[i]]);
            }
            return dst;
        }

        // GATHERV
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE>
        inline VEC_TYPE & gather(VEC_TYPE & dst, SCALAR_TYPE* base, UINT_VEC_TYPE const & indices) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, base[indices[i]]);
            }
            return dst;
        }

        // MGATHERV
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & gather(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE* base, UINT_VEC_TYPE const & indices) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) dst.insert(i, base[indices[i]]);
            }
            return dst;
        }

        // SCATTERS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline SCALAR_TYPE* scatter(VEC_TYPE const & src, SCALAR_TYPE* base, uint64_t* indices) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                base[indices[i]] = src[i];
            }
            return base;
        }
        
        // MSCATTERS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline SCALAR_TYPE* scatter(MASK_TYPE const & mask, VEC_TYPE const & src, SCALAR_TYPE* base, uint64_t* indices) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) base[indices[i]] = src[i];
            }
            return base;
        }
        
        // SCATTERV
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE>
        inline SCALAR_TYPE* scatter(VEC_TYPE const & src, SCALAR_TYPE* base, UINT_VEC_TYPE const & indices) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++)
            {
                base[indices[i]] = src[i];
            }
            return base;
        }

        // MSCATTERV
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        inline SCALAR_TYPE* scatter(MASK_TYPE const & mask, VEC_TYPE const & src, SCALAR_TYPE* base, UINT_VEC_TYPE const & indices) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++)
            {
                if(mask[i] == true) base[indices[i]] = src[i];
            }
            return base;
        }

        // pack (VEC, VEC_HALF_TYPE, VEC_HALF_TYPE)
        template<typename VEC_TYPE, typename VEC_HALF_TYPE>
        inline VEC_TYPE & pack(VEC_TYPE & dst, VEC_HALF_TYPE const & src1, VEC_HALF_TYPE const & src2) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_HALF_TYPE::length(); i++) {
                dst.insert(i, src1[i]);
                dst.insert(i + VEC_HALF_TYPE::length(), src2[i]);
            }
        }

        // TOOD:
        // pack (VEC, VEC_QUARTER_LEN, VEC_QUARTER_LEN, VEC_QUARTER_LEN, VEC_QUARTER_LEN)
        // ...
        // packLow (VEC, VEC_HALF_LEN)
        // packHigh (VEC, VEC_HALF_LEN)
            
        // unpack (VEC, VEC_HALF_LEN, VEC_HALF_LEN) 
        template<typename VEC_TYPE, typename VEC_HALF_TYPE>
        inline void unpack(VEC_TYPE const & src, VEC_HALF_TYPE & dst1, VEC_HALF_TYPE & dst2) {
            UME_EMULATION_WARNING();
            uint32_t halfLength = VEC_HALF_TYPE::length();
            for(uint32_t i = 0; i < halfLength; i++) {
                dst1.insert(i, src[i]);
                dst2.insert(i, src[i + halfLength]);
            }
        }

        // unpack (VEC, VEC_QUARTER_LEN, VEC_QUARTER_LEN, VEC_QUARTER_LEN, VEC_QUARTER_LEN)
        // ...
        // unpackLow (VEC, VEC_HALF_LEN)
        // unpackHigh (VEC, VEC_HALF_LEN)

        // ADDV
        template<typename VEC_TYPE>
        inline VEC_TYPE add (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] + b[i]);
            }
            return retval; 
        }
        
        // MADDV
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE add (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, mask[i] ? a[i] + b[i] : a[i]);
            }
            return retval;
        }

        // ADDS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE addScalar (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] + b);
            }
            return retval;
        }

        // MADDS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE addScalar (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, mask[i] ? a[i] + b : a[i]);
            }
            return retval;
        }
            
        // ADDA
        template<typename VEC_TYPE>
        inline VEC_TYPE & addAssign (VEC_TYPE & src, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) { src.insert(i, (src[i] + b[i])); }
            return src;
        }

        // MADDA
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & addAssign (MASK_TYPE const & mask, VEC_TYPE & src, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) src.insert(i, (src[i] + b[i]));
            }
            return src;
        }

        // ADDSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE & addAssignScalar (VEC_TYPE & src, SCALAR_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                src.insert(i, (src[i] + b));
            }
            return src;
        }

        // MADDSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & addAssignScalar (MASK_TYPE const & mask, VEC_TYPE & src, SCALAR_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) src.insert(i, src[i] + b);
            }
            return src;
        }

        // POSTINC
        template<typename VEC_TYPE>
        inline VEC_TYPE postfixIncrement(VEC_TYPE & a) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i] + 1);
            }
            return retval;
        }

        // MPOSTINC
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE postfixIncrement(MASK_TYPE const & mask, VEC_TYPE & a) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i] + 1);
            }
            return retval;
        }

        // PREFINC
        template<typename VEC_TYPE>
        inline VEC_TYPE & prefixIncrement(VEC_TYPE & a) {
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++)
            {
                a.insert(i, a[i] + 1);
            }
            return a;
        }

        // MPREFINC
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & prefixIncrement(MASK_TYPE const & mask, VEC_TYPE & a) {
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++)
            {
                if(mask[i] == true) a.insert(i, a[i] + 1);
            }
            return a;
        }

        // SUBV
        template<typename VEC_TYPE>
        inline VEC_TYPE sub ( VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] - b[i]);
            }
            return retval; 
        }

        // MSUBV
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE sub ( MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) retval.insert(i, a[i] - b[i]);
            }
            return retval; 
        }

        // SUBS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE subScalar ( VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (a[i] - b));
            }
            return retval;
        }

        // MSUBS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE subScalar ( MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) retval.insert(i, (a[i] - b));
            }
            return retval;
        }

        // NEG
        template<typename VEC_TYPE>
        inline VEC_TYPE unaryMinus (VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, -a[i]);
            }
            return retval;
        }

        // MNEG
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE unaryMinus (MASK_TYPE const & mask, VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if( mask[i] == true ) retval.insert(i, -a[i]);
            }
            return retval;
        }
            
        // SUBVA
        template<typename VEC_TYPE>
        inline VEC_TYPE & subAssign (VEC_TYPE & dst, VEC_TYPE const & b)
        {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, dst[i] - b[i]);
            }
            return dst;
        }

        // MSUBVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & subAssign (MASK_TYPE const & mask, VEC_TYPE & dst, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true ) dst.insert(i, dst[i] - b[i]);
            }
            return dst;
        }

        // SUBSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE & subAssign (VEC_TYPE & dst, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, dst[i] - b);
            }
            return dst;
        }

        // MSUBSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & subAssign (MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) dst.insert(i, dst[i] - b);
            }
            return dst;
        }

        // POSTDEC
        template<typename VEC_TYPE>
        inline VEC_TYPE postfixDecrement(VEC_TYPE & a) {
            VEC_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i] - 1);
            }
            return retval;
        }

        // MPOSTDEC
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE postfixDecrement(MASK_TYPE const & mask, VEC_TYPE & a) {
            VEC_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i] - 1);
            }
            return retval;
        }

        // PREFDEC
        template<typename VEC_TYPE>
        inline VEC_TYPE & prefixDecrement(VEC_TYPE & a) {
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i]-1 );
            }
            return a;
        }

        // MPREFDEC
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & prefixDecrement(MASK_TYPE const & mask, VEC_TYPE & a) {
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i]-1 );
            }
            return a;
        }
            
        // MULV
        template<typename VEC_TYPE>
        inline VEC_TYPE mult (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++)
            {
                retval.insert(i, a[i]*b[i] );
            }
            return retval;
        }

        // MMULV
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE mult (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? a[i]*b[i] : a[i] );
            }
            return retval;
        }

        // MULS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE mult (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]*b );
            }
            return retval;
        }

        // MMULS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE mult (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? a[i]*b : a[i]);
            }
            return retval;
        }

        // MULVA
        template<typename VEC_TYPE>
        inline VEC_TYPE & multAssign (VEC_TYPE & dst, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, dst[i] * b[i]);
            }
            return dst;
        }

        // MMULVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & multAssign (MASK_TYPE const & mask, VEC_TYPE & dst, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, dst[i] * b[i]);
            }
            return dst;
        }

        // MULSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE & multAssign (VEC_TYPE & dst, SCALAR_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, dst[i] * b);
            }
            return dst;
        }

        // MMULSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & multAssign (MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) dst.insert(i, dst[i] * b);
            }
            return dst;
        }

        // DIVV
        template<typename VEC_TYPE>
        inline VEC_TYPE div (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]/b[i] );
            }
            return retval;
        }

        // MDIVV
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE div (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? a[i]/b[i] : a[i]);
            }
            return retval;
        }
        
        // DIVS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE div (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]/b );
            }
            return retval;
        }

        // MDIVS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE div (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? (a[i]/b) : a[i]);
            }
            return retval;
        }

        // RCP
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE div (SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a/b[i] );
            }
            return retval;
        }

        // MRPC
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE div (MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ?  (a/b[i]) : a);
            }
            return retval;
        }
         
        // DIVVA
        template<typename VEC_TYPE>
        inline VEC_TYPE & divAssign(VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i]/b[i] );
            }
            return a;
        }

        // MDIVVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & divAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i]/b[i] );
            }
            return a;
        }
            
        // DIVSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE & divAssign(VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i]/b );
            }
            return a;
        }

        // MDIVSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & divAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i]/b);
            }
        }

        // LSHV
        template<typename VEC_TYPE, typename UINT_VEC_TYPE>
        inline VEC_TYPE shiftBitsLeft(VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] << b[i] );
            }
            return retval;
        }
            
        // MLSHV
        template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE shiftBitsLeft(MASK_TYPE const & mask, VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? (a[i] << b[i]) : a[i]);
            }
            return retval;
        }
        
        // LSHS
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE>
        inline VEC_TYPE shiftBitsLeftScalar(VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]<<b );
            }
            return retval;
        }
        
        // MLSHS
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
        inline VEC_TYPE shiftBitsLeftScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? (a[i] << b ) : a[i]);
            }
            return retval;
        }

        // LSHVA
        template<typename VEC_TYPE, typename UINT_VEC_TYPE>
        inline VEC_TYPE & shiftBitsLeftAssign(VEC_TYPE & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i]<< b[i] );
            }
            return a;
        }

        // MLSHVA
        template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & shiftBitsLeftAssign(MASK_TYPE const & mask, VEC_TYPE & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i]<<b[i] );
            }
            return a;
        }
    
        // LSHSA
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE>
        inline VEC_TYPE & shiftBitsLeftAssignScalar(VEC_TYPE & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i]<<b );
            }
            return a;
        }

        // MLSHSA
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & shiftBitsLeftAssignScalar(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i]<<b );
            }
            return a;
        }
            
        // RSHV
        template<typename VEC_TYPE, typename UINT_VEC_TYPE>
        inline VEC_TYPE shiftBitsRight(VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]>>b[i] );
            }
            return retval;
        }

        // MRSHV
        template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE shiftBitsRight(MASK_TYPE const & mask, VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? (a[i]>>b[i]) : a[i]);
            }
            return retval;
        }

        // RSHS
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE>
        inline VEC_TYPE shiftBitsRightScalar(VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] >> b );
            }
            return retval;
        }

        // MRSHS
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
        inline VEC_TYPE shiftBitsRightScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? (a[i] >> b) : a[i] );
            }
            return retval;
        }
                        
        // RSHVA
        template<typename VEC_TYPE, typename UINT_VEC_TYPE>
        inline VEC_TYPE & shiftBitsRightAssign(VEC_TYPE & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i]>>b[i] );
            }
            return a;
        }
            
        // MRSHVA
        template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & shiftBitsRightAssign(MASK_TYPE const & mask, VEC_TYPE & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i]>>b[i] );
            }
            return a;
        }
            
        // RSHSA
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE>
        inline VEC_TYPE & shiftBitsRightAssignScalar(VEC_TYPE & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i]>> b );
            }
            return a;
        }
                
        // MSRHSA
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & shiftBitsRightAssignScalar(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i]>> b );
            }
            return a;
        }
            
        // ROLV
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE>
        inline VEC_TYPE rotateBitsLeft(VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            uint32_t bitLength = 8*sizeof(SCALAR_TYPE);
            SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
            bool topBit;
            SCALAR_TYPE shifted;
            
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                shifted = a[i];
                // shift one bit at a time. This simplifies type dependency checks.
                for(uint32_t j = 0; j < b[i]; j++) {
                  if( (shifted & topBitMask) != 0) topBit = true;
                  else topBit = false;
                  
                  shifted <<= 1;        
                  if(topBit == true) shifted |= SCALAR_TYPE(1);
                  else               shifted &= ~(SCALAR_TYPE(1)); 
                }
                retval.insert(i, shifted); 
            }
            return retval;
        }

        // MROLV
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE rotateBitsLeft(MASK_TYPE const & mask, VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            uint32_t bitLength = 8*sizeof(SCALAR_TYPE);
            SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
            bool topBit;
            SCALAR_TYPE shifted;
            
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true)
                {
                    shifted = a[i];
                    // shift one bit at a time. This simplifies type dependency checks.
                    for(uint32_t j = 0; j < b[i]; j++) {
                      if( (shifted & topBitMask) != 0) topBit = true;
                      else topBit = false;
                      
                      shifted <<= 1;        
                      if(topBit == true) shifted |= SCALAR_TYPE(1);
                      else               shifted &= ~(SCALAR_TYPE(1)); 
                    }
                    retval.insert(i, shifted);
                } 
                else
                {
                    retval.insert(i, a[i]);
                }
            }
            return retval;
        }

        // ROLS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename SCALAR_UINT_TYPE>
        inline VEC_TYPE rotateBitsLeftScalar(VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            uint32_t bitLength = 8*sizeof(SCALAR_TYPE);
            SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
            bool topBit;
            SCALAR_TYPE shifted;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                shifted = a[i];
                // shift one bit at a time. This simplifies type dependency checks.
                for(uint32_t j = 0; j < b; j++) {
                  if( (shifted & topBitMask) != 0) topBit = true;
                  else topBit = false;
                  
                  shifted <<= 1;        
                  if(topBit == true) shifted |= SCALAR_TYPE(1);
                  else               shifted &= ~(SCALAR_TYPE(1)); 
                }
                retval.insert(i, shifted); 
            }
            return retval;
        }

        // MROLS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename SCALAR_UINT_TYPE,  typename MASK_TYPE>
        inline VEC_TYPE rotateBitsLeftScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            uint32_t bitLength = 8*sizeof(SCALAR_TYPE);
            SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
            bool topBit;
            SCALAR_TYPE shifted;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true)
                {
                    shifted = a[i];
                    // shift one bit at a time. This simplifies type dependency checks.
                    for(uint32_t j = 0; j < b; j++) {
                      if( (shifted & topBitMask) != 0) topBit = true;
                      else topBit = false;
                      
                      shifted <<= 1;        
                      if(topBit == true) shifted |= SCALAR_TYPE(1);
                      else               shifted &= ~(SCALAR_TYPE(1)); 
                    }
                    retval.insert(i, shifted);
                }
                else 
                {
                    retval.insert(i, a[i]);
                } 
            }
            return retval;
        }

        // ROLVA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE>
        inline VEC_TYPE & rotateBitsLeftAssign(VEC_TYPE & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            uint32_t bitLength = 8*sizeof(SCALAR_TYPE);
            SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
            bool topBit;
            SCALAR_TYPE shifted;
            
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                shifted = a[i];
                // shift one bit at a time. This simplifies type dependency checks.
                for(uint32_t j = 0; j < b[i]; j++) {
                  if( (shifted & topBitMask) != 0) topBit = true;
                  else topBit = false;
                  
                  shifted <<= 1;        
                  if(topBit == true) shifted |= SCALAR_TYPE(1);
                  else               shifted &= ~(SCALAR_TYPE(1)); 
                }
                a.insert(i, shifted); 
            }
            return a;
        }

        // MROLVA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & rotateBitsLeftAssign(MASK_TYPE const & mask, VEC_TYPE & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            uint32_t bitLength = 8*sizeof(SCALAR_TYPE);
            SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
            bool topBit;
            SCALAR_TYPE shifted;
            
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true)
                {
                    shifted = a[i];
                    // shift one bit at a time. This simplifies type dependency checks.
                    for(uint32_t j = 0; j < b[i]; j++) {
                      if( (shifted & topBitMask) != 0) topBit = true;
                      else topBit = false;
                      
                      shifted <<= 1;        
                      if(topBit == true) shifted |= SCALAR_TYPE(1);
                      else               shifted &= ~(SCALAR_TYPE(1)); 
                    }
                    a.insert(i, shifted);
                } 
            }
            return a;
        }
        
        // ROLSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename SCALAR_UINT_TYPE>
        inline VEC_TYPE & rotateBitsLeftAssignScalar(VEC_TYPE & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            uint32_t bitLength = 8*sizeof(SCALAR_UINT_TYPE);
            SCALAR_TYPE topBitMask = 1 << (bitLength - 1);
            bool topBit;
            SCALAR_TYPE shifted;
                        
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                shifted = a[i];
                // shift one bit at a time. This simplifies type dependency checks.
                for(uint32_t j = 0; j < b; j++) {
                  if( (shifted & topBitMask) != 0) topBit = true;
                  else topBit = false;
                  
                  shifted <<= 1;        
                  if(topBit == true)  shifted |= SCALAR_TYPE(0x1);
                  else                shifted &= ~(SCALAR_TYPE(1)); 
                }
                a.insert(i, shifted); 
            }
            return a;
        }

        // MROLSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & rotateBitsLeftAssignScalar(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            uint32_t bitLength = 8*sizeof(SCALAR_UINT_TYPE);
            SCALAR_TYPE topBitMask = 1 << (bitLength - 1);
            bool topBit;
            SCALAR_TYPE shifted;
                        
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true)
                {
                  shifted = a[i];
                  // shift one bit at a time. This simplifies type dependency checks.
                  for(uint32_t j = 0; j < b; j++) {
                    if( (shifted & topBitMask) != 0) topBit = true;
                    else topBit = false;
                    
                    shifted <<= 1;        
                    if(topBit == true)  shifted |= SCALAR_TYPE(0x1);
                    else                shifted &= ~(SCALAR_TYPE(1)); 
                  }
                  a.insert(i, shifted);
                } 
            }
            return a;
        }
    
        // RORV
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE>
        inline VEC_TYPE rotateBitsRight(VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            uint32_t bitLength = 8*sizeof(SCALAR_TYPE);
            SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
            bool bottomBit;
            SCALAR_TYPE shifted;

            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                shifted = a[i];
                // shift one bit at a time. This simplifies type dependency checks.
                for(uint32_t j = 0; j < b[i]; j++) {
                    if( (shifted & 1) != 0) bottomBit = true;
                    else bottomBit = false;

                    shifted >>= 1;
                    if(bottomBit == true) shifted |= topBitMask;
                    else                  shifted &= ~topBitMask;
                }
                retval.insert(i, shifted);
            }
            return retval;
        }

        // MRORV
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE rotateBitsRight(MASK_TYPE const & mask, VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            uint32_t bitLength = 8*sizeof(SCALAR_TYPE);
            SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
            bool bottomBit;
            SCALAR_TYPE shifted;

            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                shifted = a[i];
                if(mask[i] == true)
                {
                    // shift one bit at a time. This simplifies type dependency checks.
                    for(uint32_t j = 0; j < b[i]; j++) {
                        if( (shifted & 1) != 0) bottomBit = true;
                        else bottomBit = false;

                        shifted >>= 1;
                        if(bottomBit == true) shifted |= topBitMask;
                        else                  shifted &= ~topBitMask;
                    }
                }
                retval.insert(i, shifted);
            }
            return retval;
        }
        
        // RORS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename SCALAR_UINT_TYPE>
        inline VEC_TYPE rotateBitsRightScalar(VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            uint32_t bitLength = 8*sizeof(SCALAR_TYPE);
            SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
            bool bottomBit;
            SCALAR_TYPE shifted;

            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                shifted = a[i];
                // shift one bit at a time. This simplifies type dependency checks.
                for(uint32_t j = 0; j < b; j++) {
                    if( (shifted & 1) != 0) bottomBit = true;
                    else bottomBit = false;

                    shifted >>= 1;
                    if(bottomBit == true) shifted |= topBitMask;
                    else                  shifted &= ~topBitMask;
                }
                retval.insert(i, shifted);
            }
            return retval;
        }
        
        // MRORS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
        inline VEC_TYPE rotateBitsRightScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            uint32_t bitLength = 8*sizeof(SCALAR_TYPE);
            SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
            bool bottomBit;
            SCALAR_TYPE shifted;

            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                shifted = a[i];
                if(mask[i] == true)
                {
                    // shift one bit at a time. This simplifies type dependency checks.
                    for(uint32_t j = 0; j < b; j++) {
                        if( (shifted & 1) != 0) bottomBit = true;
                        else bottomBit = false;

                        shifted >>= 1;
                        if(bottomBit == true) shifted |= topBitMask;
                        else                  shifted &= ~topBitMask;
                    }
                }
                retval.insert(i, shifted);
            }
            return retval;
        }

        // RORVA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE >
        inline VEC_TYPE & rotateBitsRightAssign(VEC_TYPE & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            uint32_t bitLength = 8*sizeof(SCALAR_TYPE);
            SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
            bool bottomBit;
            SCALAR_TYPE shifted;

            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                shifted = a[i];
                // shift one bit at a time. This simplifies type dependency checks.
                for(uint32_t j = 0; j < b[i]; j++) {
                    if( (shifted & 1) != 0) bottomBit = true;
                    else bottomBit = false;

                    shifted >>= 1;
                    if(bottomBit == true) shifted |= topBitMask;
                    else                  shifted &= ~topBitMask;
                }
                a.insert(i, shifted);
            }
            return a;
        }

        // MRORVA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & rotateBitsRightAssign(MASK_TYPE const & mask, VEC_TYPE & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            uint32_t bitLength = 8*sizeof(SCALAR_TYPE);
            SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
            bool bottomBit;
            SCALAR_TYPE shifted;

            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                shifted = a[i];
                if(mask[i] == true)
                {
                    // shift one bit at a time. This simplifies type dependency checks.
                    for(uint32_t j = 0; j < b[i]; j++) {
                        if( (shifted & 1) != 0) bottomBit = true;
                        else bottomBit = false;

                        shifted >>= 1;
                        if(bottomBit == true) shifted |= topBitMask;
                        else                  shifted &= ~topBitMask;
                    }
                    a.insert(i, shifted);
                }
            }
            return a;
        }

        // RORSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename SCALAR_UINT_TYPE> 
        inline VEC_TYPE & rotateBitsRightAssignScalar(VEC_TYPE &  a, SCALAR_UINT_TYPE const & b) {
            UME_EMULATION_WARNING();
            uint32_t bitLength = 8*sizeof(SCALAR_TYPE);
            SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
            bool bottomBit;
            SCALAR_TYPE shifted;

            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                shifted = a[i];
                // shift one bit at a time. This simplifies type dependency checks.
                for(uint32_t j = 0; j < b; j++) {
                    if( (shifted & 1) != 0) bottomBit = true;
                    else bottomBit = false;

                    shifted >>= 1;
                    if(bottomBit == true) shifted |= topBitMask;
                    else                  shifted &= ~topBitMask;
                }
                a.insert(i, shifted);
            }
            return a;
        }

        // MRORSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE> 
        inline VEC_TYPE & rotateBitsRightAssignScalar(MASK_TYPE const & mask, VEC_TYPE &  a, SCALAR_UINT_TYPE const & b) {
            UME_EMULATION_WARNING();
            uint32_t bitLength = 8*sizeof(SCALAR_TYPE);
            SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
            bool bottomBit;
            SCALAR_TYPE shifted;

            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                shifted = a[i];
                if(mask[i] == true)
                {
                    // shift one bit at a time. This simplifies type dependency checks.
                    for(uint32_t j = 0; j < b; j++) {
                        if( (shifted & 1) != 0) bottomBit = true;
                        else bottomBit = false;

                        shifted >>= 1;
                        if(bottomBit == true) shifted |= topBitMask;
                        else                  shifted &= ~topBitMask;
                    }
                    a.insert(i, shifted);
                }
            }
            return a;
        }

        // CMPEQV
        template<typename MASK_TYPE, typename VEC_TYPE>
        inline MASK_TYPE isEqual(VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]==b[i] );
            }
            return retval;
        }
        
        // CMPEQS
        template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
        inline MASK_TYPE isEqual(VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] == b );
            }
            return retval;
        }
        
        // CMPNEV
        template<typename MASK_TYPE, typename VEC_TYPE>
        inline MASK_TYPE isNotEqual (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]!=b[i] );
            }
            return retval;
        }

        // CMPNES
        template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
        inline MASK_TYPE isNotEqual (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]!=b );
            }
            return retval;
        }

        // CMPGTV
        template<typename MASK_TYPE, typename VEC_TYPE>
        inline MASK_TYPE isGreater (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]>b[i] );
            }
            return retval;
        }

        // CMPGTS
        template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
        inline MASK_TYPE isGreater (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]>b );
            }
            return retval;
        }

        // CMPLTV
        template<typename MASK_TYPE, typename VEC_TYPE>
        inline MASK_TYPE isLesser(VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]<b[i]);
            }
            return retval;
        }

        // CMPLTS
        template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
        inline MASK_TYPE isLesser(VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]<b );
            }
            return retval;
        }

        // CMPGEV
        template<typename MASK_TYPE, typename VEC_TYPE>
        inline MASK_TYPE isGreaterEqual(VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] >= b[i] );
            }
            return retval;
        }

        // CMPGES
        template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
        inline MASK_TYPE isGreaterEqual(VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] >= b );
            }
            return retval;
        }

        // CMPLEV
        template<typename MASK_TYPE, typename VEC_TYPE>
        inline MASK_TYPE isLesserEqual(VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] <= b[i] );
            }
            return retval;
        }
        
        // CMPLES
        template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
        inline MASK_TYPE isLesserEqual(VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] <= b );
            }
            return retval;
        }

        // ANDV
        template<typename VEC_TYPE>
        inline VEC_TYPE binaryAnd (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] & b[i] );
            }
            return retval;
        }

        // MANDV
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE binaryAnd (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] ? a[i] & b[i] : a[i]) );
            }
            return retval;
        }

        // ANDS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE binaryAnd (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] & b);
            }
            return retval;
        }
            
        // MANDS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE binaryAnd (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] ? a[i] & b : a[i]) );
            }
            return retval;
        }

        // binaryAnd (scalar, VEC) -> VEC
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE binaryAnd (SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a & b[i]);
            }
            return retval;
        }

        // binaryAnd (MASK, scalar, VEC) -> VEC
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE binaryAnd (MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] ? a & b[i] : a) );
            }
            return retval;
        }

        // ANDVA
        template<typename VEC_TYPE>
        inline VEC_TYPE & binaryAndAssign (VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i] & b[i]);
            }
            return a;
        }

        // MANDVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & binaryAndAssign (MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i] & b[i]);
            }
            return a;
        }

        // ANDSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE & binaryAndAssign (VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i] & b);
            }
            return a;
        }
            
        // MANDSA 
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & binaryAndAssign (MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i] & b);
            }
            return a;
        }         

        // ORV
        template<typename VEC_TYPE>
        inline VEC_TYPE binaryOr (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] | b[i] );
            }
            return retval;
        }

        // MORV
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE binaryOr (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] ? (a[i] | b[i]) : a[i]) );
            }
            return retval;
        }

        // ORS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE binaryOr (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] | b);
            }
            return retval;
        }

        // MORS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE binaryOr (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] ? (a[i] | b) : a[i]));
            }
            return retval;
        }
        
        // ORVA
        template<typename VEC_TYPE>
        inline VEC_TYPE & binaryOrAssign (VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i] | b[i]);
            }
            return a;
        }

        // MORVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & binaryOrAssign (MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i] | b[i]);
            }
            return a;
        }

        // ORSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE & binaryOrAssign (VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i] | b);
            }
            return a;
        }

        // MORSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & binaryOrAssign (MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i] | b);
            }
            return a;
        }

        // XORV
        template<typename VEC_TYPE>
        inline VEC_TYPE binaryXor (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] ^ b[i] );
            }
            return retval;
        }

        // MXORV
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE binaryXor (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? (a[i] ^ b[i]) : a[i] );
            }
            return retval;
        }

        // XORS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE binaryXor (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] ^ b);
            }
            return retval;
        }

        // MXORS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE binaryXor (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? (a[i] ^ b) : a[i]);
            }
            return retval;
        }

        // XORVA
        template<typename VEC_TYPE>
        inline VEC_TYPE & binaryXorAssign (VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i] ^ b[i]);
            }
            return a;
        }

        // MXORVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & binaryXorAssign (MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i] ^ b[i]);
            }
            return a;
        }

        // XORSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE & binaryXorAssign (VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i] ^ b);
            }
            return a;
        }

        // MXORSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & binaryXorAssign (MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i] ^ b);
            }
            return a;
        }

        // NOT
        template<typename VEC_TYPE>
        inline VEC_TYPE binaryNot (VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, ~a[i]);
            }
            return retval;
        }

        // MNOT
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE binaryNot (MASK_TYPE const & mask, VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? (~a[i]) : (a[i]));
            }
            return retval;
        }

        // NOTA
        template<typename VEC_TYPE>
        inline VEC_TYPE & binaryNotAssign (VEC_TYPE & a) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, ~a[i]);
            }
            return a;
        }
        
        // MNOTA
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & binaryNotAssign (MASK_TYPE const & mask, VEC_TYPE & a) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, ~a[i]);
            }
            return a;
        }

        // BLENDV
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE blend (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, mask[i] ? a[i] : b[i]);
            }
            return retval;
        }

        // BLENDS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE blend (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, mask[i] ? a[i] : b);
            }
            return retval;
        }

        // BLENDVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & blendAssign (MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, b[i]);
            }
            return a;
        }
        
        // BLENDSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & blendAssign (MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, b);
            }
            return a;
        }

        // reduceAdd(VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        inline SCALAR_TYPE reduceAdd (VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a[0];
            for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                retval += a[i];
            }
            return retval;
        }

        // reduceAdd(MASK, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        inline SCALAR_TYPE reduceAdd (MASK_TYPE const & mask, VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a[0];
            for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                if( mask[i] == true ) retval += a[i];
            }
            return retval;
        }

        // reduceAdd (scalar, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        inline SCALAR_TYPE reduceAdd (SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i <VEC_TYPE::length(); i++) {
                retval += b[i];
            }
            return retval;
        }

        // reduceAdd(MASK, scalar, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        inline SCALAR_TYPE reduceAdd (MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i <VEC_TYPE::length(); i++) {
                if( mask[i] == true ) retval += b[i];
            }
            return retval;
        }

        // reduceMult(VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        inline SCALAR_TYPE reduceMult (VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a[0];
            for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                retval *= a[i];
            }
            return retval;
        }

        // reduceMult(MASK, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        inline SCALAR_TYPE reduceMult (MASK_TYPE const & mask, VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = (mask[0] == true) ? a[0] : 0; // TODO: replace 0 with const expr returning zero depending on SCALAR type.
            for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                if( mask[i] == true ) retval *= a[i];
            }
            return retval;
        }

        // reduceMult(scalar, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        inline SCALAR_TYPE reduceMultScalar (SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval *= b[i];
            }
            return retval;
        }

        // reduceMult(MASK, scalar, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        inline SCALAR_TYPE reduceMultScalar (MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if( mask[i] == true ) retval *= b[i];
            }
            return retval;
        }

        // reduceBinaryOr (VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        inline SCALAR_TYPE reduceBinaryOr (VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a[0];
            for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                retval |= a[i];
            }
            return retval;
        }

        // reduceBinaryOr (MASK, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        inline SCALAR_TYPE reduceBinaryOr (MASK_TYPE const & mask, VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = (mask[0] == true) ? a[0] : 0; // TODO: 0-initializer of SCALAR_TYPE
            for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                if( mask[i] == true ) retval |= a[i];
            }
            return retval;
        }

        // reduceBinaryOr (scalar, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        inline SCALAR_TYPE reduceBinaryOrScalar (SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval |= b[i];
            }
            return retval;
        }     

        // reduceBinaryOr (MASK, scalar, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        inline SCALAR_TYPE reduceBinaryOrScalar (MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if( mask[i] == true ) retval |= b[i];
            }
            return retval;
        }

        // reduceBinaryAnd (VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        inline SCALAR_TYPE reduceBinaryAnd(VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a[0];
            for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                retval &= a[i];
            }
            return retval;
        }

        // reduceBinaryAnd (MASK, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        inline SCALAR_TYPE reduceBinaryAnd(MASK_TYPE const & mask, VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = (mask[0] == true) ? a[0] : ~0; // TODO: 0-initializer of scalar type
            for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                if( mask[i] == true ) retval &= a[i];
            }
            return retval;
        }

        // reduceBinaryAnd (scalar, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        inline SCALAR_TYPE reduceBinaryAndScalar(SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval &= b[i];
            }
            return retval;
        }

        // reduceBinaryAnd (MASK, scalar, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        inline SCALAR_TYPE reduceBinaryAndScalar(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) retval &= b[i];
            }
            return retval;
        }

        // reduceBinaryXor() -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        inline SCALAR_TYPE reduceBinraryXor(VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = 0;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) { 
                retval ^= a[i];
            }
            return retval;
        }

        // reduceBinaryXor(MASK) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        inline SCALAR_TYPE reduceBinraryXor(MASK_TYPE const & mask, VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = 0;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) { 
                if(mask[i] == true) retval ^= a[i];
            }
            return retval;
        }

        // reduceBinaryXor(scalar) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        inline SCALAR_TYPE reduceBinaryXorScalar(SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) { 
                retval ^= b[i];
            }
            return retval;
        }

        // reduceBinaryXor(MASK, scalar) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        inline SCALAR_TYPE reduceBinaryXorScalar(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) { 
                if(mask[i] == true) retval ^= b[i];
            }
            return retval;
        }
            
        // ******************************************************************
        // * MATH FUNCTIONS                                                 
        // *****************************************************************
        namespace MATH
        {
            // MAXV
            template<typename VEC_TYPE>
            inline VEC_TYPE max(VEC_TYPE const & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (a[i] > b[i] ? a[i] : b[i]));
                }
                return retval;
            }

            // MMAXV
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE max(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, (a[i] > b[i] ? a[i] : b[i]));
                }
                return retval;
            }

            // MAXS
            template<typename VEC_TYPE, typename SCALAR_TYPE>
            inline VEC_TYPE maxScalar(VEC_TYPE const & a, SCALAR_TYPE b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (a[i] > b ? a[i] : b));
                }
                return retval;
            }
        
            // MMAXS
            template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
            inline VEC_TYPE maxScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, (a[i] > b ? a[i] : b));
                }
                return retval;
            }
            
            // MAXVA
            // MMAXVA
            // MAXSA
            // MMAXSA

            // reduceMax (VEC) -> scalar
            template<typename SCALAR_TYPE, typename VEC_TYPE>
            inline SCALAR_TYPE reduceMax(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                SCALAR_TYPE retval = a[0];
                for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                    if(a[i] > retval) retval = a[i];
                }
                return retval;
            }

            // reduceMax (MASK, VEC) -> scalar
            template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
            inline SCALAR_TYPE reduceMax(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                SCALAR_TYPE retval = std::numeric_limits<SCALAR_TYPE>::min();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if( (mask[i] == true) && a[i] > retval) retval = a[i];
                }
                return retval;
            }

            // reduceMax (scalar, VEC) -> scalar
            template<typename SCALAR_TYPE, typename VEC_TYPE>
            inline SCALAR_TYPE reduceMax(SCALAR_TYPE a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                SCALAR_TYPE retval = a;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(b[i] > retval) retval = b[i];
                }
                return retval;
            }

            // reduceMax (MASK, scalar, VEC) -> scalar
            template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
            inline SCALAR_TYPE reduceMax(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                SCALAR_TYPE retval = a;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if((mask[i] == true) && (a[i] > retval)) retval = a[i];
                }
                return retval;
            }
            
            // MINV
            template<typename VEC_TYPE>
            inline VEC_TYPE min(VEC_TYPE const & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, a[i] < b[i] ? a[i] : b[i]);
                }
                return retval;
            }

            // MMINV
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE min(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval(std::numeric_limits<decltype(a[0])>::max());
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, a[i] < b[i] ? a[i] : b[i]);
                }
                return retval;
            }

            // MINS
            template<typename VEC_TYPE, typename SCALAR_TYPE>
            inline VEC_TYPE minScalar(VEC_TYPE const & a, SCALAR_TYPE b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, a[i] < b ? a[i] : b);
                }
                return retval;
            }

            // MMINS
            template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
            inline VEC_TYPE minScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval(std::numeric_limits<SCALAR_TYPE>::max());
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, a[i] < b ? a[i] : b);
                }
                return retval;
            }
            
            // MINVA
            // MMINVA
            // MINSA
            // MMINSA

            // ABS
            template<typename VEC_TYPE>
            inline VEC_TYPE abs(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    // abs for floating point numbers is non-trivial. Using std::abs for reliability.
                    retval.insert(i, std::abs(a[i])); 
                }
                return retval;
            }

            // MABS
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE abs(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    // abs for floating point numbers is non-trivial. Using std::abs for reliability.
                    retval.insert(i, (mask[i] == true ? std::abs(a[i]) : a[i] ));
                }
                return retval;
            }

            // ABSA
            template<typename VEC_TYPE>
            inline VEC_TYPE & absAssign(VEC_TYPE & a) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    // abs for floating point numbers is non-trivial. Using std::abs for reliability.
                    a.insert(i, std::abs(a[i])); 
                }
                return a;
            }

            // MABSA
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE absAssign(MASK_TYPE const & mask, VEC_TYPE & a) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    // abs for floating point numbers is non-trivial. Using std::abs for reliability.
                    a.insert(i, (mask[i] == true ? std::abs(a[i]) : a[i] ));
                }
                return a;
            }

            // SQR
            template<typename VEC_TYPE>
            inline VEC_TYPE sqr(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, a[i] * a[i]);
                }
                return retval;
            }
            
            // MSQR
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE sqr(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, a[i] * a[i]);
                }
                return retval;
            }
            
            // SQRA
            template<typename VEC_TYPE>
            inline VEC_TYPE & sqr(VEC_TYPE & a) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    a.insert(i, a[i] * a[i]);
                }
                return a;
            }
            
            // MSQRA
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE & sqr(MASK_TYPE const & mask, VEC_TYPE & a) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) a.insert(i, a[i] * a[i]);
                }
                return a;
            }

            // SQRT
            template<typename VEC_TYPE>
            inline VEC_TYPE sqrt(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::sqrt(a[i])); 
                }
                return retval;
            }
            
            // MSQRT
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE sqrt(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (mask[i] == true) ? std::sqrt(a[i]) : a[i]);
                }
                return retval;
            }

            // SQRTA
            template<typename VEC_TYPE>
            inline VEC_TYPE & sqrtAssign (VEC_TYPE & a) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    a.insert(i, std::sqrt(a[i]));
                }
                return a;
            }

            // MSQRTA
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE & sqrtAssign(MASK_TYPE const & mask, VEC_TYPE & a) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) a.insert(i, std::sqrt(a[i]));
                }
                return a;
            }

            // POWV
            // MPOWV
            // POWS
            // MPOWS
            
            // ROUND
            template<typename VEC_TYPE>
            inline VEC_TYPE round(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::round(a[i]));
                }
                return retval;
            }
            
            // MROUND
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE round(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, std::round(a[i]));
                    else retval.insert(i, a[i]);
                }
                return retval;
            }
            
            // TRUNC
            template<typename VEC_TYPE, typename INT_VEC_TYPE>
            inline INT_VEC_TYPE truncToInt(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                INT_VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, decltype(retval[0])(std::trunc(a[i])));
                }
                return retval;
            }
            
            // MTRUNC
            template<typename VEC_TYPE, typename INT_VEC_TYPE, typename MASK_TYPE>
            inline INT_VEC_TYPE truncToInt(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                INT_VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, decltype(retval[0])(std::trunc(a[i])));
                    else retval.insert(i, 0);
                }
                return retval;
            }
            
            // FLOOR
            // MFLOOR
            
            // CEIL
            // MCEIL
            
            // FMULADDV
            // MFMULADDV
            // FADDMULV
            // MFADDMULV
            // FMULSUBV
            // MFMULSUBV
            // FSUBMULV
            // MFSUBMULV
            
            // ISFIN
            // ISINF
            // ISAN
            // ISNAN
            // ISSUB
            // ISZERO
            // ISZEROSUB
            
            // SIN
            template<typename VEC_TYPE>
            inline VEC_TYPE sin (VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::sin(a[i]));
                }
                return retval;
            }

            // MSIN
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE sin (MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (mask[i] == true) ? std::sin(a[i]) : a[i]);
                }
                return retval;
            }

            // COS
            template<typename VEC_TYPE>
            inline VEC_TYPE cos (VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::cos(a[i]));
                }
                return retval;
            }

            // MCOS
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE cos (MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (mask[i] == true) ? std::cos(a[i]) : a[i]);
                }
                return retval;
            }

            // TAN
            // MTAN
            // CTAN
            // MCTAN
        } // UME::SIMD::EMULATED_FUNCTIONS::MATH

    } // namespace UME::SIMD::EMULATED_FUNCTIONS
    
    // This class is a wrapper of scalar types that forbids implicit type conversions.
    template<typename SCALAR_TYPE> 
    class ScalarTypeWrapper
    {
    private:
        SCALAR_TYPE mValue;

    public:
        ScalarTypeWrapper()
        {
            mValue = SCALAR_TYPE(0);
        }

        // Forbid implicit construction with boolean values
        explicit ScalarTypeWrapper(bool x) : mValue(x)  {};

        // Forbid implicit construciton with character type values
        explicit ScalarTypeWrapper(signed char x) : mValue(x) {};
        explicit ScalarTypeWrapper(unsigned char x) : mValue(x) {};
        explicit ScalarTypeWrapper(char x) : mValue(x) {};
        explicit ScalarTypeWrapper(signed short int x) : mValue(x) {};
        explicit ScalarTypeWrapper(unsigned short int x) : mValue(x) {};
        explicit ScalarTypeWrapper(signed int x) : mValue(x) {};
        explicit ScalarTypeWrapper(unsigned int x) : mValue(x) {};
        explicit ScalarTypeWrapper(signed long int x) : mValue(x) {};
        explicit ScalarTypeWrapper(unsigned long int x) : mValue(x) {};
        explicit ScalarTypeWrapper(signed long long int x) : mValue(x) {};
        explicit ScalarTypeWrapper(unsigned long long int x) : mValue(x) {};

        explicit ScalarTypeWrapper(float x) : mValue(x) {};
        explicit ScalarTypeWrapper(double x) : mValue(x) {};
        explicit ScalarTypeWrapper(long double x) : mValue(x) {};

        // define cast operator
        operator SCALAR_TYPE() const { return mValue; };
               
        inline ScalarTypeWrapper & operator=(ScalarTypeWrapper const & x){
            mValue = x.mValue;
            return *this;
        }

        // Also define a non-modifying access operator
        inline SCALAR_TYPE operator[] (uint32_t index) const { return mValue; }
    };

    // This class represents a vector of VEC_LEN scalars and is used for emulation.
    template<typename SCALAR_TYPE, uint32_t VEC_LEN> 
    class SIMDVecEmuRegister
    {
    private:
        SCALAR_TYPE reg[VEC_LEN];
    public:
        SIMDVecEmuRegister() {
            UME_EMULATION_WARNING();
            for(int i = 0; i < VEC_LEN; i++) { reg[i] = 0; }
        }

        SIMDVecEmuRegister(SCALAR_TYPE x) {
            UME_EMULATION_WARNING();
            for(int i = 0; i < VEC_LEN; i++) { reg[i] = x; }
        }

        SIMDVecEmuRegister(SIMDVecEmuRegister const & x) {
            UME_EMULATION_WARNING();
            for(int i = 0; i < VEC_LEN; i++) { reg[i] = x.reg[i]; }
        }

        // Also define a non-modifying access operator
        inline SCALAR_TYPE operator[] (uint32_t index) const { 
            SCALAR_TYPE temp = reg[index];    
            return temp; 
        }
            
        inline void insert(uint32_t index, SCALAR_TYPE value){ reg[index] = value; }
    };

    template<uint32_t MASK_LEN>
    struct MaskAsInt{
        uint64_t m0;
    };
    
    template<>
    struct MaskAsInt<128> {
        uint64_t m0;
        uint64_t m1;
    };
    
    // **********************************************************************
    // *
    // *  Declaration of SIMDMaskBaseInterface class 
    // *
    // *    This class should be used as a basic class for all masks. 
    // *    All masks should implement interface contained in 
    // *    SIMDMaskBaseInterface. If the derived class does not provide an 
    // *    overload for given operation, this class will default 
    // *    to scalar emulation, thus providing interface coherence over
    // *    different plugins.
    // *
    // **********************************************************************

    template<class DERIVED_MASK_TYPE, 
            typename MASK_BASE_TYPE, 
            uint32_t MASK_LEN>
    class SIMDMaskBaseInterface {
        // Declarations only. These operators should be 
        // EXTRACT
        inline bool extract(uint32_t index);
        // EXTRACT
        inline bool operator[] (uint32_t index);
        // INSERT
        inline void insert(uint32_t index, bool value);

    protected:
        ~SIMDMaskBaseInterface() {};

    public:
        // LENGTH
        static uint32_t length () { return MASK_LEN; };

        // ALIGNMENT
        static int alignment () { return MASK_LEN*sizeof(MASK_BASE_TYPE); };
        
        // LOAD
        inline DERIVED_MASK_TYPE & load (bool* addr) {
            return EMULATED_FUNCTIONS::load<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE &>(*this), addr);
        };

        // LOADA
        inline DERIVED_MASK_TYPE & loadAligned (bool* addrAligned) {
            return EMULATED_FUNCTIONS::loadAligned<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE &>(*this), addrAligned);
        };

        // STORE
        inline MASK_BASE_TYPE* store (bool* addr) {
            return EMULATED_FUNCTIONS::store<DERIVED_MASK_TYPE, bool> (static_cast<DERIVED_MASK_TYPE const &>(*this), addr);
        };

        // STOREA
        inline MASK_BASE_TYPE* storea (bool* addrAligned) {
            return EMULATED_FUNCTIONS::storeAligned<DERIVED_MASK_TYPE, bool> (static_cast<DERIVED_MASK_TYPE const &>(*this), addrAligned);
        };

        // ASSIGN
        inline DERIVED_MASK_TYPE & assign (DERIVED_MASK_TYPE const & maskOp) {
            return EMULATED_FUNCTIONS::assign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        };

        // AND 
        DERIVED_MASK_TYPE andm ( DERIVED_MASK_TYPE const & maskOp) {
            return EMULATED_FUNCTIONS::binaryAnd<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), maskOp);
        };

        // ANDA
        inline DERIVED_MASK_TYPE & anda (DERIVED_MASK_TYPE const & maskOp) {
            return EMULATED_FUNCTIONS::binaryAndAssign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        };

        // OR
        inline DERIVED_MASK_TYPE orm (DERIVED_MASK_TYPE const & maskOp) {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), maskOp);
        }

        // ORA
        inline DERIVED_MASK_TYPE & ora (DERIVED_MASK_TYPE const & maskOp) {
            return EMULATED_FUNCTIONS::binaryOrAssign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        }

        // XOR
        inline DERIVED_MASK_TYPE xorm (DERIVED_MASK_TYPE const & maskOp) {
            return EMULATED_FUNCTIONS::binaryXor<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), maskOp);
        }
        
        // XORA
        inline DERIVED_MASK_TYPE & xora (DERIVED_MASK_TYPE const & maskOp) {
            return EMULATED_FUNCTIONS::binaryXorAssign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        }

        // NOT
        inline DERIVED_MASK_TYPE notm (){
            return EMULATED_FUNCTIONS::binaryNot<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this));
        }

        // NOTA
        inline DERIVED_MASK_TYPE nota () {
            return EMULATED_FUNCTIONS::binaryNotAssign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this));
        }

        // TOINT
        inline MaskAsInt<MASK_LEN> toInt() {
            return 0;
        }

        // FROMINT
        inline DERIVED_MASK_TYPE & fromInt(MaskAsInt<MASK_LEN>) {
            return *this;
        }
    };

    // **********************************************************************
    // *
    // *  Declaration of SIMDVecBaseInterface class 
    // *
    // *    This class should be used as a basic class for all integer and 
    // *    floating point vector types. All vectors should implement interface
    // *    contained in SIMDVecBaseInterface. If the derived class does not
    // *    provide an overload for given operation, this class will default 
    // *    to scalar emulation, thus providing interface coherence over
    // *    different plugins. This class should not be used directly in
    // *    plugins since it encapsulates only a common part of all vector
    // *    types. Plugins should use:
    // *     - "SIMDVecUnsignedInterface" for unsigned integer vectors,
    // *     - "SIMDVecSignedInterface" for signed integer vectors,
    // *     - "SIMDVecFloatInterface" for floating point vectors
    // *
    // **********************************************************************
    
    // DERIVED_VEC_TYPE - this is a derived class to be used as a part of 'Curiously Recurring Design Pattern (CRTP)'
    // SCALAR_TYPE - basic type of scalar elements packed in DERIVED_VEC_TYPE
    // VEC_LEN - number of SIMD elements in vector
    // MASK_TYPE - exact type of the mask to be used with this vector
    template<class DERIVED_VEC_TYPE, 
             typename SCALAR_TYPE, 
             uint32_t VEC_LEN,
             typename MASK_TYPE>
    class SIMDVecBaseInterface
    {
        // Other vector types necessary for this class
        typedef SIMDVecBaseInterface< 
            DERIVED_VEC_TYPE, 
            SCALAR_TYPE, 
            VEC_LEN, 
            MASK_TYPE> VEC_TYPE;

    private:
        // Forbid assignment-initialization of vector using scalar values
        // TODO: is this necessary?
        inline VEC_TYPE & operator= (const int8_t & x) { }
        inline VEC_TYPE & operator= (const int16_t & x) { }
        inline VEC_TYPE & operator= (const int32_t & x) { }
        inline VEC_TYPE & operator= (const int64_t & x) { }
        inline VEC_TYPE & operator= (const uint8_t & x) { }
        inline VEC_TYPE & operator= (const uint16_t & x) { }
        inline VEC_TYPE & operator= (const uint32_t & x) { }
        inline VEC_TYPE & operator= (const uint64_t & x) { }
        inline VEC_TYPE & operator= (const float & x) { }
        inline VEC_TYPE & operator= (const double & x) { }
 
    protected:
            
        // Making destructor protected prohibits this class from being instantiated. Effectively this class can only be used as a base class.
        ~SIMDVecBaseInterface() {};
    public:
    
        // TODO: can be marked as constexpr?
        static uint32_t length() { return VEC_LEN; };

        static int alignment() { return VEC_LEN*sizeof(SCALAR_TYPE); };
        
        inline SCALAR_TYPE extract(uint32_t index)
        {
            // Extract method should be provided for all derived classes.
            return static_cast<DERIVED_VEC_TYPE>(*this)->extract(index);
        }

        inline SCALAR_TYPE operator[] (uint32_t index) {
            // Extract method should be provided for all derived classes.
            return static_cast<DERIVED_VEC_TYPE>(*this)->extract(index);
        }

        // ASSIGNV
        inline DERIVED_VEC_TYPE & assign (DERIVED_VEC_TYPE const & src) {
            return EMULATED_FUNCTIONS::assign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), src);
        }
            
        // MASSIGNV
        inline DERIVED_VEC_TYPE & assign (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & src) {
            return EMULATED_FUNCTIONS::assign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), src);
        }

        // ASSIGNS
        inline DERIVED_VEC_TYPE & assign (SCALAR_TYPE value) {
            return EMULATED_FUNCTIONS::assign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), value);
        }

        // MASSIGNS
        inline DERIVED_VEC_TYPE & assign (MASK_TYPE const & mask, SCALAR_TYPE value) {
            return EMULATED_FUNCTIONS::assign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), value);
        }

        // LOAD
        inline DERIVED_VEC_TYPE & load (SCALAR_TYPE const *p) {
            return EMULATED_FUNCTIONS::load<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }

        // MLOAD
        inline DERIVED_VEC_TYPE & load (MASK_TYPE const & mask, SCALAR_TYPE const * p) {
            return EMULATED_FUNCTIONS::load<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }

        // LOADA
        inline DERIVED_VEC_TYPE & loada (SCALAR_TYPE const * p) {
            return EMULATED_FUNCTIONS::loadAligned<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }

        // MLOADA
        inline DERIVED_VEC_TYPE & loada (MASK_TYPE const & mask, SCALAR_TYPE const *p) {
            return EMULATED_FUNCTIONS::loadAligned<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }

        // STORE
        inline SCALAR_TYPE* store (SCALAR_TYPE* p) {
            return EMULATED_FUNCTIONS::store<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }

        // MSTORE
        inline SCALAR_TYPE* store (MASK_TYPE const & mask, SCALAR_TYPE* p) {
            return EMULATED_FUNCTIONS::store<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }

        // STOREA
        inline SCALAR_TYPE* storea (SCALAR_TYPE* p) {
            return EMULATED_FUNCTIONS::store<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }

        // MSTOREA
        inline SCALAR_TYPE* storea (MASK_TYPE const & mask, SCALAR_TYPE* p) {
           return EMULATED_FUNCTIONS::store<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }
        
        // TODO:
        // pack(VEC, VEC_HALF_LEN, VEC_HALF_LEN)
        // pack(VEC, VEC_QUARTER_LEN, VEC_QUARTER_LEN, VEC_QUARTER_LEN, VEC_QUARTER_LEN)
        // ...
        // packLow(VEC, VEC_HALF_LEN)
        // packHigh(VEC, VEC_HALF_LEN)

        // unpack(VEC, VEC_HALF_LEN, VEC_HALF_LEN)
        // unpack(VEC, VEC_QUARTER_LEN, VEC_QUARTER_LEN)
        // ...
        // unpackLow(VEC, VEC_HALF_LEN)
        // unpackHigh(VEC, VEC_HALF_LEN)

        // swizzle (VEC, swizzleMask)
        // swizzle(MASK, VEC, swizzleMask) 

        // ADDV
        inline DERIVED_VEC_TYPE add ( DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::add<DERIVED_VEC_TYPE> ( static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MADDV
        inline DERIVED_VEC_TYPE add( MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::add<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ADDS
        inline DERIVED_VEC_TYPE add(SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::addScalar<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MADDS
        inline DERIVED_VEC_TYPE add(MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::addScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ADDVA
        inline DERIVED_VEC_TYPE & adda (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::addAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MADDVA
        inline DERIVED_VEC_TYPE & adda (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::addAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // ADDSA
        inline DERIVED_VEC_TYPE & adda (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::addAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        // MADDSA
        inline DERIVED_VEC_TYPE & adda (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::addAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        // SADDV
        inline DERIVED_VEC_TYPE sadd(DERIVED_VEC_TYPE const & b) {
            return DERIVED_VEC_TYPE(); // TODO:
        } 

        // MSADDV
        // SADDS
        // MSADDS
        // SADDVA
        // MSADDVA
        // SADDSA
        // MSADDSA
        
        // POSTINC
        inline DERIVED_VEC_TYPE postInc () {
            return EMULATED_FUNCTIONS::postfixIncrement<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MPOSTINC
        inline DERIVED_VEC_TYPE postInc (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::postfixIncrement<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // PREFINC
        inline DERIVED_VEC_TYPE & prefInc () {
            return EMULATED_FUNCTIONS::prefixIncrement<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MPREFINC
        inline DERIVED_VEC_TYPE & prefInc (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::prefixIncrement<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // SUBV
        inline DERIVED_VEC_TYPE sub (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::sub<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MSUBV
        inline DERIVED_VEC_TYPE sub (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::sub<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SUBS
        inline DERIVED_VEC_TYPE sub (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::subScalar<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        // MSUBS
        inline DERIVED_VEC_TYPE sub (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::subScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SUBVA
        inline DERIVED_VEC_TYPE & suba (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::subAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MSUBVA
        inline DERIVED_VEC_TYPE & suba (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::subAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // SUBSA
        inline DERIVED_VEC_TYPE & suba (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::subAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MSUBSA
        inline DERIVED_VEC_TYPE & suba (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::subAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
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
        inline DERIVED_VEC_TYPE postDec () {
            return EMULATED_FUNCTIONS::postfixDecrement<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MPOSTDEC
        inline DERIVED_VEC_TYPE postDec (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::postfixDecrement<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // PREFDEC
        inline DERIVED_VEC_TYPE & prefDec() {
            return EMULATED_FUNCTIONS::prefixDecrement<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }
        
        // MPREFDEC
        inline DERIVED_VEC_TYPE & prefDec (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::prefixDecrement<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MULV
        inline DERIVED_VEC_TYPE mul (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::mult<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMULV
        inline DERIVED_VEC_TYPE mul (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::mult<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MULS
        inline DERIVED_VEC_TYPE mul (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::mult<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMULS
        inline DERIVED_VEC_TYPE mul (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::mult<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MULVA
        inline DERIVED_VEC_TYPE & mula (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::multAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MMULVA
        inline DERIVED_VEC_TYPE & mula (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::multAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MULSA
        inline DERIVED_VEC_TYPE mula (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::multAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MMULSA
        inline DERIVED_VEC_TYPE mula (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::multAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // DIVV
        inline DERIVED_VEC_TYPE div (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::div<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MDIVV
        inline DERIVED_VEC_TYPE div (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::div<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // DIVS
        inline DERIVED_VEC_TYPE div (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::div<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MDIVS
        inline DERIVED_VEC_TYPE div (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::div<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
          
        // DIVVA
        inline DERIVED_VEC_TYPE diva (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::divAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MDIVVA
        inline DERIVED_VEC_TYPE diva (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::divAssign<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // DIVSA
        inline DERIVED_VEC_TYPE diva (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::divAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MDIVSA
        inline DERIVED_VEC_TYPE & diva (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::divAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
       
        // RCP
        inline DERIVED_VEC_TYPE rcp () {
            return EMULATED_FUNCTIONS::div<DERIVED_VEC_TYPE, SCALAR_TYPE> (1, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MRCP

        // RCPS
        inline DERIVED_VEC_TYPE rcp (SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::div<DERIVED_VEC_TYPE, SCALAR_TYPE> (a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MRCPS
        // RCPA
        
        // RCPSA
        inline DERIVED_VEC_TYPE & rcpa (SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::divAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MRCPSA
        
        // CMPEQV
        inline MASK_TYPE cmpeq (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::isEqual<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // CMPEQS
        inline MASK_TYPE cmpeq (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::isEqual<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // CMPNEV
        inline MASK_TYPE cmpne (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::isNotEqual<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // CMPNES
        inline MASK_TYPE cmpne (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::isNotEqual<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }    

        // CMPGTV
        inline MASK_TYPE cmpgt (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::isGreater<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // CMPGTS
        inline MASK_TYPE cmpgt (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::isGreater<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // CMPLTV
        inline MASK_TYPE cmplt (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::isLesser<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // CMPLTS
        inline MASK_TYPE cmplt (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::isLesser<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // CMPGEV
        inline MASK_TYPE cmpge (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::isGreaterEqual<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // CMPGES
        inline MASK_TYPE cmpge (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::isGreaterEqual<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        // CMPLEV
        inline MASK_TYPE cmple (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::isLesserEqual<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // CMPLES
        inline MASK_TYPE cmple (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::isLesserEqual<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ANDV
        inline DERIVED_VEC_TYPE andv (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryAnd<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MANDV
        inline DERIVED_VEC_TYPE andv (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryAnd<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ANDS
        inline DERIVED_VEC_TYPE ands (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryAnd<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MANDS
        inline DERIVED_VEC_TYPE ands (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryAnd<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ANDVA
        inline DERIVED_VEC_TYPE & anda (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryAndAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MANDVA
        inline DERIVED_VEC_TYPE & anda (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryAndAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        // ANDSA
        inline DERIVED_VEC_TYPE & anda (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryAndAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MANDSA
        inline DERIVED_VEC_TYPE & anda (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryAndAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // ORV
        inline DERIVED_VEC_TYPE orv ( DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MORV
        inline DERIVED_VEC_TYPE orv ( MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ORS
        inline DERIVED_VEC_TYPE ors (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MORS
        inline DERIVED_VEC_TYPE ors (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ORVA
        inline DERIVED_VEC_TYPE & ora (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryOrAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MORVA
        inline DERIVED_VEC_TYPE & ora (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryOrAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // ORSA
        inline DERIVED_VEC_TYPE & ora (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryOrAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
            
        // MORSA
        inline DERIVED_VEC_TYPE & ora (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryOrAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // XORV
        inline DERIVED_VEC_TYPE xorv (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryXor<DERIVED_VEC_TYPE> ( static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MXORV
        inline DERIVED_VEC_TYPE xorv (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryXor<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // XORS
        inline DERIVED_VEC_TYPE xors (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryXor<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MXORS
        inline DERIVED_VEC_TYPE xorv (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryXor<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // XORVA
        inline DERIVED_VEC_TYPE & xora (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryXorAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MXORVA
        inline DERIVED_VEC_TYPE & xora (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryXorAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // XORSA
        inline DERIVED_VEC_TYPE & xora (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryXorAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MXORSA
        inline DERIVED_VEC_TYPE & xora (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryXorAssign<DERIVED_VEC_TYPE,SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // NOT
        inline DERIVED_VEC_TYPE notv () {
            return EMULATED_FUNCTIONS::binaryNot<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
    
        // MNOT
        inline DERIVED_VEC_TYPE notv (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::binaryNot<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // NOTA
        inline DERIVED_VEC_TYPE & nota () {
            return EMULATED_FUNCTIONS::binaryNotAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MNOTA
        inline DERIVED_VEC_TYPE & nota (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::binaryNotAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // BLENDV
        inline DERIVED_VEC_TYPE blend (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::blend<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BLENDS
        inline DERIVED_VEC_TYPE blend (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::blend<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BLENDVA
        inline DERIVED_VEC_TYPE & blenda (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::blendAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // BLENDSA
        inline DERIVED_VEC_TYPE & blenda (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::blendAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // HADD
        inline SCALAR_TYPE hadd () {
            return EMULATED_FUNCTIONS::reduceAdd<SCALAR_TYPE, DERIVED_VEC_TYPE>( static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
    
        // MHADD
        inline SCALAR_TYPE hadd (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::reduceAdd<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &> (*this));
        }

        // TODO: is this even necessary in the interface?
        inline SCALAR_TYPE horizontal_add_x(VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            // TODO:
            return -1;
        }

        // HMUL
        inline SCALAR_TYPE hmul () {
            return EMULATED_FUNCTIONS::reduceMult<SCALAR_TYPE, DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHMUL
        inline SCALAR_TYPE hmul (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::reduceMult<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HMULS
        inline SCALAR_TYPE hmul (SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::reduceMultScalar<SCALAR_TYPE, DERIVED_VEC_TYPE>(a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHMULS
        inline SCALAR_TYPE hmul (MASK_TYPE const & mask, SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::reduceMultScalar<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // HAND
        inline SCALAR_TYPE hand () {
            return EMULATED_FUNCTIONS::reduceBinaryAnd<SCALAR_TYPE, DERIVED_VEC_TYPE>( static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHAND
        inline SCALAR_TYPE hand (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::reduceBinaryAnd<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HANDS
        inline SCALAR_TYPE hand (SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::reduceBinaryAndScalar<SCALAR_TYPE, DERIVED_VEC_TYPE>(a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHANDS
        inline SCALAR_TYPE hand (MASK_TYPE const & mask, SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::reduceBinaryAndScalar<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HOR
        inline SCALAR_TYPE hor () {
            return EMULATED_FUNCTIONS::reduceBinaryOr<SCALAR_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHOR
        inline SCALAR_TYPE hor (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::reduceBinaryOr<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HORS
        inline SCALAR_TYPE hor (SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::reduceBinaryOrScalar<SCALAR_TYPE, DERIVED_VEC_TYPE> (a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHORS
        inline SCALAR_TYPE hor (MASK_TYPE const & mask, SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::reduceBinaryOrScalar<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // HXOR
        // MHXOR
        // HXORS
        // MHXORS

        // ******************************************************************
        // * Fused arithmetics
        // ******************************************************************

        // FMULADDV
        // FMMULADDV
        // FMULSUBV
        // FMMULSUBV
        // FADDMULV
        // FMADDMULV
        // FSUBMULV
        // FMSUBMULV

        // ******************************************************************
        // * Additional math functions
        // ******************************************************************

        // MAXV
        inline DERIVED_VEC_TYPE max (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::MATH::max<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMAXV
        inline DERIVED_VEC_TYPE max (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::MATH::max<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MAXS
        inline DERIVED_VEC_TYPE max (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::MATH::maxScalar<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMAXS
        inline DERIVED_VEC_TYPE max (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::MATH::maxScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        // MINV
        inline DERIVED_VEC_TYPE min (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::MATH::min<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMINV
        inline DERIVED_VEC_TYPE min (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::MATH::min<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        // MINS
        inline DERIVED_VEC_TYPE min (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::MATH::minScalar<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMINS
        inline DERIVED_VEC_TYPE min (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::MATH::minScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        // MINAV
        // MMINAV
        // MINAS
        // MMINAS
        
        // HMAX
        inline DERIVED_VEC_TYPE hmax () {
            return EMULATED_FUNCTIONS::MATH::reduceMax<SCALAR_TYPE, DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // IMAX

        // MHMAX
        inline DERIVED_VEC_TYPE hmax (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::MATH::reduceMax<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MIMAX

        // HMAXS
        inline DERIVED_VEC_TYPE hmax (SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::MATH::reduceMax<SCALAR_TYPE, DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // MHMAXS
        inline DERIVED_VEC_TYPE hmax (MASK_TYPE const & mask, SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::MATH::reduceMax<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HMIN
        // IMIN
        // MHMIN
        // MIMIN
    };

    // ***************************************************************************
    // *
    // *    Definition of interface for vectors using UNSIGNED INTEGER scalar types
    // *
    // ***************************************************************************
    template<typename DERIVED_VEC_TYPE,
             typename DERIVED_VEC_UINT_TYPE,
             typename SCALAR_TYPE,
             typename SCALAR_UINT_TYPE, 
             uint32_t VEC_LEN,
             typename MASK_TYPE> 
    class SIMDVecUnsignedInterface : public SIMDVecBaseInterface< 
        DERIVED_VEC_TYPE,
        SCALAR_TYPE, 
        VEC_LEN,
        MASK_TYPE>
    {
        // Other vector types necessary for this class
        typedef SIMDVecUnsignedInterface< DERIVED_VEC_TYPE, 
            DERIVED_VEC_UINT_TYPE,
            SCALAR_TYPE,
            SCALAR_UINT_TYPE, 
            VEC_LEN, 
            MASK_TYPE> VEC_TYPE;

    private:

        // Forbid assignment-initialization of vector using scalar values
        // TODO: is this necessary?
        /*inline VEC_TYPE & operator= (const int8_t & x) { }
        inline VEC_TYPE & operator= (const int16_t & x) { }
        inline VEC_TYPE & operator= (const int32_t & x) { }
        inline VEC_TYPE & operator= (const int64_t & x) { }
        inline VEC_TYPE & operator= (const uint8_t & x) { }
        inline VEC_TYPE & operator= (const uint16_t & x) { }
        inline VEC_TYPE & operator= (const uint32_t & x) { }
        inline VEC_TYPE & operator= (const uint64_t & x) { }
        inline VEC_TYPE & operator= (const float & x) { }
        inline VEC_TYPE & operator= (const double & x) { }
 */
        SCALAR_UINT_TYPE operator[] (SCALAR_UINT_TYPE index) const; // Declaration only! This operator has to be implemented in derived class.
        inline DERIVED_VEC_TYPE & insert(uint32_t index, SCALAR_UINT_TYPE value); // Declaration only! This operator has to be implemented in derived class.

    protected:
            
        // Making destructor protected prohibits this class from being instantiated. Effectively this class can only be used as a base class.
        ~SIMDVecUnsignedInterface() {};
    public:
        
        // GATHER
        inline DERIVED_VEC_TYPE & gather (SCALAR_TYPE * baseAddr, uint64_t* indices) {
            return EMULATED_FUNCTIONS::gather<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }

        // MGATHER
        inline DERIVED_VEC_TYPE & gather (MASK_TYPE const & mask, SCALAR_TYPE* baseAddr, uint64_t* indices) {
            return EMULATED_FUNCTIONS::gather<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }

        // GATHERV
        inline DERIVED_VEC_TYPE gather (SCALAR_TYPE * baseAddr, DERIVED_VEC_UINT_TYPE const & indices) {
            return EMULATED_FUNCTIONS::gather<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_VEC_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }
        
        // MGATHERV
        inline DERIVED_VEC_TYPE gather (MASK_TYPE const & mask, SCALAR_TYPE* baseAddr, DERIVED_VEC_UINT_TYPE const & indices) {
            return EMULATED_FUNCTIONS::gather<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_VEC_UINT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }

        // SCATTER
        inline SCALAR_TYPE* scatter (SCALAR_TYPE* baseAddr, uint64_t* indices) {
            return EMULATED_FUNCTIONS::scatter<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }
        
        // MSCATTER
        inline SCALAR_TYPE*  scatter (MASK_TYPE const & mask, SCALAR_TYPE* baseAddr, uint64_t* indices) {
            return EMULATED_FUNCTIONS::scatter<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }       

        // SCATTERV
        inline SCALAR_TYPE*  scatter (SCALAR_TYPE* baseAddr, DERIVED_VEC_UINT_TYPE const & indices) {
            return EMULATED_FUNCTIONS::scatter<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_VEC_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }
        
        // MSCATTERV
        inline SCALAR_TYPE*  scatter (MASK_TYPE const & mask, SCALAR_TYPE* baseAddr, DERIVED_VEC_UINT_TYPE const & indices) {
            return EMULATED_FUNCTIONS::scatter<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_VEC_UINT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }       

        // LSHV
        inline DERIVED_VEC_TYPE lsh (DERIVED_VEC_UINT_TYPE const & b) {
            return EMULATED_FUNCTIONS::shiftBitsLeft<DERIVED_VEC_TYPE, DERIVED_VEC_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MLSHV
        inline DERIVED_VEC_TYPE lsh (MASK_TYPE const & mask, DERIVED_VEC_UINT_TYPE const & b) {
            return EMULATED_FUNCTIONS::shiftBitsLeft<DERIVED_VEC_TYPE, DERIVED_VEC_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        // LSHS
        inline DERIVED_VEC_TYPE lsh (SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::shiftBitsLeftScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MLSHS
        inline DERIVED_VEC_TYPE lsh (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::shiftBitsLeftScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // LSHVA
        inline DERIVED_VEC_TYPE & lsha (DERIVED_VEC_UINT_TYPE const & b) {
            return EMULATED_FUNCTIONS::shiftBitsLeftAssign<DERIVED_VEC_TYPE, DERIVED_VEC_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        // MLSHVA
        inline DERIVED_VEC_TYPE & lsha (MASK_TYPE const & mask, DERIVED_VEC_UINT_TYPE const & b) {
            return EMULATED_FUNCTIONS::shiftBitsLeftAssign<DERIVED_VEC_TYPE, DERIVED_VEC_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        // LSHSA
        inline DERIVED_VEC_TYPE & lsha (SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::shiftBitsLeftAssignScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MLSHSA   
        inline DERIVED_VEC_TYPE & lsha (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::shiftBitsLeftAssignScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        // RSHV 
        inline DERIVED_VEC_TYPE rsh (DERIVED_VEC_UINT_TYPE const & b) {
            return EMULATED_FUNCTIONS::shiftBitsRight<DERIVED_VEC_TYPE, DERIVED_VEC_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MRSHV
        inline DERIVED_VEC_TYPE rsh (MASK_TYPE const & mask, DERIVED_VEC_UINT_TYPE const & b) {
            return EMULATED_FUNCTIONS::shiftBitsRight<DERIVED_VEC_TYPE, DERIVED_VEC_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        // RSHS
        inline DERIVED_VEC_TYPE rsh (SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::shiftBitsRightScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MRSHS
        inline DERIVED_VEC_TYPE rsh (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::shiftBitsRightScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // RSHVA
        inline DERIVED_VEC_TYPE & rsha (DERIVED_VEC_UINT_TYPE const & b) {
            return EMULATED_FUNCTIONS::shiftBitsRightAssign<DERIVED_VEC_TYPE, DERIVED_VEC_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        // RSHVA
        inline DERIVED_VEC_TYPE & rsha (MASK_TYPE const & mask, DERIVED_VEC_UINT_TYPE const & b) {
            return EMULATED_FUNCTIONS::shiftBitsRightAssign<DERIVED_VEC_TYPE, DERIVED_VEC_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        // RSHSA
        inline DERIVED_VEC_TYPE & rsha (SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::shiftBitsRightAssignScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MRSHSA
        inline DERIVED_VEC_TYPE & rsha (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::shiftBitsRightAssignScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // ROLV
        inline DERIVED_VEC_TYPE rol (DERIVED_VEC_UINT_TYPE const & b) {
            return EMULATED_FUNCTIONS::rotateBitsLeft<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_VEC_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MROLV
        inline DERIVED_VEC_TYPE rol (MASK_TYPE const & mask, DERIVED_VEC_UINT_TYPE const & b) {
            return EMULATED_FUNCTIONS::rotateBitsLeft<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_VEC_UINT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ROLS
        inline DERIVED_VEC_TYPE rol (SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::rotateBitsLeftScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        // MROLS
        inline DERIVED_VEC_TYPE rol (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::rotateBitsLeftScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ROLVA
        inline DERIVED_VEC_TYPE & rola (DERIVED_VEC_UINT_TYPE const & b) {
            return EMULATED_FUNCTIONS::rotateBitsLeftAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_VEC_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MROLVA
        inline DERIVED_VEC_TYPE & rola (MASK_TYPE const & mask, DERIVED_VEC_UINT_TYPE const & b) {
            return EMULATED_FUNCTIONS::rotateBitsLeftAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_VEC_UINT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // ROLSA
        inline DERIVED_VEC_TYPE & rola (SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::rotateBitsLeftAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MROLSA
        inline DERIVED_VEC_TYPE & rola (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::rotateBitsLeftAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }        
        
        // RORV
        inline DERIVED_VEC_TYPE ror (DERIVED_VEC_UINT_TYPE const & b) {
            return EMULATED_FUNCTIONS::rotateBitsRight<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_VEC_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MRORV
        inline DERIVED_VEC_TYPE ror (MASK_TYPE const & mask, DERIVED_VEC_UINT_TYPE const & b) {
            return EMULATED_FUNCTIONS::rotateBitsRight<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_VEC_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // RORS
        inline DERIVED_VEC_TYPE ror (SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::rotateBitsRightScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MRORS
        inline DERIVED_VEC_TYPE ror (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::rotateBitsRightScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // RORVA
        inline DERIVED_VEC_TYPE rora (DERIVED_VEC_UINT_TYPE const & b) {
            return EMULATED_FUNCTIONS::rotateBitsRightAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_VEC_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MRORVA
        inline DERIVED_VEC_TYPE rora (MASK_TYPE const & mask, DERIVED_VEC_UINT_TYPE const & b) {
            return EMULATED_FUNCTIONS::rotateBitsRightAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_VEC_UINT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // RORSA
        inline DERIVED_VEC_TYPE rora (SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::rotateBitsRightAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MRORSA
        inline DERIVED_VEC_TYPE rora (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::rotateBitsRightAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

    };

    // ***************************************************************************
    // *
    // *    Definition of interface for vectors using SIGNED INTEGER scalar types
    // *
    // ***************************************************************************
    template<typename DERIVED_VEC_TYPE,
             typename DERIVED_VEC_UINT_TYPE,
             typename SCALAR_TYPE, 
             uint32_t VEC_LEN,
             typename SCALAR_UINT_TYPE,
             typename MASK_TYPE>
    class SIMDVecSignedInterface : public SIMDVecUnsignedInterface<
        DERIVED_VEC_TYPE, 
        DERIVED_VEC_UINT_TYPE, 
        SCALAR_TYPE, 
        SCALAR_UINT_TYPE, 
        VEC_LEN, 
        MASK_TYPE>
    {
        // Other vector types necessary for this class
        typedef SIMDVecSignedInterface< DERIVED_VEC_TYPE,
                             DERIVED_VEC_UINT_TYPE,
                             SCALAR_TYPE,
                             VEC_LEN, 
                             SCALAR_UINT_TYPE,
                             MASK_TYPE> VEC_TYPE;

    private:
        // Forbid assignment-initialization of vector using scalar values
        // TODO: is this necessary?
        inline VEC_TYPE & operator= (const int8_t & x) { }
        inline VEC_TYPE & operator= (const int16_t & x) { }
        inline VEC_TYPE & operator= (const int32_t & x) { }
        inline VEC_TYPE & operator= (const int64_t & x) { }
        inline VEC_TYPE & operator= (const uint8_t & x) { }
        inline VEC_TYPE & operator= (const uint16_t & x) { }
        inline VEC_TYPE & operator= (const uint32_t & x) { }
        inline VEC_TYPE & operator= (const uint64_t & x) { }
        inline VEC_TYPE & operator= (const float & x) { }
        inline VEC_TYPE & operator= (const double & x) { }
        
        SCALAR_TYPE operator[] (SCALAR_UINT_TYPE index) const; // Declaration only! This operator has to be implemented in derived class.
        inline DERIVED_VEC_TYPE & insert (uint32_t index, SCALAR_TYPE value); // Declaration only! This operator has to be implemented in derived class.
    protected:
            
        // Making destructor protected prohibits this class from being instantiated. Effectively this class can only be used as a base class.
        ~SIMDVecSignedInterface() {};
    public:
        
        // NEG
        inline DERIVED_VEC_TYPE neg () {
            return EMULATED_FUNCTIONS::unaryMinus<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MNEG
        inline DERIVED_VEC_TYPE neg (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::unaryMinus<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ABS
        inline DERIVED_VEC_TYPE abs () {
            return EMULATED_FUNCTIONS::MATH::abs<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MABS
        inline DERIVED_VEC_TYPE abs (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::MATH::abs<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ABSA
        inline DERIVED_VEC_TYPE absa () {
            return EMULATED_FUNCTIONS::MATH::absAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MABSA
        inline DERIVED_VEC_TYPE absa (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::MATH::absAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
    };

    // ***************************************************************************
    // *
    // *    Definition of interface for vectors using FLOATING POINT scalar types
    // *
    // ***************************************************************************
    template<typename DERIVED_VEC_TYPE,
             typename DERIVED_VEC_UINT_TYPE,
             typename DERIVED_VEC_INT_TYPE, // corresponding integer type
             typename SCALAR_FLOAT_TYPE, 
             uint32_t VEC_LEN,
             typename SCALAR_UINT_TYPE,
             typename MASK_TYPE>     // TODO: REMOVE THIS?
    class SIMDVecFloatInterface : public SIMDVecSignedInterface< 
        DERIVED_VEC_TYPE, 
        DERIVED_VEC_UINT_TYPE, 
        SCALAR_FLOAT_TYPE,
        VEC_LEN,
        SCALAR_UINT_TYPE,
        MASK_TYPE>
    {
        // Other vector types necessary for this class
        typedef SIMDVecFloatInterface< DERIVED_VEC_TYPE,
                    DERIVED_VEC_UINT_TYPE,
                    DERIVED_VEC_INT_TYPE,
                    SCALAR_FLOAT_TYPE,
                    VEC_LEN, 
                    SCALAR_UINT_TYPE,
                    MASK_TYPE> VEC_TYPE;
    private:

        // Forbid assignment-initialization of vector using scalar values
        // TODO: is this necessary?
        inline VEC_TYPE & operator= (const int8_t & x) { }
        inline VEC_TYPE & operator= (const int16_t & x) { }
        inline VEC_TYPE & operator= (const int32_t & x) { }
        inline VEC_TYPE & operator= (const int64_t & x) { }
        inline VEC_TYPE & operator= (const uint8_t & x) { }
        inline VEC_TYPE & operator= (const uint16_t & x) { }
        inline VEC_TYPE & operator= (const uint32_t & x) { }
        inline VEC_TYPE & operator= (const uint64_t & x) { }
        inline VEC_TYPE & operator= (const float & x) { }
        inline VEC_TYPE & operator= (const double & x) { }
 
    protected:
            
        // Making destructor protected prohibits this class from being instantiated. Effectively this class can only be used as a base class.
        ~SIMDVecFloatInterface() {};
        
        SCALAR_FLOAT_TYPE operator[] (SCALAR_UINT_TYPE index) const; // Declaration only! This operator has to be implemented in derived class.
        inline DERIVED_VEC_TYPE & insert(uint32_t index, SCALAR_FLOAT_TYPE value); // Declaration only! This operator has to be implemented in derived class.
    public:

        // ********************************************************************
        // * MATH FUNCTIONS
        // ********************************************************************

        // SQR
        inline DERIVED_VEC_TYPE sqr () {
            return EMULATED_FUNCTIONS::MATH::sqr<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        // MSQR
        // SQRA
        // MSQRA

        // SQRT
        inline DERIVED_VEC_TYPE sqrt () {
            return EMULATED_FUNCTIONS::MATH::sqrt<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // MSQRT
        inline DERIVED_VEC_TYPE sqrt (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::MATH::sqrt<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // SQRTA
        inline DERIVED_VEC_TYPE & sqrta () {
            return EMULATED_FUNCTIONS::MATH::sqrtAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }
        
        // MSQRTA
        inline DERIVED_VEC_TYPE & sqrta (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::MATH::sqrtAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }
        
        // POWV
        // MPOWV
        // POWS
        // MPOWS

        // ROUND
        inline DERIVED_VEC_TYPE round () {
            return EMULATED_FUNCTIONS::MATH::round<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // MROUND
        inline DERIVED_VEC_TYPE round (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::MATH::round<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // TRUNC
        inline DERIVED_VEC_INT_TYPE trunc () {
            return EMULATED_FUNCTIONS::MATH::truncToInt<DERIVED_VEC_TYPE, DERIVED_VEC_INT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MTRUNC
        inline DERIVED_VEC_INT_TYPE trunc (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::MATH::truncToInt<DERIVED_VEC_TYPE, DERIVED_VEC_INT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // FLOOR
        // MFLOOR
        // CEIL
        // MCEIL

        // ISFIN
        // ISINF
        // ISAN
        // ISNAN
        // ISSUB
        // ISZERO
        // ISZEROSUB

        // SIN
        inline DERIVED_VEC_TYPE sin () {
            return EMULATED_FUNCTIONS::MATH::sin<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MSIN
        inline DERIVED_VEC_TYPE sin (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::MATH::sin<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // COS
        inline DERIVED_VEC_TYPE cos () {
            return EMULATED_FUNCTIONS::MATH::cos<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MCOS
        inline DERIVED_VEC_TYPE cos (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::MATH::cos<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // TAN
        // MTAN
        // CTAN
        // MCTAN
    };

} // namespace UME::SIMD
} // namespace UME

#endif
