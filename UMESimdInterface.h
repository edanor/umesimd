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
    #include <math.h>

    inline float       trunc( float f ) { return (f>0) ? floor(f) : ceil(f); }
    inline double      trunc( double d ) { return (d>0) ? floor(d) : ceil(d); }
    inline long double trunc( long double ld ) { return (ld>0) ? floor(ld) : ceil(ld); }
    float round(float d) { return static_cast<float>(static_cast<int>(d + 0.5f)); }
    double round(double d) { return static_cast<double>(static_cast<long>(d + 0.5)); }
    //double      trunc( Integral arg );
    inline bool       isnan( float f ) { return _isnan((double)f) != 0 ? true : false; }
    inline bool       isnan( double d ) { return _isnan(d) != 0 ? true : false; }
    inline bool       isfinite( float f ) { return _finite((double)f) != 0 ? true : false; }
    inline bool       isfinite( double d ) { return _finite(d) != 0 ? true : false; }
    inline bool       isinf( float f ) { return !isfinite(f) && !isnan(f); }
    inline bool       isinf( double d) { return !isfinite(d) && !isnan(d); }
    inline bool       isnormal( float f) {
        uint32_t temp0 = *reinterpret_cast<uint32_t*>(&f);
        uint32_t temp1 = temp0 << 1; // remove sign bit
        uint32_t temp2 = 0xFF000000; 
        uint32_t exponent = temp1 & temp2;    // retrieve exponent
        uint32_t mantisse = temp1 & (~temp2); // retrieve mantisse
        bool issubnormal = (exponent == 0) && (mantisse != 0);
        return (f != 0.0f) && (!issubnormal) && (isfinite(f)) && (!std::isnan(f));
    }
    inline bool       isnormal( double d) {
        uint64_t temp0 = *reinterpret_cast<uint64_t*>(&d);
        uint64_t temp1 = temp0 << 1; // remove sign bit
        uint64_t temp2 = 0xFFE0000000000000ll;
        uint64_t exponent = temp1 & temp2;    // retrieve exponent
        uint64_t mantisse = temp1 & (~temp2); // retrive mantisse
        bool issubnormal = (exponent == 0) && (mantisse != 0);
        return (d != 0.0) && (!issubnormal) && (!std::isnan(d));
    }
}
#endif
#endif

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
        // ASSIGN
        template<typename VEC_TYPE>
        inline VEC_TYPE & assign(VEC_TYPE & dst, VEC_TYPE const & src) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, src[i]);
            }
            return dst;
        }

        // MASSIGN
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & assign(MASK_TYPE const & mask, VEC_TYPE & dst, VEC_TYPE const & src) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) dst.insert(i, src[i]);
            }
            return dst;
        }

        // ASSIGNS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE & assign(VEC_TYPE & dst, SCALAR_TYPE src) {
            UME_EMULATION_WARNING();
            for( uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, src);
            }
            return dst;
        }

        // MASSIGNS
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
        inline SCALAR_TYPE* store(VEC_TYPE const & src, SCALAR_TYPE * p) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++)
            {
                p[i] = src[i];
            }
            return p;
        }

        // MSTORE
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline SCALAR_TYPE* store(MASK_TYPE const & mask, VEC_TYPE & src, SCALAR_TYPE * p) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++)
            {
                if(mask[i] == true) p[i] = src[i];
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

        // PACK
        template<typename VEC_TYPE, typename VEC_HALF_TYPE>
        inline VEC_TYPE & pack(VEC_TYPE & dst, VEC_HALF_TYPE const & src1, VEC_HALF_TYPE const & src2) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_HALF_TYPE::length(); i++) {
                dst.insert(i, src1[i]);
                dst.insert(i + VEC_HALF_TYPE::length(), src2[i]);
            }
            return dst;
        }

        // PACKLO
        template<typename VEC_TYPE, typename VEC_HALF_TYPE>
        inline VEC_TYPE & packLow(VEC_TYPE & dst, VEC_HALF_TYPE const & src1) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_HALF_TYPE::length(); i++) {
                dst.insert(i, src1[i]);
            }
            return dst;
        }

        // PACKHI
        template<typename VEC_TYPE, typename VEC_HALF_TYPE>
        inline VEC_TYPE & packHigh(VEC_TYPE & dst, VEC_HALF_TYPE const & src1) {
            UME_EMULATION_WARNING();
            for(uint32_t i = VEC_HALF_TYPE::length(); i < VEC_TYPE::length(); i++) {
                dst.insert(i, src1[i - VEC_HALF_TYPE::length()]);
            }
            return dst;
        }
        
        // UNPACK
        template<typename VEC_TYPE, typename VEC_HALF_TYPE>
        inline void unpack(VEC_TYPE const & src, VEC_HALF_TYPE & dst1, VEC_HALF_TYPE & dst2) {
            UME_EMULATION_WARNING();
            uint32_t halfLength = VEC_HALF_TYPE::length();
            for(uint32_t i = 0; i < halfLength; i++) {
                dst1.insert(i, src[i]);
                dst2.insert(i, src[i + halfLength]);
            }
        }

        // UNPACKLO
        template<typename VEC_TYPE, typename VEC_HALF_TYPE>
        inline VEC_HALF_TYPE unpackLow(VEC_TYPE const & src) {
            UME_EMULATION_WARNING();
            VEC_HALF_TYPE retval;
            for(uint32_t i = 0; i < VEC_HALF_TYPE::length(); i++) {
                retval.insert(i, src[i]);
            }
            return retval;
        }

        // UNPACKHI
        template<typename VEC_TYPE, typename VEC_HALF_TYPE>
        inline VEC_HALF_TYPE unpackHigh(VEC_TYPE const & src) {
            UME_EMULATION_WARNING();
            VEC_HALF_TYPE retval;
            for(uint32_t i = 0; i < VEC_HALF_TYPE::length(); i++) {
                retval.insert(i, src[i + VEC_HALF_TYPE::length()]);
            }
            return retval;
        }

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
            
        // ADDVA
        template<typename VEC_TYPE>
        inline VEC_TYPE & addAssign (VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) { a.insert(i, (a[i] + b[i])); }
            return a;
        }

        // MADDVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & addAssign (MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, (a[i] + b[i]));
            }
            return a;
        }

        // ADDSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE & addAssignScalar (VEC_TYPE & a, SCALAR_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, (a[i] + b));
            }
            return a;
        }

        // MADDSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & addAssignScalar (MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i] + b);
            }
            return a;
        }

        // SADDV
        template<typename VEC_TYPE>
        inline VEC_TYPE addSaturated (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            decltype(a[0]) temp = 0;
            // maximum value
            decltype(a[0]) satValue = std::numeric_limits<decltype(a[0])>::max();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = (a[i] > (satValue - b[i])) ? satValue : (a[i] + b[i]);
                retval.insert(i, temp);
            }
            return retval;
        }

        // MSADDV
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE addSaturated (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            decltype(a[0]) temp = 0;
            // maximum value
            decltype(a[0]) satValue = std::numeric_limits<decltype(a[0])>::max();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) {
                    temp = (a[i] > (satValue - b[i])) ? satValue : (a[i] + b[i]);
                    retval.insert(i, temp);
                }
                else {
                    retval.insert(i, a[i]);
                }
            }
            return retval;
        }

        // SADDS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE addSaturatedScalar (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            decltype(a[0]) temp = 0;
            // maximum value
            decltype(a[0]) satValue = std::numeric_limits<decltype(a[0])>::max();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = (a[i] > (satValue - b)) ? satValue : (a[i] + b);
                retval.insert(i, temp);
            }
            return retval;
        }

        // MSADDS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE addSaturatedScalar (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            decltype(a[0]) temp = 0;
            // maximum value
            decltype(a[0]) satValue = std::numeric_limits<decltype(a[0])>::max();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) {
                    temp = (a[i] > (satValue - b)) ? satValue : (a[i] + b);
                    retval.insert(i, temp);
                }
                else {
                    retval.insert(i, a[i]);
                }
            }
            return retval;
        }

        // SADDVA
        template<typename VEC_TYPE>
        inline VEC_TYPE & addSaturatedAssign(VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            decltype(a[0]) temp = 0;
            // maximum value
            decltype(a[0]) satValue = std::numeric_limits<decltype(a[0])>::max();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = (a[i] > (satValue - b[i])) ? satValue : (a[i] + b[i]);
                a.insert(i, temp);
            }
            return a;
        }
        
        // MSADDVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & addSaturatedAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            decltype(a[0]) temp = 0;
            // maximum value
            decltype(a[0]) satValue = std::numeric_limits<decltype(a[0])>::max();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) {
                    temp = (a[i] > (satValue - b[i])) ? satValue : (a[i] + b[i]);
                    a.insert(i, temp);
                }
            }
            return a;
        }

        // SADDSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE & addSaturatedScalarAssign(VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            decltype(a[0]) temp = 0;
            // maximum value
            decltype(a[0]) satValue = std::numeric_limits<decltype(a[0])>::max();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = (a[i] > (satValue - b)) ? satValue : (a[i] + b);
                a.insert(i, temp);
            }
            return a;
        }

        // MSADDSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & addSaturatedScalarAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            decltype(a[0]) temp = 0;
            // maximum value
            decltype(a[0]) satValue = std::numeric_limits<decltype(a[0])>::max();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) {
                    temp = (a[i] > (satValue - b)) ? satValue : (a[i] + b);
                    a.insert(i, temp);
                }
            }
            return a;
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
                else retval.insert(i, a[i]);
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
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // SUBFROMV
        template<typename VEC_TYPE>
        inline VEC_TYPE subFrom (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] - b[i]);
            }
            return retval;
        }

        // MSUBFROMV
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE subFrom (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) retval.insert(i, a[i] - b[i]);
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // SUBFROMS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE subFromScalar (SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a - b[i]);
            }
            return retval;
        }

        // MSUBFROMS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE subFromScalar (MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) retval.insert(i, a - b[i]);
                else retval.insert(i, a);
            }
            return retval;
        }

        // SUBFROMVA
        template<typename VEC_TYPE>
        inline VEC_TYPE & subFromAssign (VEC_TYPE const & a, VEC_TYPE & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                b.insert(i, a[i] - b[i]);
            }
            return b;
        }

        // MSUBFROMVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & subFromAssign (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) b.insert(i, a[i] - b[i]);
                else b.insert(i, a[i]);
            }
            return b;
        }

        // SUBFROMSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE & subFromScalarAssign (SCALAR_TYPE a, VEC_TYPE & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                b.insert(i, a - b[i]);
            }
            return b;
        }

        // MSUBFROMSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & subFromScalarAssign (MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) b.insert(i, a - b[i]);
                else b.insert(i, a);
            }
            return b;
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
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // NEGA
        template<typename VEC_TYPE>
        inline VEC_TYPE & unaryMinusAssign (VEC_TYPE & a) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, -a[i]);
            }
            return a;
        }

        // MNEGA
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & unaryMinusAssign (MASK_TYPE const & mask, VEC_TYPE & a) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if( mask[i] == true ) a.insert(i, -a[i]);
            }
            return a;
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

        // SSUBV
        template<typename VEC_TYPE>
        inline VEC_TYPE subSaturated (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            decltype(a[0]) temp = 0;
            // maximum value
            decltype(a[0]) satValue = std::numeric_limits<decltype(a[0])>::min();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = (a[i] < (satValue + b[i])) ? satValue : (a[i] - b[i]);
                retval.insert(i, temp);
            }
            return retval;
        }

        // MSSUBV
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE subSaturated (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            decltype(a[0]) temp = 0;
            // maximum value
            decltype(a[0]) satValue = std::numeric_limits<decltype(a[0])>::min();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) {
                    temp = (a[i] < (satValue + b[i])) ? satValue : (a[i] - b[i]);
                    retval.insert(i, temp);
                }
                else {
                    retval.insert(i, a[i]);
                }
            }
            return retval;
        }

        // SSUBS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE subSaturated (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            decltype(a[0]) temp = 0;
            // maximum value
            decltype(a[0]) satValue = std::numeric_limits<decltype(a[0])>::min();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = (a[i] < (satValue + b)) ? satValue : (a[i] - b);
                retval.insert(i, temp);
            }
            return retval;
        }

        // MSSUBS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE subSaturated (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            decltype(a[0]) temp = 0;
            // maximum value
            decltype(a[0]) satValue = std::numeric_limits<decltype(a[0])>::min();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) {
                    temp = (a[i] < (satValue + b)) ? satValue : (a[i] - b);
                    retval.insert(i, temp);
                }
                else {
                    retval.insert(i, a[i]);
                }
            }
            return retval;
        }

        // SSUBVA
        template<typename VEC_TYPE>
        inline VEC_TYPE & subSaturatedAssign (VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            decltype(a[0]) temp = 0;
            // maximum value
            decltype(a[0]) satValue = std::numeric_limits<decltype(a[0])>::min();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = (a[i] < (satValue + b[i])) ? satValue : (a[i] - b[i]);
                a.insert(i, temp);
            }
            return a;
        }

        // MSSUBV
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & subSaturatedAssign (MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            decltype(a[0]) temp = 0;
            // maximum value
            decltype(a[0]) satValue = std::numeric_limits<decltype(a[0])>::min();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) {
                    temp = (a[i] < (satValue + b[i])) ? satValue : (a[i] - b[i]);
                    a.insert(i, temp);
                }
            }
            return a;
        }

        // SSUBS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE & subSaturatedScalarAssign (VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            decltype(a[0]) temp = 0;
            // maximum value
            decltype(a[0]) satValue = std::numeric_limits<decltype(a[0])>::min();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = (a[i] < (satValue + b)) ? satValue : (a[i] - b);
                a.insert(i, temp);
            }
            return a;
        }

        // MSSUBS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & subSaturatedScalarAssign (MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            decltype(a[0]) temp = 0;
            // maximum value
            decltype(a[0]) satValue = std::numeric_limits<decltype(a[0])>::min();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) {
                    temp = (a[i] < (satValue + b)) ? satValue : (a[i] - b);
                    a.insert(i, temp);
                }
            }
            return a;
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
                if(mask[i] == true) dst.insert(i, dst[i] * b[i]);
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
            return a;
        }

        // RCP
        template<typename VEC_TYPE>
        inline VEC_TYPE rcp(VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, decltype(retval[0])(1.0)/b[i]);
            }
            return retval;
        }
        
        // MRCP
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE rcp(MASK_TYPE const & mask, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) retval.insert(i, decltype(retval[0])(1.0)/b[i]);
                else retval.insert(i, b[i]);
            }
            return retval;
        }
        
        // RCPS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE rcpScalar(SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a/b[i]);
            }
            return retval;
        }

        // MRCPS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE rcpScalar(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) retval.insert(i, a/b[i]);
                else retval.insert(i, b[i]);
            }
            return retval;
        }

        // RCPA
        template<typename VEC_TYPE>
        inline VEC_TYPE & rcpAssign(VEC_TYPE & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                b.insert(i, decltype(b[0])(1.0)/b[i]);
            }
            return b;
        }
        
        // MRCPA
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & rcpAssign(MASK_TYPE const & mask, VEC_TYPE & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) b.insert(i, decltype(b[0])(1.0)/b[i]);
            }
            return b;
        }

        // RCPSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        inline VEC_TYPE & rcpScalarAssign(SCALAR_TYPE a, VEC_TYPE & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                b.insert(i, a/b[i]);
            }
            return b;
        }

        // MRCPSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & rcpScalarAssign(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) b.insert(i, a/b[i]);
            }
            return b;
        }

        // LSHV
        template<typename VEC_TYPE, typename UINT_VEC_TYPE>
        inline VEC_TYPE shiftBitsLeft(VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (a[i] << b[i]) );
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
                retval.insert(i, (a[i] << b) );
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
                a.insert(i, (a[i] << b[i]));
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
                a.insert(i, (a[i] << b) );
            }
            return a;
        }

        // MLSHSA
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & shiftBitsLeftAssignScalar(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, (a[i] << b) );
            }
            return a;
        }
            
        // RSHV
        template<typename VEC_TYPE, typename UINT_VEC_TYPE>
        inline VEC_TYPE shiftBitsRight(VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (a[i] >> b[i]));
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
                retval.insert(i, (a[i] >> b) );
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
                a.insert(i, (a[i] >> b[i]));
            }
            return a;
        }
            
        // MRSHVA
        template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & shiftBitsRightAssign(MASK_TYPE const & mask, VEC_TYPE & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, (a[i] >> b[i]) );
            }
            return a;
        }
            
        // RSHSA
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE>
        inline VEC_TYPE & shiftBitsRightAssignScalar(VEC_TYPE & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, (a[i] >> b) );
            }
            return a;
        }
                
        // MSRHSA
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & shiftBitsRightAssignScalar(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, (a[i] >> b) );
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

        // CMPEX 
        template<typename VEC_TYPE>
        inline bool isExact(VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            bool retval = true;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(a[i] != b[i]) {
                    retval = false;
                    break;
                }
            }
            return retval;
        }
        
        // CMPEQRV
        template<typename MASK_TYPE, typename VEC_TYPE>
        inline MASK_TYPE isEqualInRange(VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & margin) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if((a[i] < b[i] + margin[i]) && (a[i] > b[i] - margin[i]))
                    retval.insert(i, true);       
                else 
                    retval.insert(i, false);
            }
            return retval;
        }

        // CMPEQRS
        template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
        inline MASK_TYPE isEqualInRange(VEC_TYPE const & a, VEC_TYPE const & b, SCALAR_TYPE margin) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if((a[i] < b[i] + margin) && (a[i] > b[i] - margin))
                    retval.insert(i, true);
                else
                    retval.insert(i, false);
            }
            return retval;
        }

        // UNIQUE
        template<typename VEC_TYPE>
        inline bool unique(VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            bool retval = true;
            for (uint32_t i = 0; i < VEC_TYPE::length()-1; i++) {
                for (uint32_t j = i + 1; j < VEC_TYPE::length(); j++) {
                    if (a[i] == a[j])
                    {
                        // Break earlier from innermost loop
                        retval = false;
                        break;
                    }
                }
                // Break earlier from outermost loop
                if (retval == false) break;
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

        // BNOT
        template<typename VEC_TYPE>
        inline VEC_TYPE binaryNot (VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                decltype(a[0]) temp = a[i];
                decltype(a[0]) temp2 = ~a[i];
                retval.insert(i, temp2); //~a[i]);
            }
            return retval;
        }

        // MBNOT
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE binaryNot (MASK_TYPE const & mask, VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? (~a[i]) : (a[i]));
            }
            return retval;
        }

        // BNOTA
        template<typename VEC_TYPE>
        inline VEC_TYPE & binaryNotAssign (VEC_TYPE & a) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, ~a[i]);
            }
            return a;
        }
        
        // MBNOTA
        template<typename VEC_TYPE, typename MASK_TYPE>
        inline VEC_TYPE & binaryNotAssign (MASK_TYPE const & mask, VEC_TYPE & a) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, ~a[i]);
            }
            return a;
        }

        // LNOT
        template<typename MASK_TYPE>
        inline MASK_TYPE logicalNot(MASK_TYPE const & mask) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval(false);
            for(uint32_t i = 0; i < MASK_TYPE::length(); i++) {
                if(mask[i] == false) retval.insert(i, true);
            }
            return retval;
        }
        
        // LNOTA
        template<typename MASK_TYPE>
        inline MASK_TYPE & logicalNotAssign(MASK_TYPE & mask) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < MASK_TYPE::length(); i++) {
                mask.insert(i, !mask[i]);
            }
            return mask;
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
        
        // SWIZZLE
        template<typename VEC_TYPE, typename SWIZZLE_MASK_TYPE>
        inline VEC_TYPE swizzle(SWIZZLE_MASK_TYPE const & sMask, VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[sMask[i]]);
            }
            return retval;
        }

        // SWIZZLEA
        template<typename VEC_TYPE, typename SWIZZLE_MASK_TYPE>
        inline VEC_TYPE & swizzleAssign(SWIZZLE_MASK_TYPE const & sMask, VEC_TYPE & a) {
            UME_EMULATION_WARNING();
            VEC_TYPE temp(a);
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, temp[sMask[i]]);
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
        
        // HLAND
        template<typename MASK_TYPE>
        inline bool reduceLogicalAnd(MASK_TYPE const & a) {
            UME_EMULATION_WARNING();
            bool retval = a[0];
            for(uint32_t i = 1; i < MASK_TYPE::length(); i++) {
                retval &= a[i];
            }
            return retval;
        }

        // HLOR
        template<typename MASK_TYPE>
        inline bool reduceLogicalOr(MASK_TYPE const & a) {
            UME_EMULATION_WARNING();
            bool retval = a[0];
            for(uint32_t i = 1; i < MASK_TYPE::length(); i++) {
                retval |= a[i];
            }
            return retval;
        }
        
        // HLXOR
        template<typename MASK_TYPE>
        inline bool reduceLogicalXor(MASK_TYPE const & a) {
            UME_EMULATION_WARNING();
            bool retval = a[0];
            for(uint32_t i = 1; i < MASK_TYPE::length(); i++) {
                retval ^= a[i];
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

        // reduceBinaryXor() -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        inline SCALAR_TYPE reduceBinaryXor(VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = 0;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) { 
                retval ^= a[i];
            }
            return retval;
        }

        // reduceBinaryXor(MASK) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        inline SCALAR_TYPE reduceBinaryXor(MASK_TYPE const & mask, VEC_TYPE const & a) {
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

		// xTOy (UTOI, ITOU, UTOF, FTOU)
		template<typename UINT_VEC_TYPE, typename INT_VEC_TYPE>
		inline UINT_VEC_TYPE xtoy(INT_VEC_TYPE const & a) {
			UME_EMULATION_WARNING();
			static_assert(UINT_VEC_TYPE::length() == INT_VEC_TYPE::length(),
				"Cannot cast between vectors of different lengths");
			UINT_VEC_TYPE retval;
			for (uint32_t i = 0; i < INT_VEC_TYPE::length();i++) {
				retval.insert(i, decltype(retval[0])(a[i]));
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
                    else retval.insert(i, a[i]);
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
                    else retval.insert(i, a[i]);
                }
                return retval;
            }
            
            // MAXVA
            template<typename VEC_TYPE>
            inline VEC_TYPE & maxAssign(VEC_TYPE & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(b[i] > a[i])a.insert(i, b[i]);
                }
                return a;
            }

            // MMAXVA
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE & maxAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] ==true && (b[i] > a[i]))a.insert(i, b[i]);
                }
                return a;
            }

            // MAXSA
            template<typename VEC_TYPE, typename SCALAR_TYPE>
            inline VEC_TYPE & maxScalarAssign(VEC_TYPE & a, SCALAR_TYPE b) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(b > a[i]) a.insert(i, b);
                }
                return a;
            }

            // MMAXSA
            template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
            inline VEC_TYPE & maxScalarAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true && (b > a[i])) a.insert(i, b);
                }
                return a;
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
                    else retval.insert(i, a[i]);
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
                    else retval.insert(i, a[i]);
                }
                return retval;
            }

            // MINSA
            template<typename VEC_TYPE, typename SCALAR_TYPE>
            inline VEC_TYPE & minScalarAssign(VEC_TYPE & a, SCALAR_TYPE b) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(b < a[i]) a.insert(i, b);
                }
                return a;
            }

            // MMINSA
            template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
            inline VEC_TYPE & minScalarAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true && (b < a[i])) a.insert(i, b);
                }
                return a;
            }
            
            // MINVA
            template<typename VEC_TYPE>
            inline VEC_TYPE & minAssign(VEC_TYPE & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(b[i] < a[i]) a.insert(i, b[i]);
                }
                return a;
            }

            // MMINVA
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE & minAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true && (b[i] < a[i])) a.insert(i, b[i]);
                }
                return a;
            }

            // HMAX
            template<typename SCALAR_TYPE, typename VEC_TYPE>
            inline SCALAR_TYPE reduceMax(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                SCALAR_TYPE retval = a[0];
                for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                    if(a[i] > retval) retval = a[i];
                }
                return retval;
            }
            
            // MHMAX
            template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
            inline SCALAR_TYPE reduceMax(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                SCALAR_TYPE retval = std::numeric_limits<SCALAR_TYPE>::min();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if( (mask[i] == true) && a[i] > retval) retval = a[i];
                }
                return retval;
            }

            // HMAXS
            template<typename SCALAR_TYPE, typename VEC_TYPE>
            inline SCALAR_TYPE reduceMax(SCALAR_TYPE a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                SCALAR_TYPE retval = a;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(b[i] > retval) retval = b[i];
                }
                return retval;
            }

            // MHMAXS
            template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
            inline SCALAR_TYPE reduceMax(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                SCALAR_TYPE retval = a;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if((mask[i] == true) && (a[i] > retval)) retval = a[i];
                }
                return retval;
            }

            // IMAX
            template<typename VEC_TYPE>
            inline uint32_t indexMax(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                uint32_t indexMax = 0;
                decltype(a[0]) maxVal = a[0];
                for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                    if(a[i] > maxVal) {
                        maxVal = a[i];
                        indexMax = i;
                    }
                }
                return indexMax;
            }

            // MIMAX
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline uint32_t indexMax(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                uint32_t indexMax = 0xFFFFFFFF;
                decltype(a[0]) maxVal = std::numeric_limits<decltype(a[0])>::min();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true && a[i] > maxVal) {
                        maxVal = a[i];
                        indexMax = i;
                    }
                }
                return indexMax;
            }

            // HMIN
            template<typename SCALAR_TYPE, typename VEC_TYPE> 
            inline SCALAR_TYPE reduceMin(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                SCALAR_TYPE retval = a[0];
                for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                    if(a[i] < retval) retval = a[i];
                }
                return retval;
            }

            // MHMIN
            template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
            inline SCALAR_TYPE reduceMin(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                SCALAR_TYPE retval = std::numeric_limits<SCALAR_TYPE>::max();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if( (mask[i] == true) && a[i] < retval) retval = a[i];
                }
                return retval;
            }

            // IMIN
            template<typename VEC_TYPE>
            inline uint32_t indexMin(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                uint32_t indexMin = 0;
                decltype(a[0]) minVal = std::numeric_limits<decltype(a[0])>::max();
                for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                    if(a[i] < minVal) {
                        minVal = a[i];
                        indexMin = i;
                    }
                }
                return indexMin;
            }

            // MIMIN
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline uint32_t indexMin(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                uint32_t indexMin = 0xFFFFFFFF;
                decltype(a[0]) minVal = std::numeric_limits<decltype(a[0])>::max();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true && a[i] < minVal) {
                        minVal = a[i];
                        indexMin = i;
                    }
                }
                return indexMin;
            }

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
                    else retval.insert(i, a[i]);
                }
                return retval;
            }
            
            // SQRA
            template<typename VEC_TYPE>
            inline VEC_TYPE & sqrAssign(VEC_TYPE & a) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    a.insert(i, a[i] * a[i]);
                }
                return a;
            }
            
            // MSQRA
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE & sqrAssign(MASK_TYPE const & mask, VEC_TYPE & a) {
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

            // RSQRT
            template<typename VEC_TYPE>
            inline VEC_TYPE rsqrt(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, decltype(retval[0])(1.0)/std::sqrt(a[i])); 
                }
                return retval;
            }
            // MRSQRT
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE rsqrt(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                decltype(retval[0]) temp;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    temp = decltype(retval[0])(1.0)/std::sqrt(a[i]);
                    retval.insert(i, (mask[i] == true) ? temp : a[i]);
                }
                return retval;
            }
            // RSQRTA
            template<typename VEC_TYPE>
            inline VEC_TYPE & rsqrtAssign (VEC_TYPE & a) {
                UME_EMULATION_WARNING();
                decltype(a[0]) temp;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    temp = decltype(a[0])(1.0)/std::sqrt(a[i]);
                    a.insert(i, temp);
                }
                return a;
            }
            // MRSQRTA
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE & rsqrtAssign(MASK_TYPE const & mask, VEC_TYPE & a) {
                UME_EMULATION_WARNING();
                decltype(a[0]) temp;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    temp = decltype(a[0])(1.0)/std::sqrt(a[i]);
                    if(mask[i] == true) a.insert(i, temp);
                }
                return a;
            }
            
            // POWV
            template<typename VEC_TYPE>
            inline VEC_TYPE pow(VEC_TYPE const & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::pow(a[i], b[i]));
                }
                return retval;
            }

            // MPOWV
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE pow(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, std::pow(a[i], b[i]));
                    else retval.insert(i, a[i]);
                }
                return retval;
            }

            // POWS
            template<typename VEC_TYPE, typename SCALAR_TYPE>
            inline VEC_TYPE pows(VEC_TYPE const & a, SCALAR_TYPE b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::pow(a[i], b));
                }
                return retval;
            }

            // MPOWS
            template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
            inline VEC_TYPE pows(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, std::pow(a[i], b));
                    else retval.insert(i, a[i]);
                }
                return retval;
            }
            
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
            template<typename VEC_TYPE>
            inline VEC_TYPE floor(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length();i++) {
                    retval.insert(i, std::floor(a[i]));
                }
                return retval;
            }

            // MFLOOR
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE floor(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length();i++) {
                    if(mask[i] == true) retval.insert(i, std::floor(a[i]));
                    else retval.insert(i, a[i]);
                }
                return retval;
            }
            
            // CEIL
            template<typename VEC_TYPE>
            inline VEC_TYPE ceil(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length();i++) {
                    retval.insert(i, std::ceil(a[i]));
                }
                return retval;
            }

            // MCEIL
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE ceil(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length();i++) {
                    if(mask[i] == true) retval.insert(i, std::ceil(a[i]));
                    else retval.insert(i, a[i]);
                }
                return retval;
            }
            
            // FMULADDV
            template<typename VEC_TYPE>
            inline VEC_TYPE fmuladd(VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (a[i]*b[i]) + c[i]);
                }
                return retval;
            }

            // MFMULADDV
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE fmuladd(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, (a[i]*b[i]) + c[i]);
                    else retval.insert(i, a[i]);
                }
                return retval;
            }
            
            // FADDMULV
            template<typename VEC_TYPE>
            inline VEC_TYPE faddmul(VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (a[i] + b[i]) * c[i]);
                }
                return retval;
            }

            // MFADDMULV
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE faddmul(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, (a[i] + b[i]) * c[i]);
                    else retval.insert(i, a[i]);
                }
                return retval;
            }

            // FMULSUBV
            template<typename VEC_TYPE>
            inline VEC_TYPE fmulsub(VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (a[i]*b[i]) - c[i]);
                }
                return retval;
            }

            // MFMULSUBV
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE fmulsub(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, (a[i]*b[i]) - c[i]);
                    else retval.insert(i, a[i]);
                }
                return retval;
            }

            // FSUBMULV
            template<typename VEC_TYPE>
            inline VEC_TYPE fsubmul(VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (a[i] - b[i]) * c[i]);
                }
                return retval;
            }

            // MFSUBMULV
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE fsubmul(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, (a[i] - b[i]) * c[i]);
                    else retval.insert(i, a[i]);
                }
                return retval;
            }

            // ISFIN
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline MASK_TYPE isfin(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                MASK_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::isfinite(a[i]));
                }
                return retval;
            }

            // ISINF
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline MASK_TYPE isinf(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                MASK_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::isinf(a[i]));
                }
                return retval;
            }

            // ISAN
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline MASK_TYPE isan(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                MASK_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, !std::isnan(a[i]));
                }
                return retval;
            }

            // ISNAN
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline MASK_TYPE isnan(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                MASK_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::isnan(a[i]));
                }
                return retval;
            }

            // ISNORM
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline MASK_TYPE isnorm(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                MASK_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::isnormal(a[i]));
                }
                return retval;
            }

            // ISSUB
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline MASK_TYPE issub(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                MASK_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    bool isZero = (a[i] == (decltype(a[0])(0.0)));
                    bool isNormal = std::isnormal(a[i]);
                    bool isFinite = std::isfinite(a[i]);
                    bool isNan = std::isnan(a[i]);
                    bool isSubnormal = !isNan && isFinite && !isZero && !isNormal;
                    retval.insert(i, isSubnormal);
                }
                return retval;
            }

            // ISZERO
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline MASK_TYPE iszero(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                MASK_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (a[i] == (decltype(a[0])(0.0))));
                }
                return retval;
            }

            // ISZEROSUB
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline MASK_TYPE iszerosub(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                MASK_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    bool isZero = (a[i] == (decltype(a[0])(0.0)));
                    bool isNormal = std::isnormal(a[i]);
                    bool isFinite = std::isfinite(a[i]);
                    bool isNan = std::isnan(a[i]);
                    bool isSubnormal = !isNan && isFinite && !isZero && !isNormal;
                    retval.insert(i, isSubnormal || isZero);
                }
                return retval;
            }

            // EXP
            template<typename VEC_TYPE>
            inline VEC_TYPE exp (VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::exp(a[i]));
                }
                return retval;
            }

            // MEXP
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE exp (MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (mask[i] == true) ? std::exp(a[i]) : a[i]);
                }
                return retval;
            }

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
            template<typename VEC_TYPE>
            inline VEC_TYPE tan (VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::tan(a[i]));
                }
                return retval;
            }

            // MTAN
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE tan (MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, std::tan(a[i]));
                    else retval.insert(i, a[i]);
                }
                return retval;
            }

            // CTAN
            template<typename VEC_TYPE>
            inline VEC_TYPE ctan (VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, decltype(retval[0])(1.0)/std::tan(a[i]));
                }
                return retval;
            }

            // MCTAN
            template<typename VEC_TYPE, typename MASK_TYPE>
            inline VEC_TYPE ctan (MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, decltype(retval[0])(1.0)/std::tan(a[i]));
                    else retval.insert(i, a[i]);
                }
                return retval;
            }

        } // UME::SIMD::EMULATED_FUNCTIONS::MATH
    } // namespace UME::SIMD::EMULATED_FUNCTIONS
    
    // **********************************************************************
    // *
    // *  Declaration of SwizzleMaskInterface class
    // *
    // **********************************************************************
 
    //// Checks if N is power of 2
    //template<unsigned int N>
    //struct isPow2
    //{
    //    enum {
    //        value = N && !(N & (N -1))
    //    };
    //};

    //// Calculates number of bits required to represent element of swizzle mask.
    //template<unsigned int N, unsigned int P=0>
    //struct SwizzleMaskBitsPerElement
    //{
    //    //static const unsigned int value = LogBase2<N/2, P+1>.value;
    //    enum {
    //        value = SwizzleMaskBitsPerElement<N/2 + !(isPow2<N>::value), P+1>::value
    //    };
    //};

    //// Partial specialization for base case
    //template<unsigned P>
    //struct SwizzleMaskBitsPerElement<0, P>
    //{
    //    enum {
    //        value = P
    //    };
    //};

    //template<unsigned P>
    //struct SwizzleMaskBitsPerElement<1, P>
    //{
    //    enum {
    //        value = P
    //    };
    //};
    
    template<class DERIVED_MASK_TYPE, uint32_t SMASK_LEN>
    class SIMDSwizzleMaskBaseInterface
    {
        // Declarations only. These operators should be overriden in derived types.
        // EXTRACT
        inline bool extract(uint32_t index);
        // EXTRACT
        inline bool operator[] (uint32_t index);
        // INSERT
        inline void insert(uint32_t index, uint32_t value);

    protected:
        ~SIMDSwizzleMaskBaseInterface() {};

    public:
        // LENGTH
        constexpr static uint32_t length () { return SMASK_LEN; };

        // ALIGNMENT
        static int alignment () { return SMASK_LEN*sizeof(uint32_t); };
    };

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
         /*      
        inline ScalarTypeWrapper & operator=(ScalarTypeWrapper const & x){
            mValue = x.mValue;
            return *this;
        }*/

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
        // Declarations only. These operators should be overriden in derived types.
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
        constexpr static uint32_t length () { return MASK_LEN; };

        // ALIGNMENT
        constexpr static int alignment () { return MASK_LEN*sizeof(MASK_BASE_TYPE); };
        
        // LOAD
        inline DERIVED_MASK_TYPE & load (bool const * addr) {
            return EMULATED_FUNCTIONS::load<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE &>(*this), addr);
        };

        // LOADA
        inline DERIVED_MASK_TYPE & loadAligned (bool const * addrAligned) {
            return EMULATED_FUNCTIONS::loadAligned<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE &>(*this), addrAligned);
        };

        // STORE
        inline bool* store (bool* addr) const {
            return EMULATED_FUNCTIONS::store<DERIVED_MASK_TYPE, bool> (static_cast<DERIVED_MASK_TYPE const &>(*this), addr);
        };

        // STOREA
        inline bool* storea (bool* addrAligned) const {
            return EMULATED_FUNCTIONS::storeAligned<DERIVED_MASK_TYPE, bool> (static_cast<DERIVED_MASK_TYPE const &>(*this), addrAligned);
        };

        // ASSIGN
        inline DERIVED_MASK_TYPE & assign (DERIVED_MASK_TYPE const & maskOp) {
            return EMULATED_FUNCTIONS::assign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        };

        inline DERIVED_MASK_TYPE & operator= (DERIVED_MASK_TYPE const & maskOp) {
            return EMULATED_FUNCTIONS::assign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        };

        // LAND 
        inline DERIVED_MASK_TYPE land ( DERIVED_MASK_TYPE const & maskOp) const {
            return EMULATED_FUNCTIONS::binaryAnd<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), maskOp);
        };
        
        inline DERIVED_MASK_TYPE operator& ( DERIVED_MASK_TYPE const & maskOp) const {
            return land(maskOp);
        };

        // LANDA
        inline DERIVED_MASK_TYPE & landa (DERIVED_MASK_TYPE const & maskOp) {
            return EMULATED_FUNCTIONS::binaryAndAssign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        };
        
        inline DERIVED_MASK_TYPE & operator&= (DERIVED_MASK_TYPE const & maskOp) {
            return landa(maskOp);
        };

        // LOR
        inline DERIVED_MASK_TYPE lor (DERIVED_MASK_TYPE const & maskOp) const {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), maskOp);
        }

        inline DERIVED_MASK_TYPE operator| (DERIVED_MASK_TYPE const & maskOp) const {
            return lor(maskOp);
        }

        // LORA
        inline DERIVED_MASK_TYPE & lora (DERIVED_MASK_TYPE const & maskOp) {
            return EMULATED_FUNCTIONS::binaryOrAssign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        }

        inline DERIVED_MASK_TYPE & operator|= (DERIVED_MASK_TYPE const & maskOp) {
            return lora(maskOp);
        }

        // LXOR
        inline DERIVED_MASK_TYPE lxor (DERIVED_MASK_TYPE const & maskOp) const {
            return EMULATED_FUNCTIONS::binaryXor<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), maskOp);
        }
        
        inline DERIVED_MASK_TYPE operator^ (DERIVED_MASK_TYPE const & maskOp) const {
            return lxor(maskOp);
        }

        // LXORA
        inline DERIVED_MASK_TYPE & lxora (DERIVED_MASK_TYPE const & maskOp) {
            return EMULATED_FUNCTIONS::binaryXorAssign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        }

        inline DERIVED_MASK_TYPE & operator^= (DERIVED_MASK_TYPE const & maskOp) {
            return lxora(maskOp);
        }

        // LNOT
        inline DERIVED_MASK_TYPE lnot () const {
            return EMULATED_FUNCTIONS::logicalNot<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this));
        }
        
        inline DERIVED_MASK_TYPE operator!() const {
            return lnot();
        }

        // LNOTA
        inline DERIVED_MASK_TYPE & lnota () {
            return EMULATED_FUNCTIONS::logicalNotAssign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this));
        }

        // TOINT
        //inline MaskAsInt<MASK_LEN> toInt() {
        //    return 0;
        //}

        // FROMINT
        //inline DERIVED_MASK_TYPE & fromInt(MaskAsInt<MASK_LEN>) {
        //    return *this;
        //}
        // HLAND
        inline bool hland() const {
            return EMULATED_FUNCTIONS::reduceLogicalAnd<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this));
        }

        // HLOR
        inline bool hlor() const {
            return EMULATED_FUNCTIONS::reduceLogicalOr<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this));
        }

        //HLXOR
        inline bool hlxor() const {
            return EMULATED_FUNCTIONS::reduceLogicalXor<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this));
        }
    };

    
    // **********************************************************************
    // *
    // *  Declaration of IntermediateMask class 
    // *
    // *    This class is a helper class used in masked version of
    // *    operator[]. This object is not copyable and can only be created
    // *    from its vector type (VEC_TYPE) for temporary use. 
    // *
    // **********************************************************************
    template<class VEC_TYPE, class MASK_TYPE>
    class IntermediateMask {
    public:
        // MASSIGN
        inline void operator=(VEC_TYPE const & vecRhs) const {
            mVecRef.assign(mMaskRef, vecRhs);
        }

        // MADDVA
        inline void operator+=(VEC_TYPE const & vecRhs) const {
            mVecRef.adda(mMaskRef, vecRhs);
        }
        
        // MSUBVA
        inline void operator-= (VEC_TYPE const & vecRhs) const {
            mVecRef.suba(mMaskRef, vecRhs);
        }

        // MMULVA
        inline void operator*= (VEC_TYPE const & vecRhs) const {
            mVecRef.mula(mMaskRef, vecRhs);
        }

        // MDIVVA
        inline void operator/= (VEC_TYPE const & vecRhs) const {
            mVecRef.diva(mMaskRef, vecRhs);
        }

        // MBANDVA
        inline void operator&= (VEC_TYPE const & vecRhs) const {
            mVecRef.banda(mMaskRef, vecRhs);
        }

        // MBORVA
        inline void operator|= (VEC_TYPE const & vecRhs) const {
            mVecRef.bora(mMaskRef, vecRhs);
        }

        // MBXORVA
        inline void operator^= (VEC_TYPE const & vecRhs) const {
            mVecRef.bxora(mMaskRef, vecRhs);
        }

        // This object should be only constructible by the
        // vector type using it.
        IntermediateMask();
        IntermediateMask(IntermediateMask const &);
        IntermediateMask & operator= (IntermediateMask const &); 

        IntermediateMask(uint32_t);
    private:
        friend VEC_TYPE;

        inline explicit IntermediateMask(MASK_TYPE const & mask, VEC_TYPE & vec) : mMaskRef(mask), mVecRef(vec) {}

        MASK_TYPE const & mMaskRef;
        VEC_TYPE & mVecRef;
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
             typename MASK_TYPE,
             typename SWIZZLE_MASK_TYPE>
    class SIMDVecBaseInterface
    {
        // Other vector types necessary for this class
        typedef SIMDVecBaseInterface< 
            DERIVED_VEC_TYPE, 
            SCALAR_TYPE, 
            VEC_LEN, 
            MASK_TYPE,
            SWIZZLE_MASK_TYPE> VEC_TYPE;

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
        constexpr static uint32_t length() { return VEC_LEN; }

        constexpr static uint32_t alignment() { return VEC_LEN*sizeof(SCALAR_TYPE); }
        
        // ZERO-VEC
        static DERIVED_VEC_TYPE zero() { return DERIVED_VEC_TYPE(SCALAR_TYPE(0)); }

        // ONE-VEC
        static DERIVED_VEC_TYPE one() { return DERIVED_VEC_TYPE(SCALAR_TYPE(1)); }

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

        // PREFETCH0
        static inline void prefetch0 (SCALAR_TYPE const *p) {
            // DO NOTHING!
        }

        // PREFETCH1
        static inline void prefetch1 (SCALAR_TYPE const *p) {
            // DO NOTHING!
        }

        // PREFETCH2
        static inline void prefetch2 (SCALAR_TYPE const *p) {
            // DO NOTHING!
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
        
        // BLENDV
        inline DERIVED_VEC_TYPE blend (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::blend<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BLENDS
        inline DERIVED_VEC_TYPE blend (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::blend<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SWIZZLE
        DERIVED_VEC_TYPE swizzle (SWIZZLE_MASK_TYPE const & sMask) const {
            return EMULATED_FUNCTIONS::swizzle<DERIVED_VEC_TYPE, SWIZZLE_MASK_TYPE> (sMask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SWIZZLEA
        DERIVED_VEC_TYPE swizzlea (SWIZZLE_MASK_TYPE const & sMask) {
            return EMULATED_FUNCTIONS::swizzleAssign<DERIVED_VEC_TYPE, SWIZZLE_MASK_TYPE> (sMask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // ADDV
        inline DERIVED_VEC_TYPE add (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::add<DERIVED_VEC_TYPE> ( static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        inline DERIVED_VEC_TYPE operator+ (DERIVED_VEC_TYPE const & b) const {
            return add(b);
        }
        
        // MADDV
        inline DERIVED_VEC_TYPE add (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::add<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        // ADDS
        inline DERIVED_VEC_TYPE add (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::addScalar<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MADDS
        inline DERIVED_VEC_TYPE add(MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::addScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ADDVA
        inline DERIVED_VEC_TYPE & adda (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::addAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        inline DERIVED_VEC_TYPE & operator+= (DERIVED_VEC_TYPE const & b) {
            return this->adda(b);
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
        inline DERIVED_VEC_TYPE sadd(DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::addSaturated<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        } 

        // MSADDV
        inline DERIVED_VEC_TYPE sadd(MASK_TYPE const & mask, DERIVED_VEC_TYPE b) const {
            return EMULATED_FUNCTIONS::addSaturated<DERIVED_VEC_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SADDS
        inline DERIVED_VEC_TYPE sadd(SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::addSaturatedScalar<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MSADDS
        inline DERIVED_VEC_TYPE sadd(MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::addSaturatedScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SADDVA
        inline DERIVED_VEC_TYPE & sadda(DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::addSaturatedAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MSADDVA
        inline DERIVED_VEC_TYPE & sadda(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::addSaturatedAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // SADDSA
        inline DERIVED_VEC_TYPE & sadda(SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::addSaturatedScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MSADDSA
        inline DERIVED_VEC_TYPE & sadda(MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::addSaturatedScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        // POSTINC
        inline DERIVED_VEC_TYPE postinc () {
            return EMULATED_FUNCTIONS::postfixIncrement<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }
        
        inline DERIVED_VEC_TYPE operator++ (int) {
            return postinc();
        }

        // MPOSTINC
        inline DERIVED_VEC_TYPE postinc (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::postfixIncrement<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // PREFINC
        inline DERIVED_VEC_TYPE & prefinc () {
            return EMULATED_FUNCTIONS::prefixIncrement<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }
        
        inline DERIVED_VEC_TYPE & operator++ () {
            return prefinc();
        }

        // MPREFINC
        inline DERIVED_VEC_TYPE & prefinc (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::prefixIncrement<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // SUBV
        inline DERIVED_VEC_TYPE sub (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::sub<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MSUBV
        inline DERIVED_VEC_TYPE sub (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::sub<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SUBS
        inline DERIVED_VEC_TYPE sub (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::subScalar<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MSUBS
        inline DERIVED_VEC_TYPE sub (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::subScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SUBVA
        inline DERIVED_VEC_TYPE & suba (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::subAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        inline DERIVED_VEC_TYPE & operator-= (DERIVED_VEC_TYPE const & b) {
            return suba(b);
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
        inline DERIVED_VEC_TYPE ssub (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::subSaturated<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MSSUBV
        inline DERIVED_VEC_TYPE ssub (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::subSaturated<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SSUBS
        inline DERIVED_VEC_TYPE ssub (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::subSaturated<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MSSUBS
        inline DERIVED_VEC_TYPE ssub (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::subSaturated<DERIVED_VEC_TYPE, SCALAR_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SSUBVA
        inline DERIVED_VEC_TYPE & ssuba (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::subSaturatedAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MSSUBVA
        inline DERIVED_VEC_TYPE & ssuba (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::subSaturatedAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // SSUBSA
        inline DERIVED_VEC_TYPE & ssuba (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::subSaturatedScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MSSUBSA
        inline DERIVED_VEC_TYPE & ssuba (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::subSaturatedScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // SUBFROMV
        inline DERIVED_VEC_TYPE subfrom (DERIVED_VEC_TYPE const & a) const {
            return EMULATED_FUNCTIONS::subFrom<DERIVED_VEC_TYPE>(a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MSUBFROMV
        inline DERIVED_VEC_TYPE subfrom (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & a) const {
            return EMULATED_FUNCTIONS::subFrom<DERIVED_VEC_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SUBFROMS
        inline DERIVED_VEC_TYPE subfrom (SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::subFromScalar<DERIVED_VEC_TYPE, SCALAR_TYPE>(a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MSUBFROMS
        inline DERIVED_VEC_TYPE subfrom (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::subFromScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SUBFROMVA
        inline DERIVED_VEC_TYPE & subfroma (DERIVED_VEC_TYPE const & a) {
            return EMULATED_FUNCTIONS::subFromAssign<DERIVED_VEC_TYPE>(a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MSUBFROMVA
        inline DERIVED_VEC_TYPE & subfroma (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & a) {
            return EMULATED_FUNCTIONS::subFromAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // SUBFROMSA
        inline DERIVED_VEC_TYPE & subfroma (SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::subFromScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MSUBFROMSA
        inline DERIVED_VEC_TYPE & subfroma (MASK_TYPE const & mask, SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::subFromScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // POSTDEC
        inline DERIVED_VEC_TYPE postdec () {
            return EMULATED_FUNCTIONS::postfixDecrement<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        inline DERIVED_VEC_TYPE operator-- (int) {
            return postdec();
        }

        // MPOSTDEC
        inline DERIVED_VEC_TYPE postdec (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::postfixDecrement<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // PREFDEC
        inline DERIVED_VEC_TYPE & prefdec() {
            return EMULATED_FUNCTIONS::prefixDecrement<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }
        
        inline DERIVED_VEC_TYPE & operator-- () {
            return prefdec();
        }

        // MPREFDEC
        inline DERIVED_VEC_TYPE & prefdec (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::prefixDecrement<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MULV
        inline DERIVED_VEC_TYPE mul (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::mult<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        inline DERIVED_VEC_TYPE operator* (DERIVED_VEC_TYPE const & b) const {
            return mul(b);
        }

        // MMULV
        inline DERIVED_VEC_TYPE mul (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::mult<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MULS
        inline DERIVED_VEC_TYPE mul (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::mult<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMULS
        inline DERIVED_VEC_TYPE mul (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::mult<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MULVA
        inline DERIVED_VEC_TYPE & mula (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::multAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        inline DERIVED_VEC_TYPE & operator*= (DERIVED_VEC_TYPE const & b) {
            return mula(b);
        }

        // MMULVA
        inline DERIVED_VEC_TYPE & mula (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::multAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MULSA
        inline DERIVED_VEC_TYPE & mula (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::multAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MMULSA
        inline DERIVED_VEC_TYPE & mula (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::multAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // DIVV
        inline DERIVED_VEC_TYPE div (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::div<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        inline DERIVED_VEC_TYPE operator/ (DERIVED_VEC_TYPE const & b) const {
            return div(b);
        }

        // MDIVV
        inline DERIVED_VEC_TYPE div (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::div<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // DIVS
        inline DERIVED_VEC_TYPE div (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::div<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MDIVS
        inline DERIVED_VEC_TYPE div (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::div<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
          
        // DIVVA
        inline DERIVED_VEC_TYPE diva (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::divAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        inline DERIVED_VEC_TYPE operator/= (DERIVED_VEC_TYPE const & b) {
            return diva(b);
        }

        // MDIVVA
        inline DERIVED_VEC_TYPE diva (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::divAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
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
        inline DERIVED_VEC_TYPE rcp () const {
            return EMULATED_FUNCTIONS::rcp<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MRCP
        inline DERIVED_VEC_TYPE rcp (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::rcp<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // RCPS
        inline DERIVED_VEC_TYPE rcp (SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::rcpScalar<DERIVED_VEC_TYPE, SCALAR_TYPE> (a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MRCPS
        inline DERIVED_VEC_TYPE rcp (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::rcpScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // RCPA
        inline DERIVED_VEC_TYPE & rcpa () {
            return EMULATED_FUNCTIONS::rcpAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MRCPA
        inline DERIVED_VEC_TYPE & rcpa (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::rcpAssign<DERIVED_VEC_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }
        
        // RCPSA
        inline DERIVED_VEC_TYPE & rcpa (SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::rcpScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MRCPSA
        inline DERIVED_VEC_TYPE & rcpa (MASK_TYPE const & mask, SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::rcpScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }
        
        // CMPEQV
        inline MASK_TYPE cmpeq (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::isEqual<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        inline MASK_TYPE operator== (DERIVED_VEC_TYPE const & b) const {
            return cmpeq(b);
        }

        // CMPEQS
        inline MASK_TYPE cmpeq (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::isEqual<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // CMPNEV
        inline MASK_TYPE cmpne (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::isNotEqual<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        inline MASK_TYPE operator!= (DERIVED_VEC_TYPE const & b) const {
            return cmpne(b);
        }

        // CMPNES
        inline MASK_TYPE cmpne (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::isNotEqual<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // CMPGTV
        inline MASK_TYPE cmpgt (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::isGreater<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        inline MASK_TYPE operator> (DERIVED_VEC_TYPE const & b) const {
            return cmpgt(b);
        }

        // CMPGTS
        inline MASK_TYPE cmpgt (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::isGreater<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // CMPLTV
        inline MASK_TYPE cmplt (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::isLesser<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        inline MASK_TYPE operator< (DERIVED_VEC_TYPE const & b) const {
            return cmplt(b);
        }

        // CMPLTS
        inline MASK_TYPE cmplt (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::isLesser<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // CMPGEV
        inline MASK_TYPE cmpge (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::isGreaterEqual<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        inline MASK_TYPE operator>= (DERIVED_VEC_TYPE const & b) const {
            return cmpge(b);
        }

        // CMPGES
        inline MASK_TYPE cmpge (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::isGreaterEqual<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        // CMPLEV
        inline MASK_TYPE cmple (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::isLesserEqual<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        inline MASK_TYPE operator<= (DERIVED_VEC_TYPE const & b) const {
            return cmple(b);
        }

        // CMPLES
        inline MASK_TYPE cmple (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::isLesserEqual<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // CMPEV
        inline bool cmpe (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::isExact<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // CMPES
        inline bool cmpe (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::isExact<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), DERIVED_VEC_TYPE(b));
        }

        // UNIQUE
        inline bool unique() const {
            return EMULATED_FUNCTIONS::unique<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HADD
        inline SCALAR_TYPE hadd () const {
            return EMULATED_FUNCTIONS::reduceAdd<SCALAR_TYPE, DERIVED_VEC_TYPE>( static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
    
        // MHADD
        inline SCALAR_TYPE hadd (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::reduceAdd<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &> (*this));
        }
        
        // HADDS
        inline SCALAR_TYPE hadd (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::reduceAdd<SCALAR_TYPE, DERIVED_VEC_TYPE>(b, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHADDS
        inline SCALAR_TYPE hadd (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::reduceAdd<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, b, static_cast<DERIVED_VEC_TYPE const &> (*this));
        }

        // HMUL
        inline SCALAR_TYPE hmul () const {
            return EMULATED_FUNCTIONS::reduceMult<SCALAR_TYPE, DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHMUL
        inline SCALAR_TYPE hmul (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::reduceMult<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HMULS
        inline SCALAR_TYPE hmul (SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::reduceMultScalar<SCALAR_TYPE, DERIVED_VEC_TYPE>(a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHMULS
        inline SCALAR_TYPE hmul (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::reduceMultScalar<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // ******************************************************************
        // * Fused arithmetics
        // ******************************************************************

        // FMULADDV
        inline DERIVED_VEC_TYPE fmuladd(DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            return EMULATED_FUNCTIONS::MATH::fmuladd<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // MFMULADDV
        inline DERIVED_VEC_TYPE fmuladd(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            return EMULATED_FUNCTIONS::MATH::fmuladd<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // FMULSUBV
        inline DERIVED_VEC_TYPE fmulsub(DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            return EMULATED_FUNCTIONS::MATH::fmulsub<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // MFMULSUBV
        inline DERIVED_VEC_TYPE fmulsub(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            return EMULATED_FUNCTIONS::MATH::fmulsub<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // FADDMULV
        inline DERIVED_VEC_TYPE faddmul(DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            return EMULATED_FUNCTIONS::MATH::faddmul<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // MFADDMULV
        inline DERIVED_VEC_TYPE faddmul(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            return EMULATED_FUNCTIONS::MATH::faddmul<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }
        
        // FSUBMULV
        inline DERIVED_VEC_TYPE fsubmul(DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            return EMULATED_FUNCTIONS::MATH::fsubmul<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // MFSUBMULV
        inline DERIVED_VEC_TYPE fsubmul(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            return EMULATED_FUNCTIONS::MATH::fsubmul<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // ******************************************************************
        // * Additional math functions
        // ******************************************************************

        // MAXV
        inline DERIVED_VEC_TYPE max (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::MATH::max<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMAXV
        inline DERIVED_VEC_TYPE max (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::MATH::max<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MAXS
        inline DERIVED_VEC_TYPE max (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::MATH::maxScalar<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMAXS
        inline DERIVED_VEC_TYPE max (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::MATH::maxScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MAXVA
        inline DERIVED_VEC_TYPE & maxa (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::MATH::maxAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MMAXVA
        inline DERIVED_VEC_TYPE & maxa (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::MATH::maxAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MAXSA
        inline DERIVED_VEC_TYPE & maxa (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::MATH::maxScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MMAXSA
        inline DERIVED_VEC_TYPE & maxa (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::MATH::maxScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        // MINV
        inline DERIVED_VEC_TYPE min (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::MATH::min<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMINV
        inline DERIVED_VEC_TYPE min (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::MATH::min<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MINS
        inline DERIVED_VEC_TYPE min (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::MATH::minScalar<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMINS
        inline DERIVED_VEC_TYPE min (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::MATH::minScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        // MINVA
        inline DERIVED_VEC_TYPE & mina (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::MATH::minAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MMINVA
        inline DERIVED_VEC_TYPE & mina (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::MATH::minAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        // MINSA
        inline DERIVED_VEC_TYPE & mina (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::MATH::minScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MMINSA
        inline DERIVED_VEC_TYPE & mina (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::MATH::minScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // HMAX
        inline SCALAR_TYPE hmax () const {
            return EMULATED_FUNCTIONS::MATH::reduceMax<SCALAR_TYPE, DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHMAX
        inline SCALAR_TYPE hmax (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::reduceMax<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // IMAX
        inline uint32_t imax() const {
            return EMULATED_FUNCTIONS::MATH::indexMax<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MIMAX
        inline uint32_t imax(MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::indexMax<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HMIN
        inline SCALAR_TYPE hmin() const {
            return EMULATED_FUNCTIONS::MATH::reduceMin<SCALAR_TYPE, DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHMIN
        inline SCALAR_TYPE hmin(MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::reduceMin<SCALAR_TYPE, DERIVED_VEC_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // IMIN
        inline uint32_t imin() const {
            return EMULATED_FUNCTIONS::MATH::indexMin<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MIMIN
        inline uint32_t imin(MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::indexMin<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
    };
    
    // ***************************************************************************
    // *
    // *    Definition of Bitwise Interface. Bitwise operations can only be
    // *    performed on integer (signed and unsigned) data types in C++.
    // *    While making bitwise operations on floating points is sometimes
    // *    necessary, it is not safe and not portable. 
    // *
    // ***************************************************************************
    template<typename DERIVED_VEC_TYPE,
             typename SCALAR_TYPE,
             typename MASK_TYPE>
    class SIMDVecBitwiseInterface {
        
        typedef SIMDVecBitwiseInterface< 
            DERIVED_VEC_TYPE, 
            SCALAR_TYPE,
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
 
    public:
        // BANDV
        inline DERIVED_VEC_TYPE band (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::binaryAnd<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        inline DERIVED_VEC_TYPE operator& (DERIVED_VEC_TYPE const & b) const {
            return band(b);
        }

        // MBANDV
        inline DERIVED_VEC_TYPE band (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::binaryAnd<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BANDS
        inline DERIVED_VEC_TYPE band (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::binaryAnd<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MBANDS
        inline DERIVED_VEC_TYPE band (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::binaryAnd<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BANDVA
        inline DERIVED_VEC_TYPE & banda (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryAndAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        inline DERIVED_VEC_TYPE & operator&= (DERIVED_VEC_TYPE const & b) {
            return banda(b);
        }

        // MBANDVA
        inline DERIVED_VEC_TYPE & banda (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryAndAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        // BANDSA
        inline DERIVED_VEC_TYPE & banda (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryAndAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MBANDSA
        inline DERIVED_VEC_TYPE & banda (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryAndAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // BORV
        inline DERIVED_VEC_TYPE bor ( DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        inline DERIVED_VEC_TYPE operator| ( DERIVED_VEC_TYPE const & b) const {
            return bor(b);
        }

        // MBORV
        inline DERIVED_VEC_TYPE bor ( MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BORS
        inline DERIVED_VEC_TYPE bor (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MBORS
        inline DERIVED_VEC_TYPE bor (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BORVA
        inline DERIVED_VEC_TYPE & bora (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryOrAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        inline DERIVED_VEC_TYPE & operator|= (DERIVED_VEC_TYPE const & b) {
            return bora(b);
        }

        // MBORVA
        inline DERIVED_VEC_TYPE & bora (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryOrAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // BORSA
        inline DERIVED_VEC_TYPE & bora (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryOrAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MBORSA
        inline DERIVED_VEC_TYPE & bora (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryOrAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // BXORV
        inline DERIVED_VEC_TYPE bxor (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::binaryXor<DERIVED_VEC_TYPE> ( static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        inline DERIVED_VEC_TYPE operator^ (DERIVED_VEC_TYPE const & b) const {
            return bxor(b);
        }

        // MBXORV
        inline DERIVED_VEC_TYPE bxor (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::binaryXor<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BXORS
        inline DERIVED_VEC_TYPE bxor (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::binaryXor<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MBXORS
        inline DERIVED_VEC_TYPE bxor (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::binaryXor<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BXORVA
        inline DERIVED_VEC_TYPE & bxora (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryXorAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        inline DERIVED_VEC_TYPE & operator^= (DERIVED_VEC_TYPE const & b) {
            return bxora(b);
        }

        // MBXORVA
        inline DERIVED_VEC_TYPE & bxora (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryXorAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // BXORSA
        inline DERIVED_VEC_TYPE & bxora (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryXorAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MBXORSA
        inline DERIVED_VEC_TYPE & bxora (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryXorAssign<DERIVED_VEC_TYPE,SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // BNOT
        inline DERIVED_VEC_TYPE bnot () const {
            return EMULATED_FUNCTIONS::binaryNot<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
    
        inline DERIVED_VEC_TYPE operator~ () const {
            return bnot();
        }

        // MBNOT
        inline DERIVED_VEC_TYPE bnot (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::binaryNot<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // BNOTA
        inline DERIVED_VEC_TYPE & bnota () {
            return EMULATED_FUNCTIONS::binaryNotAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MBNOTA
        inline DERIVED_VEC_TYPE & bnota (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::binaryNotAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // HBAND
        inline SCALAR_TYPE hband ()const  {
            return EMULATED_FUNCTIONS::reduceBinaryAnd<SCALAR_TYPE, DERIVED_VEC_TYPE>( static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBAND
        inline SCALAR_TYPE hband (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::reduceBinaryAnd<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HBANDS
        inline SCALAR_TYPE hband (SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::reduceBinaryAndScalar<SCALAR_TYPE, DERIVED_VEC_TYPE>(a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBANDS
        inline SCALAR_TYPE hband (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::reduceBinaryAndScalar<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HBOR
        inline SCALAR_TYPE hbor () const {
            return EMULATED_FUNCTIONS::reduceBinaryOr<SCALAR_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBOR
        inline SCALAR_TYPE hbor (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::reduceBinaryOr<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HBORS
        inline SCALAR_TYPE hbor (SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::reduceBinaryOrScalar<SCALAR_TYPE, DERIVED_VEC_TYPE> (a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBORS
        inline SCALAR_TYPE hbor (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::reduceBinaryOrScalar<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // HBXOR
        inline SCALAR_TYPE hbxor () const {
            return EMULATED_FUNCTIONS::reduceBinaryXor<SCALAR_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBXOR
        inline SCALAR_TYPE hbxor (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::reduceBinaryXor<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HBXORS
        inline SCALAR_TYPE hbxor (SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::reduceBinaryXorScalar<SCALAR_TYPE, DERIVED_VEC_TYPE> (a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBXORS
        inline SCALAR_TYPE hbxor (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::reduceBinaryXorScalar<SCALAR_TYPE, DERIVED_VEC_TYPE> (mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
    };
    
    // ***************************************************************************
    // *
    // *    Definition of Gather/Scatter interface. This interface creates
    // *    an abstraction for gather and scatter operations. It needs to be
    // *    separate from base interface, because it is aware of unsigned
    // *    types (used for indexing).
    // *
    // ***************************************************************************
    template<typename DERIVED_VEC_TYPE,
             typename DERIVED_UINT_VEC_TYPE,
             typename SCALAR_TYPE,
             typename MASK_TYPE>
    class SIMDVecGatherScatterInterface
    {
        typedef SIMDVecGatherScatterInterface< 
            DERIVED_VEC_TYPE, 
            DERIVED_UINT_VEC_TYPE,
            SCALAR_TYPE,
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
        inline DERIVED_VEC_TYPE gather (SCALAR_TYPE * baseAddr, DERIVED_UINT_VEC_TYPE const & indices) {
            return EMULATED_FUNCTIONS::gather<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }
        
        // MGATHERV
        inline DERIVED_VEC_TYPE gather (MASK_TYPE const & mask, SCALAR_TYPE* baseAddr, DERIVED_UINT_VEC_TYPE const & indices) {
            return EMULATED_FUNCTIONS::gather<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
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
        inline SCALAR_TYPE*  scatter (SCALAR_TYPE* baseAddr, DERIVED_UINT_VEC_TYPE const & indices) {
            return EMULATED_FUNCTIONS::scatter<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }
        
        // MSCATTERV
        inline SCALAR_TYPE*  scatter (MASK_TYPE const & mask, SCALAR_TYPE* baseAddr, DERIVED_UINT_VEC_TYPE const & indices) {
            return EMULATED_FUNCTIONS::scatter<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }       
    };
    
    // ***************************************************************************
    // *
    // *    Definition of Shift/Rotate interface. This interface creates
    // *    an abstraction for bitwise shift and rotation operations. These
    // *    operations should only be used on signed and unsigned integer
    // *    vector types.
    // *
    // ***************************************************************************
    template<typename DERIVED_VEC_TYPE,
             typename DERIVED_UINT_VEC_TYPE,
             typename SCALAR_TYPE,
             typename SCALAR_UINT_TYPE,
             typename MASK_TYPE>
    class SIMDVecShiftRotateInterface
    {
        typedef SIMDVecShiftRotateInterface< 
            DERIVED_VEC_TYPE, 
            DERIVED_UINT_VEC_TYPE,
            SCALAR_TYPE,
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
 
    public:
        // LSHV
        inline DERIVED_VEC_TYPE lsh (DERIVED_UINT_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::shiftBitsLeft<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MLSHV
        inline DERIVED_VEC_TYPE lsh (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::shiftBitsLeft<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        // LSHS
        inline DERIVED_VEC_TYPE lsh (SCALAR_UINT_TYPE b) const {
            return EMULATED_FUNCTIONS::shiftBitsLeftScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MLSHS
        inline DERIVED_VEC_TYPE lsh (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) const {
            return EMULATED_FUNCTIONS::shiftBitsLeftScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // LSHVA
        inline DERIVED_VEC_TYPE & lsha (DERIVED_UINT_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::shiftBitsLeftAssign<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        // MLSHVA
        inline DERIVED_VEC_TYPE & lsha (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::shiftBitsLeftAssign<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
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
        inline DERIVED_VEC_TYPE rsh (DERIVED_UINT_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::shiftBitsRight<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MRSHV
        inline DERIVED_VEC_TYPE rsh (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::shiftBitsRight<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        // RSHS
        inline DERIVED_VEC_TYPE rsh (SCALAR_UINT_TYPE b) const {
            return EMULATED_FUNCTIONS::shiftBitsRightScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MRSHS
        inline DERIVED_VEC_TYPE rsh (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) const {
            return EMULATED_FUNCTIONS::shiftBitsRightScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // RSHVA
        inline DERIVED_VEC_TYPE & rsha (DERIVED_UINT_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::shiftBitsRightAssign<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        // MRSHVA
        inline DERIVED_VEC_TYPE & rsha (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::shiftBitsRightAssign<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
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
        inline DERIVED_VEC_TYPE rol (DERIVED_UINT_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::rotateBitsLeft<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MROLV
        inline DERIVED_VEC_TYPE rol (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::rotateBitsLeft<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ROLS
        inline DERIVED_VEC_TYPE rol (SCALAR_UINT_TYPE b) const {
            return EMULATED_FUNCTIONS::rotateBitsLeftScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        // MROLS
        inline DERIVED_VEC_TYPE rol (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) const {
            return EMULATED_FUNCTIONS::rotateBitsLeftScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ROLVA
        inline DERIVED_VEC_TYPE & rola (DERIVED_UINT_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::rotateBitsLeftAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MROLVA
        inline DERIVED_VEC_TYPE & rola (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::rotateBitsLeftAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
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
        inline DERIVED_VEC_TYPE ror (DERIVED_UINT_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::rotateBitsRight<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MRORV
        inline DERIVED_VEC_TYPE ror (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::rotateBitsRight<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // RORS
        inline DERIVED_VEC_TYPE ror (SCALAR_UINT_TYPE b) const {
            return EMULATED_FUNCTIONS::rotateBitsRightScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MRORS
        inline DERIVED_VEC_TYPE ror (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) const {
            return EMULATED_FUNCTIONS::rotateBitsRightScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // RORVA
        inline DERIVED_VEC_TYPE rora (DERIVED_UINT_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::rotateBitsRightAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MRORVA
        inline DERIVED_VEC_TYPE rora (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::rotateBitsRightAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
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
    // *    Definition of Sign interface. This interface creates
    // *    an abstraction for operations that are aware of scalar types sign.
    // *    this interface should be reserved for signed integer and floating
    // *    point vector types.
    // *
    // ***************************************************************************
    template<typename DERIVED_VEC_TYPE, typename MASK_TYPE>
    class SIMDVecSignInterface
    {        
        // Other vector types necessary for this class
        typedef SIMDVecSignInterface< 
            DERIVED_VEC_TYPE,
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
 
    public:

        // NEG
        inline DERIVED_VEC_TYPE neg () const {
            return EMULATED_FUNCTIONS::unaryMinus<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MNEG
        inline DERIVED_VEC_TYPE neg (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::unaryMinus<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // NEGA
        inline DERIVED_VEC_TYPE & nega() {
            return EMULATED_FUNCTIONS::unaryMinusAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MNEGA
        inline DERIVED_VEC_TYPE & nega(MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::unaryMinusAssign<DERIVED_VEC_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // ABS
        inline DERIVED_VEC_TYPE abs () const {
            return EMULATED_FUNCTIONS::MATH::abs<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MABS
        inline DERIVED_VEC_TYPE abs (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::abs<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ABSA
        inline DERIVED_VEC_TYPE absa () {
            return EMULATED_FUNCTIONS::MATH::absAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MABSA
        inline DERIVED_VEC_TYPE absa (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::MATH::absAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }
    };

    // ***************************************************************************
    // *
    // *    Definition of Packable Interface. Pack operations can only be 
    // *    performed on SIMD vector with lengths higher than 1 and being
    // *    powers of 2. Vectors of such lengths have to derive from one of type
    // *    interfaces: signed, unsigned or float and from packable interface.
    // *    SIMD vectors of length 1 should only use type interface.
    // *
    // ***************************************************************************
    template<class DERIVED_VEC_TYPE,
             class DERIVED_HALF_VEC_TYPE>
    class SIMDVecPackableInterface
    {        
        // Other vector types necessary for this class
        typedef SIMDVecPackableInterface< 
            DERIVED_VEC_TYPE, 
            DERIVED_HALF_VEC_TYPE> VEC_TYPE;

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
 
    public:

        // PACK
        DERIVED_VEC_TYPE & pack(DERIVED_HALF_VEC_TYPE const & a, DERIVED_HALF_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::pack<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE> (
                    static_cast<DERIVED_VEC_TYPE &>(*this), 
                    static_cast<DERIVED_HALF_VEC_TYPE const &>(a),
                    static_cast<DERIVED_HALF_VEC_TYPE const &>(b)
                );
        }
        
        // PACKLO
        DERIVED_VEC_TYPE & packlo(DERIVED_HALF_VEC_TYPE const & a) {
            return EMULATED_FUNCTIONS::packLow<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE> (
                    static_cast<DERIVED_VEC_TYPE &>(*this), 
                    static_cast<DERIVED_HALF_VEC_TYPE const &>(a)
                );
        }

        // PACKHI
        DERIVED_VEC_TYPE & packhi(DERIVED_HALF_VEC_TYPE const & a) {
            return EMULATED_FUNCTIONS::packHigh<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE> (
                    static_cast<DERIVED_VEC_TYPE &>(*this), 
                    static_cast<DERIVED_HALF_VEC_TYPE const &>(a)
                );
        }
        
        // UNPACK
        void unpack(DERIVED_HALF_VEC_TYPE & a, DERIVED_HALF_VEC_TYPE & b) const {
            EMULATED_FUNCTIONS::unpack<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE> (
                    static_cast<DERIVED_VEC_TYPE const &>(*this), 
                    static_cast<DERIVED_HALF_VEC_TYPE &>(a),
                    static_cast<DERIVED_HALF_VEC_TYPE &>(b)
                );
        }

        // UNPACKLO
        DERIVED_HALF_VEC_TYPE unpacklo() const {
            return EMULATED_FUNCTIONS::unpackLow<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE> (
                        static_cast<DERIVED_VEC_TYPE const &> (*this)
                    );
        }

        // UNPACKHI
        DERIVED_HALF_VEC_TYPE unpackhi() const {
            return EMULATED_FUNCTIONS::unpackHigh<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE> (
                        static_cast<DERIVED_VEC_TYPE const &> (*this)
                    );
        }
    };
    
    // ***************************************************************************
    // *
    // *    Definition of interface for vectors using UNSIGNED INTEGER scalar types
    // *
    // ***************************************************************************
    template<typename DERIVED_UINT_VEC_TYPE,
             typename SCALAR_UINT_TYPE, 
             uint32_t VEC_LEN,
             typename MASK_TYPE,
             typename SWIZZLE_MASK_TYPE> 
    class SIMDVecUnsignedInterface : 
        public SIMDVecBaseInterface< 
            DERIVED_UINT_VEC_TYPE,
            SCALAR_UINT_TYPE, 
            VEC_LEN,
            MASK_TYPE,
            SWIZZLE_MASK_TYPE>,
        public SIMDVecBitwiseInterface<
            DERIVED_UINT_VEC_TYPE,
            SCALAR_UINT_TYPE,
            MASK_TYPE>,
        public SIMDVecGatherScatterInterface<
            DERIVED_UINT_VEC_TYPE,   // DERIVED_VEC_TYPE
            DERIVED_UINT_VEC_TYPE,
            SCALAR_UINT_TYPE,
            MASK_TYPE>,
        public SIMDVecShiftRotateInterface<
            DERIVED_UINT_VEC_TYPE,   // DERIVED_VEC_TYPE
            DERIVED_UINT_VEC_TYPE,
            SCALAR_UINT_TYPE,        // SCALAR_TYPE
            SCALAR_UINT_TYPE,
            MASK_TYPE>
    {
        // Other vector types necessary for this class
        typedef SIMDVecUnsignedInterface< 
            DERIVED_UINT_VEC_TYPE, 
            SCALAR_UINT_TYPE,
            VEC_LEN, 
            MASK_TYPE,
            SWIZZLE_MASK_TYPE> VEC_TYPE;
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
 
        SCALAR_UINT_TYPE operator[] (SCALAR_UINT_TYPE index) const; // Declaration only! This operator has to be implemented in derived class.
        inline DERIVED_UINT_VEC_TYPE & insert(uint32_t index, SCALAR_UINT_TYPE value); // Declaration only! This operator has to be implemented in derived class.

    protected:
            
        // Making destructor protected prohibits this class from being instantiated. Effectively this class can only be used as a base class.
        ~SIMDVecUnsignedInterface() {};
    public:
        // Everything already handled by other interface classes

        // SUBV
        inline DERIVED_UINT_VEC_TYPE operator- (DERIVED_UINT_VEC_TYPE const & b) const {
            return this->sub(b);
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
             typename MASK_TYPE,
             typename SWIZZLE_MASK_TYPE>
    class SIMDVecSignedInterface : 
        public SIMDVecBaseInterface< 
            DERIVED_VEC_TYPE,
            SCALAR_TYPE, 
            VEC_LEN,
            MASK_TYPE,
            SWIZZLE_MASK_TYPE>,
        public SIMDVecBitwiseInterface<
            DERIVED_VEC_TYPE,
            SCALAR_TYPE,
            MASK_TYPE>,
        public SIMDVecGatherScatterInterface<
            DERIVED_VEC_TYPE,   // DERIVED_VEC_TYPE
            DERIVED_VEC_UINT_TYPE,   // DERIVEC_UINT_VEC_TYPE // TODO: replace this with DERIVED_VEC_TYPE when other types independant!
            SCALAR_TYPE,
            MASK_TYPE>,
        public SIMDVecShiftRotateInterface<
            DERIVED_VEC_TYPE,
            DERIVED_VEC_UINT_TYPE,
            SCALAR_TYPE,
            SCALAR_UINT_TYPE,
            MASK_TYPE>,
        public SIMDVecSignInterface<
            DERIVED_VEC_TYPE,
            MASK_TYPE>
    {
        // Other vector types necessary for this class
        typedef SIMDVecSignedInterface< DERIVED_VEC_TYPE,
                             DERIVED_VEC_UINT_TYPE,
                             SCALAR_TYPE,
                             VEC_LEN, 
                             SCALAR_UINT_TYPE,
                             MASK_TYPE,
                             SWIZZLE_MASK_TYPE> VEC_TYPE;

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
        // Everything already handled by other interface classes
        
        // SUBV
        inline DERIVED_VEC_TYPE operator- (DERIVED_VEC_TYPE const & b) const {
            return this->sub(b);
        }
        
        // NEG
        inline DERIVED_VEC_TYPE operator- () const {
            return this->neg();
        }
		
		// ITOU
		inline DERIVED_VEC_UINT_TYPE itou () const {
			return EMULATED_FUNCTIONS::xtoy<DERIVED_VEC_UINT_TYPE, DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
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
             typename MASK_TYPE,
             typename SWIZZLE_MASK_TYPE>
    class SIMDVecFloatInterface :  
        public SIMDVecBaseInterface< 
            DERIVED_VEC_TYPE,
            SCALAR_FLOAT_TYPE, 
            VEC_LEN,
            MASK_TYPE,
            SWIZZLE_MASK_TYPE>,
        public SIMDVecGatherScatterInterface<
            DERIVED_VEC_TYPE,   // DERIVED_VEC_TYPE
            DERIVED_VEC_UINT_TYPE,   // DERIVEC_UINT_VEC_TYPE // TODO: replace this with DERIVED_VEC_TYPE when other types independant!
            SCALAR_FLOAT_TYPE,
            MASK_TYPE>,
        public SIMDVecSignInterface<
            DERIVED_VEC_TYPE,
            MASK_TYPE>
    {
        // Other vector types necessary for this class
        typedef SIMDVecFloatInterface< DERIVED_VEC_TYPE,
                    DERIVED_VEC_UINT_TYPE,
                    DERIVED_VEC_INT_TYPE,
                    SCALAR_FLOAT_TYPE,
                    VEC_LEN, 
                    SCALAR_UINT_TYPE,
                    MASK_TYPE,
                    SWIZZLE_MASK_TYPE> VEC_TYPE;
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

        // SUBV
        inline DERIVED_VEC_TYPE operator- (DERIVED_VEC_TYPE const & b) const {
            return this->sub(b);
        }
        
        // NEG
        inline DERIVED_VEC_TYPE operator- () const {
            return this->neg();
        }

        // CMPEQRV
        //inline DERIVED_VEC_TYPE 

        // ********************************************************************
        // * MATH FUNCTIONS
        // ********************************************************************

        // SQR
        inline DERIVED_VEC_TYPE sqr () const {
            return EMULATED_FUNCTIONS::MATH::sqr<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MSQR
        inline DERIVED_VEC_TYPE sqr (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::sqr<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SQRA
        inline DERIVED_VEC_TYPE & sqra () {
            return EMULATED_FUNCTIONS::MATH::sqrAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MSQRA
        inline DERIVED_VEC_TYPE & sqra (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::MATH::sqrAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // SQRT
        inline DERIVED_VEC_TYPE sqrt () const {
            return EMULATED_FUNCTIONS::MATH::sqrt<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // MSQRT
        inline DERIVED_VEC_TYPE sqrt (MASK_TYPE const & mask) const {
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

        // RSQRT
        inline DERIVED_VEC_TYPE rsqrt () const {
            return EMULATED_FUNCTIONS::MATH::rsqrt<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MRSQRT
        inline DERIVED_VEC_TYPE rsqrt (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::rsqrt<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SQRTA
        inline DERIVED_VEC_TYPE & rsqrta () {
            return EMULATED_FUNCTIONS::MATH::rsqrtAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MSQRTA
        inline DERIVED_VEC_TYPE & rsqrta (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::MATH::rsqrtAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }
        
        // POWV
        // Disabled, see Issue #10
        //inline DERIVED_VEC_TYPE pow (DERIVED_VEC_TYPE const & b) const {
        //    return EMULATED_FUNCTIONS::MATH::pow<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        // }

        // MPOWV    
        // Disabled, see Issue #10    
        //inline DERIVED_VEC_TYPE pow (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
        //    return EMULATED_FUNCTIONS::MATH::pow<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        //}

        // POWS
        // Disabled, see Issue #10
        //inline DERIVED_VEC_TYPE pow (SCALAR_FLOAT_TYPE b) const {
        //    return EMULATED_FUNCTIONS::MATH::pows<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        //}

        // MPOWS
        // Disabled, see Issue #10
        //inline DERIVED_VEC_TYPE pow (MASK_TYPE const & mask, SCALAR_FLOAT_TYPE b) const {
        //    return EMULATED_FUNCTIONS::MATH::pows<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        //}

        // ROUND
        inline DERIVED_VEC_TYPE round () const {
            return EMULATED_FUNCTIONS::MATH::round<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // MROUND
        inline DERIVED_VEC_TYPE round (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::round<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // TRUNC
        inline DERIVED_VEC_INT_TYPE trunc () const {
            return EMULATED_FUNCTIONS::MATH::truncToInt<DERIVED_VEC_TYPE, DERIVED_VEC_INT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MTRUNC
        inline DERIVED_VEC_INT_TYPE trunc (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::truncToInt<DERIVED_VEC_TYPE, DERIVED_VEC_INT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // FLOOR
        inline DERIVED_VEC_TYPE floor () const {
            return EMULATED_FUNCTIONS::MATH::floor<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MFLOOR
        inline DERIVED_VEC_TYPE floor (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::floor<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // CEIL
        inline DERIVED_VEC_TYPE ceil () const {
            return EMULATED_FUNCTIONS::MATH::ceil<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MCEIL
        inline DERIVED_VEC_TYPE ceil (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::ceil<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISFIN
        inline MASK_TYPE isfin () const {
            return EMULATED_FUNCTIONS::MATH::isfin<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISINF
        inline MASK_TYPE isinf () const {
            return EMULATED_FUNCTIONS::MATH::isinf<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISAN
        inline MASK_TYPE isan () const {
            return EMULATED_FUNCTIONS::MATH::isan<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISNAN
        inline MASK_TYPE isnan () const {
            return EMULATED_FUNCTIONS::MATH::isnan<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISNORM
        inline MASK_TYPE isnorm() const {
            return EMULATED_FUNCTIONS::MATH::isnorm<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISSUB
        inline MASK_TYPE issub () const {
            return EMULATED_FUNCTIONS::MATH::issub<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISZERO
        inline MASK_TYPE iszero () const {
            return EMULATED_FUNCTIONS::MATH::iszero<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISZEROSUB
        inline MASK_TYPE iszerosub () const {
            return EMULATED_FUNCTIONS::MATH::iszerosub<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // EXP
        inline DERIVED_VEC_TYPE exp () const {
            return EMULATED_FUNCTIONS::MATH::exp<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MEXP
        inline DERIVED_VEC_TYPE exp (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::exp<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SIN
        inline DERIVED_VEC_TYPE sin () const {
            return EMULATED_FUNCTIONS::MATH::sin<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MSIN
        inline DERIVED_VEC_TYPE sin (MASK_TYPE const & mask)const  {
            return EMULATED_FUNCTIONS::MATH::sin<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // COS
        inline DERIVED_VEC_TYPE cos () const {
            return EMULATED_FUNCTIONS::MATH::cos<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MCOS
        inline DERIVED_VEC_TYPE cos (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::cos<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // TAN
        inline DERIVED_VEC_TYPE tan () const {
            return EMULATED_FUNCTIONS::MATH::tan<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MTAN
        inline DERIVED_VEC_TYPE tan (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::tan<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // CTAN
        inline DERIVED_VEC_TYPE ctan () const {
            return EMULATED_FUNCTIONS::MATH::ctan<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MCTAN
        inline DERIVED_VEC_TYPE ctan (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::ctan<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

    };
    
    // This is just an experimental setup! Providing functions like this to handle interface
    // is possible although it will be pretty extensive in number of necessary declarations.
    template<typename VEC_TYPE>
    inline VEC_TYPE addv (VEC_TYPE const & src1, VEC_TYPE const & src2) {
        return src1.add(src2);
    }

    template<typename VEC_TYPE>
    inline VEC_TYPE & addva (VEC_TYPE & src1, VEC_TYPE const & src2) {
        return src1.addva(src2);
    }
    
    // How to restrict template parameter resolution to certain types only?
    template<typename VEC_TYPE>
    inline VEC_TYPE adds (float src1, VEC_TYPE const & src2) {
        return src2.add(src1);
    }
    
    template<typename VEC_TYPE>
    inline VEC_TYPE adds (VEC_TYPE const & src1, float src2) {
        return src1.add(src2);
    }
} // namespace UME::SIMD
} // namespace UME

#endif
