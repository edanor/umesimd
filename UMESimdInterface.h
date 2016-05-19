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
        UME_FORCE_INLINE VEC_TYPE & assign(VEC_TYPE & dst, VEC_TYPE const & src) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, src[i]);
            }
            return dst;
        }

        // MASSIGN
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & assign(MASK_TYPE const & mask, VEC_TYPE & dst, VEC_TYPE const & src) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) dst.insert(i, src[i]);
            }
            return dst;
        }

        // ASSIGNS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE & assign(VEC_TYPE & dst, SCALAR_TYPE src) {
            UME_EMULATION_WARNING();
            for( uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, src);
            }
            return dst;
        }

        // MASSIGNS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & assign(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE src) {
            UME_EMULATION_WARNING();
            for( uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) dst.insert(i, src);
            }
            return dst;
        }

        // LOAD
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE & load(VEC_TYPE & dst, SCALAR_TYPE const * p) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, p[i]);
            }
            return dst;
        }

        // MLOAD
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & load(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE const * p) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++)
            {
                if(mask[i] == true) dst.insert(i, p[i]);
            }
            return dst;
        }

        // LOADA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE & loadAligned(VEC_TYPE & dst, SCALAR_TYPE const * p) {
            UME_ALIGNMENT_CHECK(p, VEC_TYPE::alignment());
            return EMULATED_FUNCTIONS::load<VEC_TYPE, SCALAR_TYPE>(dst, p);
        }

        // MLOADA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & loadAligned(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE const * p) {
            UME_ALIGNMENT_CHECK(p, VEC_TYPE::alignment());
            return EMULATED_FUNCTIONS::load<VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, dst, p);
        }

        // STORE
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE* store(VEC_TYPE const & src, SCALAR_TYPE * p) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++)
            {
                p[i] = src[i];
            }
            return p;
        }

        // MSTORE
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE* store(MASK_TYPE const & mask, VEC_TYPE const & src, SCALAR_TYPE * p) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++)
            {
                if(mask[i] == true) p[i] = src[i];
            }
            return p;
        }

        // STOREA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE* storeAligned(VEC_TYPE const & src, SCALAR_TYPE *p) {
            UME_ALIGNMENT_CHECK(p, VEC_TYPE::alignment());
            return store<VEC_TYPE, SCALAR_TYPE>(src, p); 
        }

        // MSTOREA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE* storeAligned(MASK_TYPE const & mask, VEC_TYPE const & src, SCALAR_TYPE *p) {
            UME_ALIGNMENT_CHECK(p, VEC_TYPE::alignment());
            return store<MASK_TYPE, VEC_TYPE, SCALAR_TYPE>(mask, src, p);
        }

        // GATHERS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE & gather(VEC_TYPE & dst, SCALAR_TYPE* base, uint32_t* indices) {
            UME_EMULATION_WARNING();
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert( i, base[indices[i]]);
            }
            return dst;
        }

        // MGATHERS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & gather(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE* base, uint32_t* indices) {
            UME_EMULATION_WARNING();
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) dst.insert( i, base[indices[i]]);
            }
            return dst;
        }

        // GATHERV
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & gather(VEC_TYPE & dst, SCALAR_TYPE* base, UINT_VEC_TYPE const & indices) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, base[indices[i]]);
            }
            return dst;
        }

        // MGATHERV
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & gather(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE* base, UINT_VEC_TYPE const & indices) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) dst.insert(i, base[indices[i]]);
            }
            return dst;
        }

        // SCATTERS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE* scatter(VEC_TYPE const & src, SCALAR_TYPE* base, uint32_t* indices) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                base[indices[i]] = src[i];
            }
            return base;
        }

        // MSCATTERS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE* scatter(MASK_TYPE const & mask, VEC_TYPE const & src, SCALAR_TYPE* base, uint32_t* indices) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) base[indices[i]] = src[i];
            }
            return base;
        }

        // SCATTERV
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE* scatter(VEC_TYPE const & src, SCALAR_TYPE* base, UINT_VEC_TYPE const & indices) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++)
            {
                base[indices[i]] = src[i];
            }
            return base;
        }

        // MSCATTERV
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE* scatter(MASK_TYPE const & mask, VEC_TYPE const & src, SCALAR_TYPE* base, UINT_VEC_TYPE const & indices) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++)
            {
                if(mask[i] == true) base[indices[i]] = src[i];
            }
            return base;
        }

        // PACK
        template<typename VEC_TYPE, typename VEC_HALF_TYPE>
        UME_FORCE_INLINE VEC_TYPE & pack(VEC_TYPE & dst, VEC_HALF_TYPE const & src1, VEC_HALF_TYPE const & src2) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_HALF_TYPE::length(); i++) {
                dst.insert(i, src1[i]);
                dst.insert(i + VEC_HALF_TYPE::length(), src2[i]);
            }
            return dst;
        }

        // PACKLO
        template<typename VEC_TYPE, typename VEC_HALF_TYPE>
        UME_FORCE_INLINE VEC_TYPE & packLow(VEC_TYPE & dst, VEC_HALF_TYPE const & src1) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_HALF_TYPE::length(); i++) {
                dst.insert(i, src1[i]);
            }
            return dst;
        }

        // PACKHI
        template<typename VEC_TYPE, typename VEC_HALF_TYPE>
        UME_FORCE_INLINE VEC_TYPE & packHigh(VEC_TYPE & dst, VEC_HALF_TYPE const & src1) {
            UME_EMULATION_WARNING();
            for(uint32_t i = VEC_HALF_TYPE::length(); i < VEC_TYPE::length(); i++) {
                dst.insert(i, src1[i - VEC_HALF_TYPE::length()]);
            }
            return dst;
        }
        
        // UNPACK
        template<typename VEC_TYPE, typename VEC_HALF_TYPE>
        UME_FORCE_INLINE void unpack(VEC_TYPE const & src, VEC_HALF_TYPE & dst1, VEC_HALF_TYPE & dst2) {
            UME_EMULATION_WARNING();
            uint32_t halfLength = VEC_HALF_TYPE::length();
            for(uint32_t i = 0; i < halfLength; i++) {
                dst1.insert(i, src[i]);
                dst2.insert(i, src[i + halfLength]);
            }
        }

        // UNPACKLO
        template<typename VEC_TYPE, typename VEC_HALF_TYPE>
        UME_FORCE_INLINE VEC_HALF_TYPE unpackLow(VEC_TYPE const & src) {
            UME_EMULATION_WARNING();
            VEC_HALF_TYPE retval;
            for(uint32_t i = 0; i < VEC_HALF_TYPE::length(); i++) {
                retval.insert(i, src[i]);
            }
            return retval;
        }

        // UNPACKHI
        template<typename VEC_TYPE, typename VEC_HALF_TYPE>
        UME_FORCE_INLINE VEC_HALF_TYPE unpackHigh(VEC_TYPE const & src) {
            UME_EMULATION_WARNING();
            VEC_HALF_TYPE retval;
            for(uint32_t i = 0; i < VEC_HALF_TYPE::length(); i++) {
                retval.insert(i, src[i + VEC_HALF_TYPE::length()]);
            }
            return retval;
        }

        // ADDV
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE add (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] + b[i]);
            }
            return retval; 
        }
        
        // MADDV
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE add (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, mask[i] ? a[i] + b[i] : a[i]);
            }
            return retval;
        }

        // ADDS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE addScalar (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] + b);
            }
            return retval;
        }

        // MADDS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE addScalar (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, mask[i] ? a[i] + b : a[i]);
            }
            return retval;
        }

        // ADDVA
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & addAssign (VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) { a.insert(i, (a[i] + b[i])); }
            return a;
        }

        // MADDVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & addAssign (MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, (a[i] + b[i]));
            }
            return a;
        }

        // ADDSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE & addAssignScalar (VEC_TYPE & a, SCALAR_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, (a[i] + b));
            }
            return a;
        }

        // MADDSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & addAssignScalar (MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i] + b);
            }
            return a;
        }

        // SADDV
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE addSaturated (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            decltype(a.extract(0)) temp = 0;
            // maximum value
            decltype(a.extract(0)) maxValue = std::numeric_limits<decltype(a.extract(0))>::max();
            decltype(a.extract(0)) minValue = std::numeric_limits<decltype(a.extract(0))>::min();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (a[i] > 0 && b[i] > 0) {
                    temp = a[i] > (maxValue - b[i]) ? maxValue : (a[i] + b[i]);
                }
                else if (a[i] < 0 && b[i] < 0) {
                    temp = a[i] < (minValue - b[i]) ? minValue : (a[i] + b[i]);
                }
                else
                {
                    temp = a[i] + b[i];
                }
                retval.insert(i, temp);
            }
            return retval;
        }

        // MSADDV
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE addSaturated (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            decltype(a.extract(0)) temp = 0;
            // maximum value
            decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::max();
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
        UME_FORCE_INLINE VEC_TYPE addSaturatedScalar (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            decltype(a.extract(0)) temp = 0;
            // maximum value
            decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::max();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = (a[i] > (satValue - b)) ? satValue : (a[i] + b);
                retval.insert(i, temp);
            }
            return retval;
        }

        // MSADDS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE addSaturatedScalar (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            decltype(a.extract(0)) temp = 0;
            // maximum value
            decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::max();
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
        UME_FORCE_INLINE VEC_TYPE & addSaturatedAssign(VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            decltype(a.extract(0)) temp = 0;
            // maximum value
            decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::max();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = (a[i] > (satValue - b[i])) ? satValue : (a[i] + b[i]);
                a.insert(i, temp);
            }
            return a;
        }
        
        // MSADDVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & addSaturatedAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            decltype(a.extract(0)) temp = 0;
            // maximum value
            decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::max();
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
        UME_FORCE_INLINE VEC_TYPE & addSaturatedScalarAssign(VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            decltype(a.extract(0)) temp = 0;
            // maximum value
            decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::max();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = (a[i] > (satValue - b)) ? satValue : (a[i] + b);
                a.insert(i, temp);
            }
            return a;
        }

        // MSADDSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & addSaturatedScalarAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            decltype(a.extract(0)) temp = 0;
            // maximum value
            decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::max();
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
        UME_FORCE_INLINE VEC_TYPE postfixIncrement(VEC_TYPE & a) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i] + 1);
            }
            return retval;
        }

        // MPOSTINC
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE postfixIncrement(MASK_TYPE const & mask, VEC_TYPE & a) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i] + 1);
            }
            return retval;
        }

        // PREFINC
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & prefixIncrement(VEC_TYPE & a) {
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++)
            {
                a.insert(i, a[i] + 1);
            }
            return a;
        }

        // MPREFINC
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & prefixIncrement(MASK_TYPE const & mask, VEC_TYPE & a) {
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++)
            {
                if(mask[i] == true) a.insert(i, a[i] + 1);
            }
            return a;
        }

        // SUBV
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE sub ( VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] - b[i]);
            }
            return retval; 
        }

        // MSUBV
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE sub ( MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
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
        UME_FORCE_INLINE VEC_TYPE subScalar ( VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (a[i] - b));
            }
            return retval;
        }

        // MSUBS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE subScalar ( MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
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
        UME_FORCE_INLINE VEC_TYPE subFrom (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] - b[i]);
            }
            return retval;
        }

        // MSUBFROMV
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE subFrom (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
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
        UME_FORCE_INLINE VEC_TYPE subFromScalar (SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a - b[i]);
            }
            return retval;
        }

        // MSUBFROMS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE subFromScalar (MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
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
        UME_FORCE_INLINE VEC_TYPE & subFromAssign (VEC_TYPE const & a, VEC_TYPE & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                b.insert(i, a[i] - b[i]);
            }
            return b;
        }

        // MSUBFROMVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & subFromAssign (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) b.insert(i, a[i] - b[i]);
                else b.insert(i, a[i]);
            }
            return b;
        }

        // SUBFROMSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE & subFromScalarAssign (SCALAR_TYPE a, VEC_TYPE & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                b.insert(i, a - b[i]);
            }
            return b;
        }

        // MSUBFROMSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & subFromScalarAssign (MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) b.insert(i, a - b[i]);
                else b.insert(i, a);
            }
            return b;
        }

        // NEG
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE unaryMinus (VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, -a[i]);
            }
            return retval;
        }

        // MNEG
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE unaryMinus (MASK_TYPE const & mask, VEC_TYPE const & a) {
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
        UME_FORCE_INLINE VEC_TYPE & unaryMinusAssign (VEC_TYPE & a) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, -a[i]);
            }
            return a;
        }

        // MNEGA
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & unaryMinusAssign (MASK_TYPE const & mask, VEC_TYPE & a) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if( mask[i] == true ) a.insert(i, -a[i]);
            }
            return a;
        }
            
        // SUBVA
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & subAssign (VEC_TYPE & dst, VEC_TYPE const & b)
        {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, dst[i] - b[i]);
            }
            return dst;
        }

        // MSUBVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & subAssign (MASK_TYPE const & mask, VEC_TYPE & dst, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true ) dst.insert(i, dst[i] - b[i]);
            }
            return dst;
        }

        // SUBSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE & subAssign (VEC_TYPE & dst, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, dst[i] - b);
            }
            return dst;
        }

        // MSUBSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & subAssign (MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) dst.insert(i, dst[i] - b);
            }
            return dst;
        }

        // SSUBV
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE subSaturated (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            decltype(a.extract(0)) temp = 0;
            // maximum value
            decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::min();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = (a[i] < (satValue + b[i])) ? satValue : (a[i] - b[i]);
                retval.insert(i, temp);
            }
            return retval;
        }

        // MSSUBV
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE subSaturated (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            decltype(a.extract(0)) temp = 0;
            // maximum value
            decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::min();
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
        UME_FORCE_INLINE VEC_TYPE subSaturated (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            decltype(a.extract(0)) temp = 0;
            // maximum value
            decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::min();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = (a[i] < (satValue + b)) ? satValue : (a[i] - b);
                retval.insert(i, temp);
            }
            return retval;
        }

        // MSSUBS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE subSaturated (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            decltype(a.extract(0)) temp = 0;
            // maximum value
            decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::min();
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
        UME_FORCE_INLINE VEC_TYPE & subSaturatedAssign (VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            decltype(a.extract(0)) temp = 0;
            // maximum value
            decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::min();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = (a[i] < (satValue + b[i])) ? satValue : (a[i] - b[i]);
                a.insert(i, temp);
            }
            return a;
        }

        // MSSUBV
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & subSaturatedAssign (MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            decltype(a.extract(0)) temp = 0;
            // maximum value
            decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::min();
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
        UME_FORCE_INLINE VEC_TYPE & subSaturatedScalarAssign (VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            decltype(a.extract(0)) temp = 0;
            // maximum value
            decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::min();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = (a[i] < (satValue + b)) ? satValue : (a[i] - b);
                a.insert(i, temp);
            }
            return a;
        }

        // MSSUBS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & subSaturatedScalarAssign (MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            decltype(a.extract(0)) temp = 0;
            // maximum value
            decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::min();
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
        UME_FORCE_INLINE VEC_TYPE postfixDecrement(VEC_TYPE & a) {
            VEC_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i] - 1);
            }
            return retval;
        }

        // MPOSTDEC
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE postfixDecrement(MASK_TYPE const & mask, VEC_TYPE & a) {
            VEC_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i] - 1);
            }
            return retval;
        }

        // PREFDEC
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & prefixDecrement(VEC_TYPE & a) {
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i]-1 );
            }
            return a;
        }

        // MPREFDEC
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & prefixDecrement(MASK_TYPE const & mask, VEC_TYPE & a) {
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i]-1 );
            }
            return a;
        }
            
        // MULV
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE mult (VEC_TYPE const & a, VEC_TYPE const & b) {
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
        UME_FORCE_INLINE VEC_TYPE mult (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? a[i]*b[i] : a[i] );
            }
            return retval;
        }

        // MULS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE mult (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]*b );
            }
            return retval;
        }

        // MMULS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE mult (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? a[i]*b : a[i]);
            }
            return retval;
        }

        // MULVA
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & multAssign (VEC_TYPE & dst, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, dst[i] * b[i]);
            }
            return dst;
        }

        // MMULVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & multAssign (MASK_TYPE const & mask, VEC_TYPE & dst, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) dst.insert(i, dst[i] * b[i]);
            }
            return dst;
        }

        // MULSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE & multAssign (VEC_TYPE & dst, SCALAR_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                dst.insert(i, dst[i] * b);
            }
            return dst;
        }

        // MMULSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & multAssign (MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) dst.insert(i, dst[i] * b);
            }
            return dst;
        }

        // DIVV
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE div (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]/b[i] );
            }
            return retval;
        }

        // MDIVV
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE div (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? a[i]/b[i] : a[i]);
            }
            return retval;
        }

        // DIVS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE div (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]/b );
            }
            return retval;
        }

        // MDIVS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE div (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? (a[i]/b) : a[i]);
            }
            return retval;
        }

        // RCP
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE div (SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a/b[i] );
            }
            return retval;
        }

        // MRPC
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE div (MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ?  (a/b[i]) : a);
            }
            return retval;
        }

        // DIVVA
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & divAssign(VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i]/b[i] );
            }
            return a;
        }

        // MDIVVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & divAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i]/b[i] );
            }
            return a;
        }
            
        // DIVSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE & divAssign(VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i]/b );
            }
            return a;
        }

        // MDIVSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & divAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i]/b);
            }
            return a;
        }

        // RCP
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE rcp(VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, decltype(retval.extract(0))(1.0)/b[i]);
            }
            return retval;
        }

        // MRCP
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE rcp(MASK_TYPE const & mask, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) retval.insert(i, decltype(retval.extract(0))(1.0)/b[i]);
                else retval.insert(i, b[i]);
            }
            return retval;
        }

        // RCPS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE rcpScalar(SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a/b[i]);
            }
            return retval;
        }

        // MRCPS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE rcpScalar(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
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
        UME_FORCE_INLINE VEC_TYPE & rcpAssign(VEC_TYPE & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                b.insert(i, decltype(b.extract(0))(1.0)/b[i]);
            }
            return b;
        }

        // MRCPA
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & rcpAssign(MASK_TYPE const & mask, VEC_TYPE & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) b.insert(i, decltype(b.extract(0))(1.0)/b[i]);
            }
            return b;
        }

        // RCPSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE & rcpScalarAssign(SCALAR_TYPE a, VEC_TYPE & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                b.insert(i, a/b[i]);
            }
            return b;
        }

        // MRCPSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & rcpScalarAssign(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) b.insert(i, a/b[i]);
            }
            return b;
        }

        // LSHV
        template<typename VEC_TYPE, typename UINT_VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE shiftBitsLeft(VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (a[i] << b[i]) );
            }
            return retval;
        }

        // MLSHV
        template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE shiftBitsLeft(MASK_TYPE const & mask, VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? (a[i] << b[i]) : a[i]);
            }
            return retval;
        }

        // LSHS
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE>
        UME_FORCE_INLINE VEC_TYPE shiftBitsLeftScalar(VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (a[i] << b) );
            }
            return retval;
        }

        // MLSHS
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE shiftBitsLeftScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? (a[i] << b ) : a[i]);
            }
            return retval;
        }

        // LSHVA
        template<typename VEC_TYPE, typename UINT_VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & shiftBitsLeftAssign(VEC_TYPE & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, (a[i] << b[i]));
            }
            return a;
        }

        // MLSHVA
        template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & shiftBitsLeftAssign(MASK_TYPE const & mask, VEC_TYPE & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i]<<b[i] );
            }
            return a;
        }

        // LSHSA
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE>
        UME_FORCE_INLINE VEC_TYPE & shiftBitsLeftAssignScalar(VEC_TYPE & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, (a[i] << b) );
            }
            return a;
        }

        // MLSHSA
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & shiftBitsLeftAssignScalar(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, (a[i] << b) );
            }
            return a;
        }

        // RSHV
        template<typename VEC_TYPE, typename UINT_VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE shiftBitsRight(VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (a[i] >> b[i]));
            }
            return retval;
        }

        // MRSHV
        template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE shiftBitsRight(MASK_TYPE const & mask, VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? (a[i]>>b[i]) : a[i]);
            }
            return retval;
        }

        // RSHS
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE>
        UME_FORCE_INLINE VEC_TYPE shiftBitsRightScalar(VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (a[i] >> b) );
            }
            return retval;
        }

        // MRSHS
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE shiftBitsRightScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? (a[i] >> b) : a[i] );
            }
            return retval;
        }

        // RSHVA
        template<typename VEC_TYPE, typename UINT_VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & shiftBitsRightAssign(VEC_TYPE & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, (a[i] >> b[i]));
            }
            return a;
        }

        // MRSHVA
        template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & shiftBitsRightAssign(MASK_TYPE const & mask, VEC_TYPE & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, (a[i] >> b[i]) );
            }
            return a;
        }

        // RSHSA
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE>
        UME_FORCE_INLINE VEC_TYPE & shiftBitsRightAssignScalar(VEC_TYPE & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, (a[i] >> b) );
            }
            return a;
        }

        // MSRHSA
        template<typename VEC_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & shiftBitsRightAssignScalar(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, (a[i] >> b) );
            }
            return a;
        }

        // ROLV
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE, typename SCALAR_UINT_TYPE>
        UME_FORCE_INLINE VEC_TYPE rotateBitsLeft(VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            uint32_t bitLength = 8*sizeof(SCALAR_TYPE);
            SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
            bool topBit;
            SCALAR_TYPE shifted;
            SCALAR_TYPE raw_a[VEC_TYPE::length()];
            SCALAR_UINT_TYPE raw_b[UINT_VEC_TYPE::length()];
            SCALAR_TYPE raw_retval[VEC_TYPE::length()];

            a.store(raw_a);
            b.store(raw_b);

            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                shifted = raw_a[i];
                // shift one bit at a time. This simplifies type dependency checks.
                for(uint32_t j = 0; j < raw_b[i]; j++) {
                  if( (shifted & topBitMask) != 0) topBit = true;
                  else topBit = false;
                  
                  shifted <<= 1;        
                  if(topBit == true) shifted |= SCALAR_TYPE(1);
                  else               shifted &= ~(SCALAR_TYPE(1)); 
                }
                raw_retval[i] = shifted; 
            }
            retval.load(raw_retval);
            return retval;
        }

        // MROLV
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE rotateBitsLeft(MASK_TYPE const & mask, VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
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
        UME_FORCE_INLINE VEC_TYPE rotateBitsLeftScalar(VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
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
        UME_FORCE_INLINE VEC_TYPE rotateBitsLeftScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
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
        UME_FORCE_INLINE VEC_TYPE & rotateBitsLeftAssign(VEC_TYPE & a, UINT_VEC_TYPE const & b) {
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
        UME_FORCE_INLINE VEC_TYPE & rotateBitsLeftAssign(MASK_TYPE const & mask, VEC_TYPE & a, UINT_VEC_TYPE const & b) {
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
        UME_FORCE_INLINE VEC_TYPE & rotateBitsLeftAssignScalar(VEC_TYPE & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            SCALAR_TYPE bitLength = 8*sizeof(SCALAR_UINT_TYPE);
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
                  if(topBit == true)  shifted |= SCALAR_TYPE(0x1);
                  else                shifted &= ~(SCALAR_TYPE(1)); 
                }
                a.insert(i, shifted); 
            }
            return a;
        }

        // MROLSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & rotateBitsLeftAssignScalar(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            uint32_t bitLength = 8*sizeof(SCALAR_UINT_TYPE);
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
                    if(topBit == true)  shifted |= SCALAR_TYPE(0x1);
                    else                shifted &= ~(SCALAR_TYPE(1)); 
                  }
                  a.insert(i, shifted);
                } 
            }
            return a;
        }

        // RORV
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE, typename SCALAR_UINT_TYPE>
        UME_FORCE_INLINE VEC_TYPE rotateBitsRight(VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            uint32_t bitLength = 8*sizeof(SCALAR_TYPE);
            SCALAR_UINT_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
            bool bottomBit;
            SCALAR_UINT_TYPE shifted;

            SCALAR_TYPE raw_a[VEC_TYPE::length()];
            SCALAR_UINT_TYPE raw_b[VEC_TYPE::length()];
            SCALAR_TYPE raw_retval[VEC_TYPE::length()];

            a.store(raw_a);
            b.store(raw_b);

            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                shifted = raw_a[i];
                // shift one bit at a time. This simplifies type dependency checks.
                for(uint32_t j = 0; j < raw_b[i]; j++) {
                    if( (shifted & 1) != 0) bottomBit = true;
                    else bottomBit = false;

                    shifted >>= 1;
                    if(bottomBit == true) shifted |= topBitMask;
                    else                  shifted &= ~topBitMask;
                }
                raw_retval[i] = (SCALAR_TYPE)shifted;
            }
            retval.load(raw_retval);
            return retval;
        }

        // MRORV
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE rotateBitsRight(MASK_TYPE const & mask, VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
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
        UME_FORCE_INLINE VEC_TYPE rotateBitsRightScalar(VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
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
        UME_FORCE_INLINE VEC_TYPE rotateBitsRightScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
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
        UME_FORCE_INLINE VEC_TYPE & rotateBitsRightAssign(VEC_TYPE & a, UINT_VEC_TYPE const & b) {
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
        UME_FORCE_INLINE VEC_TYPE & rotateBitsRightAssign(MASK_TYPE const & mask, VEC_TYPE & a, UINT_VEC_TYPE const & b) {
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
        UME_FORCE_INLINE VEC_TYPE & rotateBitsRightAssignScalar(VEC_TYPE &  a, SCALAR_UINT_TYPE const & b) {
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
        UME_FORCE_INLINE VEC_TYPE & rotateBitsRightAssignScalar(MASK_TYPE const & mask, VEC_TYPE &  a, SCALAR_UINT_TYPE const & b) {
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
        UME_FORCE_INLINE MASK_TYPE isEqual(VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]==b[i] );
            }
            return retval;
        }

        // CMPEQS
        template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE MASK_TYPE isEqual(VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] == b );
            }
            return retval;
        }

        // CMPNEV
        template<typename MASK_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE MASK_TYPE isNotEqual (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]!=b[i] );
            }
            return retval;
        }

        // CMPNES
        template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE MASK_TYPE isNotEqual (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]!=b );
            }
            return retval;
        }

        // CMPGTV
        template<typename MASK_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE MASK_TYPE isGreater (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]>b[i] );
            }
            return retval;
        }

        // CMPGTS
        template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE MASK_TYPE isGreater (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]>b );
            }
            return retval;
        }

        // CMPLTV
        template<typename MASK_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE MASK_TYPE isLesser(VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]<b[i]);
            }
            return retval;
        }

        // CMPLTS
        template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE MASK_TYPE isLesser(VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i]<b );
            }
            return retval;
        }

        // CMPGEV
        template<typename MASK_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE MASK_TYPE isGreaterEqual(VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] >= b[i] );
            }
            return retval;
        }

        // CMPGES
        template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE MASK_TYPE isGreaterEqual(VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] >= b );
            }
            return retval;
        }

        // CMPLEV
        template<typename MASK_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE MASK_TYPE isLesserEqual(VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] <= b[i] );
            }
            return retval;
        }

        // CMPLES
        template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE MASK_TYPE isLesserEqual(VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] <= b );
            }
            return retval;
        }

        // CMPEV 
        template<typename VEC_TYPE>
        UME_FORCE_INLINE bool isExact(VEC_TYPE const & a, VEC_TYPE const & b) {
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
        UME_FORCE_INLINE MASK_TYPE isEqualInRange(VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & margin) {
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
        UME_FORCE_INLINE MASK_TYPE isEqualInRange(VEC_TYPE const & a, VEC_TYPE const & b, SCALAR_TYPE margin) {
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
        UME_FORCE_INLINE bool unique(VEC_TYPE const & a) {
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
        UME_FORCE_INLINE VEC_TYPE binaryAnd (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] & b[i] );
            }
            return retval;
        }

        // MANDV
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE binaryAnd (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] ? a[i] & b[i] : a[i]) );
            }
            return retval;
        }

        // ANDS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE binaryAnd (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] & b);
            }
            return retval;
        }

        // MANDS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE binaryAnd (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] ? a[i] & b : a[i]) );
            }
            return retval;
        }

        // binaryAnd (scalar, VEC) -> VEC
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE binaryAnd (SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a & b[i]);
            }
            return retval;
        }

        // binaryAnd (MASK, scalar, VEC) -> VEC
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE binaryAnd (MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] ? a & b[i] : a) );
            }
            return retval;
        }

        // ANDVA
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & binaryAndAssign (VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i] & b[i]);
            }
            return a;
        }

        // MANDVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & binaryAndAssign (MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i] & b[i]);
            }
            return a;
        }

        // ANDSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE & binaryAndAssign (VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i] & b);
            }
            return a;
        }

        // MANDSA 
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & binaryAndAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) a.insert(i, a[i] & b);
            }
            return a;
        }

        // ORV
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE binaryOr (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] | b[i] );
            }
            return retval;
        }

        // MORV
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE binaryOr (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] ? (a[i] | b[i]) : a[i]) );
            }
            return retval;
        }

        // ORS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE binaryOr (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] | b);
            }
            return retval;
        }

        // MORS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE binaryOr (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] ? (a[i] | b) : a[i]));
            }
            return retval;
        }

        // ORVA
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & binaryOrAssign (VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i] | b[i]);
            }
            return a;
        }

        // MORVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & binaryOrAssign (MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i] | b[i]);
            }
            return a;
        }

        // ORSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE & binaryOrAssign (VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i] | b);
            }
            return a;
        }

        // MORSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & binaryOrAssign (MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i] | b);
            }
            return a;
        }

        // XORV
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE binaryXor (VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] ^ b[i] );
            }
            return retval;
        }

        // MXORV
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE binaryXor (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? (a[i] ^ b[i]) : a[i] );
            }
            return retval;
        }

        // XORS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE binaryXor (VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] ^ b);
            }
            return retval;
        }

        // MXORS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE binaryXor (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? (a[i] ^ b) : a[i]);
            }
            return retval;
        }

        // XORVA
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & binaryXorAssign (VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i] ^ b[i]);
            }
            return a;
        }

        // MXORVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & binaryXorAssign (MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i] ^ b[i]);
            }
            return a;
        }

        // XORSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE & binaryXorAssign (VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i] ^ b);
            }
            return a;
        }

        // MXORSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & binaryXorAssign (MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, a[i] ^ b);
            }
            return a;
        }

        // BNOT
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE binaryNot (VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                SCALAR_TYPE temp = ~a[i];
                retval.insert(i, temp);
            }
            return retval;
        }

        // MBNOT
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE binaryNot (MASK_TYPE const & mask, VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? (~a[i]) : (a[i]));
            }
            return retval;
        }

        // BNOTA
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & binaryNotAssign (VEC_TYPE & a) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, ~a[i]);
            }
            return a;
        }

        // MBNOTA
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & binaryNotAssign (MASK_TYPE const & mask, VEC_TYPE & a) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) a.insert(i, ~a[i]);
            }
            return a;
        }

        // LNOT
        template<typename MASK_TYPE>
        UME_FORCE_INLINE MASK_TYPE logicalNot(MASK_TYPE const & mask) {
            UME_EMULATION_WARNING();
            MASK_TYPE retval(false);
            for(uint32_t i = 0; i < MASK_TYPE::length(); i++) {
                if(mask[i] == false) retval.insert(i, true);
            }
            return retval;
        }

        // LNOTA
        template<typename MASK_TYPE>
        UME_FORCE_INLINE MASK_TYPE & logicalNotAssign(MASK_TYPE & mask) {
            UME_EMULATION_WARNING();
            for(uint32_t i = 0; i < MASK_TYPE::length(); i++) {
                mask.insert(i, !mask[i]);
            }
            return mask;
        }

        // BLENDV
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE blend (MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, mask[i] ? a[i] : b[i]);
            }
            return retval;
        }

        // BLENDS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE blend (MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, mask[i] ? a[i] : b);
            }
            return retval;
        }

        // SWIZZLE
        template<typename VEC_TYPE, typename SWIZZLE_MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE swizzle(SWIZZLE_MASK_TYPE const & sMask, VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            VEC_TYPE retval;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[sMask[i]]);
            }
            return retval;
        }

        // SWIZZLEA
        template<typename VEC_TYPE, typename SWIZZLE_MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & swizzleAssign(SWIZZLE_MASK_TYPE const & sMask, VEC_TYPE & a) {
            UME_EMULATION_WARNING();
            VEC_TYPE temp(a);
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, temp[sMask[i]]);
            }
            return a;
        }

        // reduceAdd(VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceAdd (VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a[0];
            for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                retval += a[i];
            }
            return retval;
        }

        // reduceAdd(MASK, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceAdd (MASK_TYPE const & mask, VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a[0];
            for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                if( mask[i] == true ) retval += a[i];
            }
            return retval;
        }

        // reduceAdd (scalar, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceAdd (SCALAR_TYPE & a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i <VEC_TYPE::length(); i++) {
                retval += b[i];
            }
            return retval;
        }

        // reduceAdd(MASK, scalar, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceAdd (MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i <VEC_TYPE::length(); i++) {
                if( mask[i] == true ) retval += b[i];
            }
            return retval;
        }

        // reduceMult(VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceMult (VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a[0];
            for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                retval *= a[i];
            }
            return retval;
        }

        // reduceMult(MASK, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceMult (MASK_TYPE const & mask, VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = (mask[0] == true) ? a[0] : 0; // TODO: replace 0 with const expr returning zero depending on SCALAR type.
            for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                if( mask[i] == true ) retval *= a[i];
            }
            return retval;
        }

        // reduceMult(scalar, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceMultScalar (SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval *= b[i];
            }
            return retval;
        }

        // reduceMult(MASK, scalar, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceMultScalar (MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if( mask[i] == true ) retval *= b[i];
            }
            return retval;
        }

        // HLAND
        template<typename MASK_TYPE>
        UME_FORCE_INLINE bool reduceLogicalAnd(MASK_TYPE const & a) {
            UME_EMULATION_WARNING();
            bool retval = a[0];
            for(uint32_t i = 1; i < MASK_TYPE::length(); i++) {
                retval &= a[i];
            }
            return retval;
        }

        // HLOR
        template<typename MASK_TYPE>
        UME_FORCE_INLINE bool reduceLogicalOr(MASK_TYPE const & a) {
            UME_EMULATION_WARNING();
            bool retval = a[0];
            for(uint32_t i = 1; i < MASK_TYPE::length(); i++) {
                retval |= a[i];
            }
            return retval;
        }

        // HLXOR
        template<typename MASK_TYPE>
        UME_FORCE_INLINE bool reduceLogicalXor(MASK_TYPE const & a) {
            UME_EMULATION_WARNING();
            bool retval = a[0];
            for(uint32_t i = 1; i < MASK_TYPE::length(); i++) {
                retval ^= a[i];
            }
            return retval;
        }

        // reduceBinaryAnd (VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceBinaryAnd(VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a[0];
            for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                retval &= a[i];
            }
            return retval;
        }

        // reduceBinaryAnd (MASK, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceBinaryAnd(MASK_TYPE const & mask, VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = (mask[0] == true) ? a[0] : (SCALAR_TYPE)-1;
            for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                if( mask[i] == true ) retval &= a[i];
            }
            return retval;
        }

        // reduceBinaryAnd (scalar, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceBinaryAndScalar(SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval &= b[i];
            }
            return retval;
        }

        // reduceBinaryAnd (MASK, scalar, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceBinaryAndScalar(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if(mask[i] == true) retval &= b[i];
            }
            return retval;
        }

        // reduceBinaryOr (VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceBinaryOr (VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a[0];
            for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                retval |= a[i];
            }
            return retval;
        }

        // reduceBinaryOr (MASK, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceBinaryOr (MASK_TYPE const & mask, VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = (mask[0] == true) ? a[0] : 0; // TODO: 0-initializer of SCALAR_TYPE
            for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                if( mask[i] == true ) retval |= a[i];
            }
            return retval;
        }

        // reduceBinaryOr (scalar, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceBinaryOrScalar (SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval |= b[i];
            }
            return retval;
        }     

        // reduceBinaryOr (MASK, scalar, VEC) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceBinaryOrScalar (MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if( mask[i] == true ) retval |= b[i];
            }
            return retval;
        }

        // reduceBinaryXor() -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceBinaryXor(VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = 0;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) { 
                retval ^= a[i];
            }
            return retval;
        }

        // reduceBinaryXor(MASK) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceBinaryXor(MASK_TYPE const & mask, VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = 0;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) { 
                if(mask[i] == true) retval ^= a[i];
            }
            return retval;
        }

        // reduceBinaryXor(scalar) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceBinaryXorScalar(SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) { 
                retval ^= b[i];
            }
            return retval;
        }

        // reduceBinaryXor(MASK, scalar) -> scalar
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceBinaryXorScalar(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            SCALAR_TYPE retval = a;
            for(uint32_t i = 0; i < VEC_TYPE::length(); i++) { 
                if(mask[i] == true) retval ^= b[i];
            }
            return retval;
        }

        // xTOy (UTOI, ITOU, UTOF, FTOU, PROMOTE, DEGRADE)
        template<typename VEC_Y_TYPE, typename SCALAR_Y_TYPE, typename VEC_X_TYPE>
        UME_FORCE_INLINE VEC_Y_TYPE xtoy(VEC_X_TYPE const & a) {
            UME_EMULATION_WARNING();
            static_assert(VEC_X_TYPE::length() == VEC_Y_TYPE::length(),
                "Cannot cast between vectors of different lengths");
            VEC_Y_TYPE retval;
            for (uint32_t i = 0; i < VEC_X_TYPE::length();i++) {
                retval.insert(i, SCALAR_Y_TYPE(a[i]));
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
            UME_FORCE_INLINE VEC_TYPE max(VEC_TYPE const & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (a[i] > b[i] ? a[i] : b[i]));
                }
                return retval;
            }

            // MMAXV
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE max(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
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
            UME_FORCE_INLINE VEC_TYPE maxScalar(VEC_TYPE const & a, SCALAR_TYPE b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (a[i] > b ? a[i] : b));
                }
                return retval;
            }

            // MMAXS
            template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE maxScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
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
            UME_FORCE_INLINE VEC_TYPE & maxAssign(VEC_TYPE & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(b[i] > a[i])a.insert(i, b[i]);
                }
                return a;
            }

            // MMAXVA
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE & maxAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] ==true && (b[i] > a[i]))a.insert(i, b[i]);
                }
                return a;
            }

            // MAXSA
            template<typename VEC_TYPE, typename SCALAR_TYPE>
            UME_FORCE_INLINE VEC_TYPE & maxScalarAssign(VEC_TYPE & a, SCALAR_TYPE b) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(b > a[i]) a.insert(i, b);
                }
                return a;
            }

            // MMAXSA
            template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE & maxScalarAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true && (b > a[i])) a.insert(i, b);
                }
                return a;
            }

            // MINS
            template<typename VEC_TYPE, typename SCALAR_TYPE>
            UME_FORCE_INLINE VEC_TYPE minScalar(VEC_TYPE const & a, SCALAR_TYPE b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, a[i] < b ? a[i] : b);
                }
                return retval;
            }

            // MMINS
            template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE minScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
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
            UME_FORCE_INLINE VEC_TYPE min(VEC_TYPE const & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, a[i] < b[i] ? a[i] : b[i]);
                }
                return retval;
            }

            // MMINV
            template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE min(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval(std::numeric_limits<SCALAR_TYPE>::max());
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, a[i] < b[i] ? a[i] : b[i]);
                    else retval.insert(i, a[i]);
                }
                return retval;
            }

            // MINSA
            template<typename VEC_TYPE, typename SCALAR_TYPE>
            UME_FORCE_INLINE VEC_TYPE & minScalarAssign(VEC_TYPE & a, SCALAR_TYPE b) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(b < a[i]) a.insert(i, b);
                }
                return a;
            }

            // MMINSA
            template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE & minScalarAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true && (b < a[i])) a.insert(i, b);
                }
                return a;
            }

            // MINVA
            template<typename VEC_TYPE>
            UME_FORCE_INLINE VEC_TYPE & minAssign(VEC_TYPE & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(b[i] < a[i]) a.insert(i, b[i]);
                }
                return a;
            }

            // MMINVA
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE & minAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true && (b[i] < a[i])) a.insert(i, b[i]);
                }
                return a;
            }

            // HMAX
            template<typename SCALAR_TYPE, typename VEC_TYPE>
            UME_FORCE_INLINE SCALAR_TYPE reduceMax(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                SCALAR_TYPE retval = a[0];
                for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                    if(a[i] > retval) retval = a[i];
                }
                return retval;
            }

            // MHMAX
            template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE SCALAR_TYPE reduceMax(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                SCALAR_TYPE retval = std::numeric_limits<SCALAR_TYPE>::min();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if( (mask[i] == true) && a[i] > retval) retval = a[i];
                }
                return retval;
            }

            // HMAXS
            template<typename SCALAR_TYPE, typename VEC_TYPE>
            UME_FORCE_INLINE SCALAR_TYPE reduceMax(SCALAR_TYPE a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                SCALAR_TYPE retval = a;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(b[i] > retval) retval = b[i];
                }
                return retval;
            }

            // MHMAXS
            template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE SCALAR_TYPE reduceMax(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                SCALAR_TYPE retval = a;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if((mask[i] == true) && (a[i] > retval)) retval = a[i];
                }
                return retval;
            }

            // IMAX
            template<typename VEC_TYPE, typename SCALAR_TYPE>
            UME_FORCE_INLINE uint32_t indexMax(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                uint32_t indexMax = 0;
                SCALAR_TYPE maxVal = a[0];
                for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                    if(a[i] > maxVal) {
                        maxVal = a[i];
                        indexMax = i;
                    }
                }
                return indexMax;
            }

            // MIMAX
            template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE uint32_t indexMax(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                uint32_t indexMax = 0xFFFFFFFF;
                SCALAR_TYPE maxVal = std::numeric_limits<SCALAR_TYPE>::min();
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
            UME_FORCE_INLINE SCALAR_TYPE reduceMin(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                SCALAR_TYPE retval = a[0];
                for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                    if(a[i] < retval) retval = a[i];
                }
                return retval;
            }

            // MHMIN
            template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE SCALAR_TYPE reduceMin(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                SCALAR_TYPE retval = std::numeric_limits<SCALAR_TYPE>::max();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if( (mask[i] == true) && a[i] < retval) retval = a[i];
                }
                return retval;
            }

            // IMIN
            template<typename VEC_TYPE, typename SCALAR_TYPE>
            UME_FORCE_INLINE uint32_t indexMin(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                uint32_t indexMin = 0;
                SCALAR_TYPE minVal = std::numeric_limits<SCALAR_TYPE>::max();
                for(uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                    if(a[i] < minVal) {
                        minVal = a[i];
                        indexMin = i;
                    }
                }
                return indexMin;
            }

            // MIMIN
            template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE uint32_t indexMin(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                uint32_t indexMin = 0xFFFFFFFF;
                SCALAR_TYPE minVal = std::numeric_limits<SCALAR_TYPE>::max();
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
            UME_FORCE_INLINE VEC_TYPE abs(VEC_TYPE const & a) {
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
            UME_FORCE_INLINE VEC_TYPE abs(MASK_TYPE const & mask, VEC_TYPE const & a) {
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
            UME_FORCE_INLINE VEC_TYPE & absAssign(VEC_TYPE & a) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    // abs for floating point numbers is non-trivial. Using std::abs for reliability.
                    a.insert(i, std::abs(a[i])); 
                }
                return a;
            }

            // MABSA
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE absAssign(MASK_TYPE const & mask, VEC_TYPE & a) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    // abs for floating point numbers is non-trivial. Using std::abs for reliability.
                    a.insert(i, (mask[i] == true ? std::abs(a[i]) : a[i] ));
                }
                return a;
            }

            // SQR
            template<typename VEC_TYPE>
            UME_FORCE_INLINE VEC_TYPE sqr(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, a[i] * a[i]);
                }
                return retval;
            }
            
            // MSQR
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE sqr(MASK_TYPE const & mask, VEC_TYPE const & a) {
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
            UME_FORCE_INLINE VEC_TYPE & sqrAssign(VEC_TYPE & a) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    a.insert(i, a[i] * a[i]);
                }
                return a;
            }
            
            // MSQRA
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE & sqrAssign(MASK_TYPE const & mask, VEC_TYPE & a) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) a.insert(i, a[i] * a[i]);
                }
                return a;
            }

            // SQRT
            template<typename VEC_TYPE>
            UME_FORCE_INLINE VEC_TYPE sqrt(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::sqrt(a[i])); 
                }
                return retval;
            }
            
            // MSQRT
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE sqrt(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (mask[i] == true) ? std::sqrt(a[i]) : a[i]);
                }
                return retval;
            }

            // SQRTA
            template<typename VEC_TYPE>
            UME_FORCE_INLINE VEC_TYPE & sqrtAssign (VEC_TYPE & a) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    a.insert(i, std::sqrt(a[i]));
                }
                return a;
            }

            // MSQRTA
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE & sqrtAssign(MASK_TYPE const & mask, VEC_TYPE & a) {
                UME_EMULATION_WARNING();
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) a.insert(i, std::sqrt(a[i]));
                }
                return a;
            }

            // RSQRT
            template<typename VEC_TYPE>
            UME_FORCE_INLINE VEC_TYPE rsqrt(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, decltype(retval.extract(0))(1.0)/std::sqrt(a[i])); 
                }
                return retval;
            }
            // MRSQRT
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE rsqrt(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                decltype(retval.extract(0)) temp;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    temp = decltype(retval.extract(0))(1.0)/std::sqrt(a[i]);
                    retval.insert(i, (mask[i] == true) ? temp : a[i]);
                }
                return retval;
            }
            // RSQRTA
            template<typename VEC_TYPE>
            UME_FORCE_INLINE VEC_TYPE & rsqrtAssign (VEC_TYPE & a) {
                UME_EMULATION_WARNING();
                decltype(a.extract(0)) temp;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    temp = decltype(a.extract(0))(1.0)/std::sqrt(a[i]);
                    a.insert(i, temp);
                }
                return a;
            }
            // MRSQRTA
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE & rsqrtAssign(MASK_TYPE const & mask, VEC_TYPE & a) {
                UME_EMULATION_WARNING();
                decltype(a.extract(0)) temp;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    temp = decltype(a.extract(0))(1.0)/std::sqrt(a[i]);
                    if(mask[i] == true) a.insert(i, temp);
                }
                return a;
            }
            
            // POWV
            template<typename VEC_TYPE>
            UME_FORCE_INLINE VEC_TYPE pow(VEC_TYPE const & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::pow(a[i], b[i]));
                }
                return retval;
            }

            // MPOWV
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE pow(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
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
            UME_FORCE_INLINE VEC_TYPE pows(VEC_TYPE const & a, SCALAR_TYPE b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::pow(a[i], b));
                }
                return retval;
            }

            // MPOWS
            template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE pows(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
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
            UME_FORCE_INLINE VEC_TYPE round(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::round(a[i]));
                }
                return retval;
            }
            
            // MROUND
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE round(MASK_TYPE const & mask, VEC_TYPE const & a) {
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
            UME_FORCE_INLINE INT_VEC_TYPE truncToInt(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                INT_VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, decltype(retval.extract(0))(std::trunc(a[i])));
                }
                return retval;
            }
            
            // MTRUNC
            template<typename VEC_TYPE, typename INT_VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE INT_VEC_TYPE truncToInt(MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                INT_VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, decltype(retval.extract(0))(std::trunc(a[i])));
                    else retval.insert(i, 0);
                }
                return retval;
            }
            
            // FLOOR
            template<typename VEC_TYPE>
            UME_FORCE_INLINE VEC_TYPE floor(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length();i++) {
                    retval.insert(i, std::floor(a[i]));
                }
                return retval;
            }

            // MFLOOR
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE floor(MASK_TYPE const & mask, VEC_TYPE const & a) {
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
            UME_FORCE_INLINE VEC_TYPE ceil(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length();i++) {
                    retval.insert(i, std::ceil(a[i]));
                }
                return retval;
            }

            // MCEIL
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE ceil(MASK_TYPE const & mask, VEC_TYPE const & a) {
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
            UME_FORCE_INLINE VEC_TYPE fmuladd(VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (a[i]*b[i]) + c[i]);
                }
                return retval;
            }

            // MFMULADDV
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE fmuladd(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
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
            UME_FORCE_INLINE VEC_TYPE faddmul(VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (a[i] + b[i]) * c[i]);
                }
                return retval;
            }

            // MFADDMULV
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE faddmul(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
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
            UME_FORCE_INLINE VEC_TYPE fmulsub(VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (a[i]*b[i]) - c[i]);
                }
                return retval;
            }

            // MFMULSUBV
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE fmulsub(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
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
            UME_FORCE_INLINE VEC_TYPE fsubmul(VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (a[i] - b[i]) * c[i]);
                }
                return retval;
            }

            // MFSUBMULV
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE fsubmul(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
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
            UME_FORCE_INLINE MASK_TYPE isfin(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                MASK_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::isfinite(a[i]));
                }
                return retval;
            }

            // ISINF
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE MASK_TYPE isinf(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                MASK_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::isinf(a[i]));
                }
                return retval;
            }

            // ISAN
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE MASK_TYPE isan(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                MASK_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (!std::isnan(a[i]) && !std::isinf(a[i])));
                }
                return retval;
            }

            // ISNAN
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE MASK_TYPE isnan(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                MASK_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::isnan(a[i]));
                }
                return retval;
            }

            // ISNORM
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE MASK_TYPE isnorm(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                MASK_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::isnormal(a[i]));
                }
                return retval;
            }

            // ISSUB
            template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE MASK_TYPE issub(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                MASK_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    bool isZero = (a[i] == SCALAR_TYPE(0.0));
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
            UME_FORCE_INLINE MASK_TYPE iszero(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                MASK_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (a[i] == (decltype(a.extract(0))(0.0))));
                }
                return retval;
            }

            // ISZEROSUB
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE MASK_TYPE iszerosub(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                MASK_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    bool isZero = (a[i] == (decltype(a.extract(0))(0.0)));
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
            UME_FORCE_INLINE VEC_TYPE exp (VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::exp(a[i]));
                }
                return retval;
            }

            // MEXP
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE exp (MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (mask[i] == true) ? std::exp(a[i]) : a[i]);
                }
                return retval;
            }

            // SIN
            template<typename VEC_TYPE>
            UME_FORCE_INLINE VEC_TYPE sin (VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::sin(a[i]));
                }
                return retval;
            }

            // MSIN
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE sin (MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (mask[i] == true) ? std::sin(a[i]) : a[i]);
                }
                return retval;
            }

            // COS
            template<typename VEC_TYPE>
            UME_FORCE_INLINE VEC_TYPE cos (VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::cos(a[i]));
                }
                return retval;
            }

            // MCOS
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE cos (MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, (mask[i] == true) ? std::cos(a[i]) : a[i]);
                }
                return retval;
            }

            // TAN
            template<typename VEC_TYPE>
            UME_FORCE_INLINE VEC_TYPE tan (VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::tan(a[i]));
                }
                return retval;
            }

            // MTAN
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE tan (MASK_TYPE const & mask, VEC_TYPE const & a) {
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
            UME_FORCE_INLINE VEC_TYPE ctan (VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, decltype(retval.extract(0))(1.0)/std::tan(a[i]));
                }
                return retval;
            }

            // MCTAN
            template<typename VEC_TYPE, typename MASK_TYPE>
            UME_FORCE_INLINE VEC_TYPE ctan (MASK_TYPE const & mask, VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for(uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    if(mask[i] == true) retval.insert(i, decltype(retval.extract(0))(1.0)/std::tan(a[i]));
                    else retval.insert(i, a[i]);
                }
                return retval;
            }

            // ATAN
            template<typename VEC_TYPE>
            UME_FORCE_INLINE VEC_TYPE atan(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::atan(a[i]));
                }
                return retval;
            }

            // ATAN2
            template<typename VEC_TYPE>
            UME_FORCE_INLINE VEC_TYPE atan2(VEC_TYPE const & a, VEC_TYPE const & b) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::atan2(a[i], b[i]));
                }
                return retval;
            }

            // LOG
            template<typename VEC_TYPE>
            UME_FORCE_INLINE VEC_TYPE log(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::log(a.extract(i)));
                }
                return retval;
            }

            // LOG10
            template<typename VEC_TYPE>
            UME_FORCE_INLINE VEC_TYPE log10(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::log10(a.extract(i)));
                }
                return retval;
            }

            // LOG2
            template<typename VEC_TYPE>
            UME_FORCE_INLINE VEC_TYPE log2(VEC_TYPE const & a) {
                UME_EMULATION_WARNING();
                VEC_TYPE retval;
                for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                    retval.insert(i, std::log2(a.extract(i)));
                }
                return retval;
            }

        } // UME::SIMD::EMULATED_FUNCTIONS::MATH
    } // namespace UME::SIMD::EMULATED_FUNCTIONS
    
    // **********************************************************************
    // *
    // *  Declaration of IndexVectorInterface class
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
        UME_FORCE_INLINE bool extract(uint32_t index);
        // EXTRACT
        UME_FORCE_INLINE bool operator[] (uint32_t index);
        // INSERT
        UME_FORCE_INLINE void insert(uint32_t index, uint32_t value);

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
        UME_FORCE_INLINE ScalarTypeWrapper & operator=(ScalarTypeWrapper const & x){
            mValue = x.mValue;
            return *this;
        }*/

        // Also define a non-modifying access operator
        UME_FORCE_INLINE SCALAR_TYPE operator[] (uint32_t index) const { return mValue; }
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
            for(unsigned int i = 0; i < VEC_LEN; i++) { reg[i] = 0; }
        }

        SIMDVecEmuRegister(SCALAR_TYPE x) {
            UME_EMULATION_WARNING();
            for(unsigned int i = 0; i < VEC_LEN; i++) { reg[i] = x; }
        }

        SIMDVecEmuRegister(SIMDVecEmuRegister const & x) {
            UME_EMULATION_WARNING();
            for(unsigned int i = 0; i < VEC_LEN; i++) { reg[i] = x.reg[i]; }
        }

        // Also define a non-modifying access operator
        UME_FORCE_INLINE SCALAR_TYPE operator[] (uint32_t index) const { 
            SCALAR_TYPE temp = reg[index];    
            return temp; 
        }
            
        UME_FORCE_INLINE void insert(uint32_t index, SCALAR_TYPE value){
            reg[index] = value; 
        }
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
        UME_FORCE_INLINE bool extract(uint32_t index);
        // EXTRACT
        UME_FORCE_INLINE bool operator[] (uint32_t index);
        // INSERT
        UME_FORCE_INLINE void insert(uint32_t index, bool value);

    protected:
        ~SIMDMaskBaseInterface() {}

    public:
        // LENGTH
        constexpr static uint32_t length() { return MASK_LEN; }

        // ALIGNMENT
        constexpr static int alignment() { return MASK_LEN*sizeof(MASK_BASE_TYPE); }

        // LOAD
        UME_FORCE_INLINE DERIVED_MASK_TYPE & load(bool const * addr) {
            return EMULATED_FUNCTIONS::load<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE &>(*this), addr);
        }

        // LOADA
        UME_FORCE_INLINE DERIVED_MASK_TYPE & loada(bool const * addrAligned) {
            return EMULATED_FUNCTIONS::loadAligned<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE &>(*this), addrAligned);
        }

        // STORE
        UME_FORCE_INLINE bool* store(bool* addr) const {
            return EMULATED_FUNCTIONS::store<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE const &>(*this), addr);
        }

        // STOREA
        UME_FORCE_INLINE bool* storea(bool* addrAligned) const {
            return EMULATED_FUNCTIONS::storeAligned<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE const &>(*this), addrAligned);
        }

        // ASSIGNV
        UME_FORCE_INLINE DERIVED_MASK_TYPE & assign(DERIVED_MASK_TYPE const & maskOp) {
            return EMULATED_FUNCTIONS::assign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE & operator= (DERIVED_MASK_TYPE const & maskOp) {
            return assign(maskOp);
        }

        // MASSIGNV
        UME_FORCE_INLINE DERIVED_MASK_TYPE & assign(DERIVED_MASK_TYPE const & mask, DERIVED_MASK_TYPE const & maskOp) {
            return EMULATED_FUNCTIONS::assign<DERIVED_MASK_TYPE, DERIVED_MASK_TYPE>(mask, static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        }

        // ASSIGNS
        UME_FORCE_INLINE DERIVED_MASK_TYPE & assign(bool scalarOp) {
            return EMULATED_FUNCTIONS::assign<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE &>(*this), scalarOp);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE & operator= (bool scalarOp) {
            return assign(scalarOp);
        }

        // MASSIGNS
        UME_FORCE_INLINE DERIVED_MASK_TYPE & assign(DERIVED_MASK_TYPE const & mask, bool scalarOp) {
            return EMULATED_FUNCTIONS::assign<DERIVED_MASK_TYPE, bool, DERIVED_MASK_TYPE>(mask, static_cast<DERIVED_MASK_TYPE &>(*this), scalarOp);
        }

        // LANDV
        UME_FORCE_INLINE DERIVED_MASK_TYPE land(DERIVED_MASK_TYPE const & maskOp) const {
            return EMULATED_FUNCTIONS::binaryAnd<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), maskOp);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE operator& (DERIVED_MASK_TYPE const & maskOp) const {
            return land(maskOp);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE operator&& (DERIVED_MASK_TYPE const & maskOp) const {
            return land(maskOp);
        }

        // LANDS
        UME_FORCE_INLINE DERIVED_MASK_TYPE land(bool value) const {
            return EMULATED_FUNCTIONS::binaryAnd<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), value);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE operator& (bool value) const {
            return land(value);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE operator&& (bool value) const {
            return land(value);
        }

        // LANDVA
        UME_FORCE_INLINE DERIVED_MASK_TYPE & landa(DERIVED_MASK_TYPE const & maskOp) {
            return EMULATED_FUNCTIONS::binaryAndAssign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE & operator&= (DERIVED_MASK_TYPE const & maskOp) {
            return landa(maskOp);
        }

        // LANDSA
        UME_FORCE_INLINE DERIVED_MASK_TYPE & landa(bool value) {
            return EMULATED_FUNCTIONS::binaryAndAssign<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE &>(*this), value);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE & operator&= (bool value) {
            return landa(value);
        }

        // LORV
        UME_FORCE_INLINE DERIVED_MASK_TYPE lor(DERIVED_MASK_TYPE const & maskOp) const {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), maskOp);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE operator| (DERIVED_MASK_TYPE const & maskOp) const {
            return lor(maskOp);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE operator|| (DERIVED_MASK_TYPE const & maskOp) const {
            return lor(maskOp);
        }

        // LORS
        UME_FORCE_INLINE DERIVED_MASK_TYPE lor(bool value) const {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), value);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE operator| (bool value) const {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE const &>(*this), value);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE operator|| (bool value) const {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE const &>(*this), value);
        }

        // LORVA
        UME_FORCE_INLINE DERIVED_MASK_TYPE & lora(DERIVED_MASK_TYPE const & maskOp) {
            return EMULATED_FUNCTIONS::binaryOrAssign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE & operator|= (DERIVED_MASK_TYPE const & maskOp) {
            return lora(maskOp);
        }

        // LORSA
        UME_FORCE_INLINE DERIVED_MASK_TYPE & lora(bool value) {
            return EMULATED_FUNCTIONS::binaryOrAssign<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE &>(*this), value);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE & operator|= (bool value) {
            return lora(value);
        }

        // LXORV
        UME_FORCE_INLINE DERIVED_MASK_TYPE lxor(DERIVED_MASK_TYPE const & maskOp) const {
            return EMULATED_FUNCTIONS::binaryXor<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), maskOp);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE operator^ (DERIVED_MASK_TYPE const & maskOp) const {
            return lxor(maskOp);
        }

        // LXORS
        UME_FORCE_INLINE DERIVED_MASK_TYPE lxor(bool value) const {
            return EMULATED_FUNCTIONS::binaryXor<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE const &>(*this), value);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE operator^ (bool value) const {
            return lxor(value);
        }

        // LXORVA
        UME_FORCE_INLINE DERIVED_MASK_TYPE & lxora(DERIVED_MASK_TYPE const & maskOp) {
            return EMULATED_FUNCTIONS::binaryXorAssign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE & operator^= (DERIVED_MASK_TYPE const & maskOp) {
            return lxora(maskOp);
        }

        // LXORSA
        UME_FORCE_INLINE DERIVED_MASK_TYPE & lxora(bool value) {
            return EMULATED_FUNCTIONS::binaryXorAssign<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE &>(*this), value);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE & operator^= (bool value) {
            return lxora(value);
        }

        // LNOT
        UME_FORCE_INLINE DERIVED_MASK_TYPE lnot () const {
            return EMULATED_FUNCTIONS::logicalNot<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this));
        }
        
        UME_FORCE_INLINE DERIVED_MASK_TYPE operator!() const {
            return lnot();
        }

        // LNOTA
        UME_FORCE_INLINE DERIVED_MASK_TYPE & lnota () {
            return EMULATED_FUNCTIONS::logicalNotAssign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this));
        }

        // CMPEQV
        UME_FORCE_INLINE DERIVED_MASK_TYPE cmpeq(DERIVED_MASK_TYPE const & b) const {
            return EMULATED_FUNCTIONS::isEqual<DERIVED_MASK_TYPE, DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE operator== (DERIVED_MASK_TYPE const & b) const {
            return cmpeq(b);
        }

        // CMPEQS
        UME_FORCE_INLINE DERIVED_MASK_TYPE cmpeq(bool b) const {
            return EMULATED_FUNCTIONS::isEqual<DERIVED_MASK_TYPE, DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE operator== (bool b) const {
            return cmpeq(b);
        }

        // CMPNEV
        UME_FORCE_INLINE DERIVED_MASK_TYPE cmpne(DERIVED_MASK_TYPE const & b) const {
            return EMULATED_FUNCTIONS::isNotEqual<DERIVED_MASK_TYPE, DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE operator!= (DERIVED_MASK_TYPE const & b) const {
            return cmpne(b);
        }

        // CMPNES
        UME_FORCE_INLINE DERIVED_MASK_TYPE cmpne(bool b) const {
            return EMULATED_FUNCTIONS::isNotEqual<DERIVED_MASK_TYPE, DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_MASK_TYPE operator!= (bool b) const {
            return  cmpne(b);
        }

        // HLAND
        UME_FORCE_INLINE bool hland() const {
            return EMULATED_FUNCTIONS::reduceLogicalAnd<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this));
        }

        // HLOR
        UME_FORCE_INLINE bool hlor() const {
            return EMULATED_FUNCTIONS::reduceLogicalOr<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this));
        }

        // HLXOR
        UME_FORCE_INLINE bool hlxor() const {
            return EMULATED_FUNCTIONS::reduceLogicalXor<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this));
        }

        // CMPEV
        UME_FORCE_INLINE bool cmpe(DERIVED_MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::isExact<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), mask);
        }

        // CMPES
        UME_FORCE_INLINE bool cmpe(bool b) const {
            return EMULATED_FUNCTIONS::isExact<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), DERIVED_MASK_TYPE(b));
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
    template<class VEC_TYPE, class SCALAR_TYPE, class MASK_TYPE>
    class IntermediateMask {
    public:
        // MASSIGNV
        UME_FORCE_INLINE void operator=(VEC_TYPE const & vecRhs) const {
            mVecRef.assign(mMaskRef, vecRhs);
        }

        // MASSIGNS
        UME_FORCE_INLINE void operator=(SCALAR_TYPE scalarRhs) const {
            mVecRef.assign(mMaskRef, scalarRhs);
        }

        // MADDVA
        UME_FORCE_INLINE void operator+=(VEC_TYPE const & vecRhs) const {
            mVecRef.adda(mMaskRef, vecRhs);
        }

        // MADDSA
        UME_FORCE_INLINE void operator+=(SCALAR_TYPE scalarRhs) const {
            mVecRef.adda(mMaskRef, scalarRhs);
        }

        // MSUBVA
        UME_FORCE_INLINE void operator-= (VEC_TYPE const & vecRhs) const {
            mVecRef.suba(mMaskRef, vecRhs);
        }

        // MSUBSA
        UME_FORCE_INLINE void operator-=(SCALAR_TYPE scalarRhs) const {
            mVecRef.suba(mMaskRef, scalarRhs);
        }

        // MMULVA
        UME_FORCE_INLINE void operator*= (VEC_TYPE const & vecRhs) const {
            mVecRef.mula(mMaskRef, vecRhs);
        }

        // MMULSA
        UME_FORCE_INLINE void operator*=(SCALAR_TYPE scalarRhs) const {
            mVecRef.mula(mMaskRef, scalarRhs);
        }

        // MDIVVA
        UME_FORCE_INLINE void operator/= (VEC_TYPE const & vecRhs) const {
            mVecRef.diva(mMaskRef, vecRhs);
        }

        // MDIVSA
        UME_FORCE_INLINE void operator/=(SCALAR_TYPE scalarRhs) const {
            mVecRef.diva(mMaskRef, scalarRhs);
        }

        // MBANDVA
        UME_FORCE_INLINE void operator&= (VEC_TYPE const & vecRhs) const {
            mVecRef.banda(mMaskRef, vecRhs);
        }

        // MBANDSA
        UME_FORCE_INLINE void operator&=(SCALAR_TYPE scalarRhs) const {
            mVecRef.banda(mMaskRef, scalarRhs);
        }

        // MBORVA
        UME_FORCE_INLINE void operator|= (VEC_TYPE const & vecRhs) const {
            mVecRef.bora(mMaskRef, vecRhs);
        }

        // MBORSA
        UME_FORCE_INLINE void operator|=(SCALAR_TYPE scalarRhs) const {
            mVecRef.bora(mMaskRef, scalarRhs);
        }

        // MBXORVA
        UME_FORCE_INLINE void operator^= (VEC_TYPE const & vecRhs) const {
            mVecRef.bxora(mMaskRef, vecRhs);
        }

        // MBXORSA
        UME_FORCE_INLINE void operator^=(SCALAR_TYPE scalarRhs) const {
            mVecRef.bxora(mMaskRef, scalarRhs);
        }

        // This object should be only constructible by the
        // vector type using it.
        IntermediateMask();
        IntermediateMask(IntermediateMask const &);
        IntermediateMask & operator= (IntermediateMask const &); 

        explicit IntermediateMask(uint32_t);
    private:
        friend VEC_TYPE;

        UME_FORCE_INLINE explicit IntermediateMask(MASK_TYPE const & mask, VEC_TYPE & vec) : mMaskRef(mask), mVecRef(vec) {}

        MASK_TYPE const & mMaskRef;
        VEC_TYPE & mVecRef;
    };

    // **********************************************************************
    // *
    // *  Declaration of IntermediateIndex class 
    // *
    // *    This class is a helper class used in assignment version of
    // *    operator[SCALAR]. This object is not copyable and can only be created
    // *    from its vector type (VEC_TYPE) for temporary use. It's purpose is
    // *    to allow LHS assignments to expressions of form:
    // *
    // *     <vec>[index] <assignment_operator> <RHS scalar value>
    // *
    // **********************************************************************
    template<class VEC_TYPE, class SCALAR_TYPE>
    class IntermediateIndex {
    public:
        // MASSIGNS
        UME_FORCE_INLINE void operator= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, scalarRhs);
        }

        UME_FORCE_INLINE void operator+= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, mVecRef_RW[mIndexRef] + scalarRhs);
        }

        UME_FORCE_INLINE void operator-= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, mVecRef_RW[mIndexRef] - scalarRhs);
        }

        UME_FORCE_INLINE void operator*= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, mVecRef_RW[mIndexRef] * scalarRhs);
        }

        UME_FORCE_INLINE void operator/= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, mVecRef_RW[mIndexRef] / scalarRhs);
        }

        UME_FORCE_INLINE void operator&= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, mVecRef_RW[mIndexRef] & scalarRhs);
        }

        UME_FORCE_INLINE void operator|= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, mVecRef_RW[mIndexRef] | scalarRhs);
        }

        UME_FORCE_INLINE void operator^= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, mVecRef_RW[mIndexRef] ^ scalarRhs);
        }

        UME_FORCE_INLINE operator SCALAR_TYPE() { return mVecRef_RW.extract(mIndexRef); }

        // Comparison operators accept any type of scalar to allow mixing 
        // scalar types.
        template<typename T>
        UME_FORCE_INLINE bool operator==(T const & rhs) { 
            return mVecRef_RW.extract(mIndexRef) == rhs;
        }
        UME_FORCE_INLINE bool operator== (IntermediateIndex const & x) {
            return mVecRef_RW.extract(mIndexRef) ==
                x.mVecRef_RW.extract(mIndexRef);
        }
        template<typename T>
        UME_FORCE_INLINE bool operator!=(T const & rhs) {
            return mVecRef_RW.extract(mIndexRef) != rhs;
        }
        UME_FORCE_INLINE bool operator!= (IntermediateIndex const & x) {
            return mVecRef_RW.extract(mIndexRef) !=
                x.mVecRef_RW.extract(mIndexRef);
        }
        template<typename T>
        UME_FORCE_INLINE SCALAR_TYPE operator+ (T const & x) {
            return mVecRef_RW.extract(mIndexRef) + SCALAR_TYPE(x);
        }
        UME_FORCE_INLINE SCALAR_TYPE operator+ (IntermediateIndex const & x) {
            return mVecRef_RW.extract(mIndexRef) +
                x.mVecRef_RW.extract(mIndexRef);
        }
        template<typename T>
        UME_FORCE_INLINE SCALAR_TYPE operator- (T const & x) {
            return mVecRef_RW.extract(mIndexRef) - SCALAR_TYPE(x);
        }
        UME_FORCE_INLINE SCALAR_TYPE operator- (IntermediateIndex const & x) {
            return mVecRef_RW.extract(mIndexRef) -
                x.mVecRef_RW.extract(mIndexRef);
        }
        template<typename T>
        UME_FORCE_INLINE SCALAR_TYPE operator* (T const & x) {
            return mVecRef_RW.extract(mIndexRef) * SCALAR_TYPE(x);
        }
        UME_FORCE_INLINE SCALAR_TYPE operator* (IntermediateIndex const & x) {
            return mVecRef_RW.extract(mIndexRef) *
                x.mVecRef_RW.extract(mIndexRef);
        }
        template<typename T>
        UME_FORCE_INLINE SCALAR_TYPE operator/ (T const & x) {
            return mVecRef_RW.extract(mIndexRef) / SCALAR_TYPE(x);
        }
        UME_FORCE_INLINE SCALAR_TYPE operator/ (IntermediateIndex const & x) {
            return mVecRef_RW.extract(mIndexRef) /
                x.mVecRef_RW.extract(mIndexRef);
        }
        template<typename T>
        UME_FORCE_INLINE SCALAR_TYPE operator& (T const & x) {
            return mVecRef_RW.extract(mIndexRef) & SCALAR_TYPE(x);
        }
        UME_FORCE_INLINE SCALAR_TYPE operator& (IntermediateIndex const & x) {
            return mVecRef_RW.extract(mIndexRef) &
                x.mVecRef_RW.extract(mIndexRef);
        }
        template<typename T>
        UME_FORCE_INLINE SCALAR_TYPE operator| (T const & x) {
            return mVecRef_RW.extract(mIndexRef) | SCALAR_TYPE(x);
        }
        UME_FORCE_INLINE SCALAR_TYPE operator| (IntermediateIndex const & x) {
            return mVecRef_RW.extract(mIndexRef) |
                x.mVecRef_RW.extract(mIndexRef);
        }
        template<typename T>
        UME_FORCE_INLINE SCALAR_TYPE operator^ (T const & x) {
            return mVecRef_RW.extract(mIndexRef) ^ SCALAR_TYPE(x);
        }
        UME_FORCE_INLINE SCALAR_TYPE operator^ (IntermediateIndex const & x) {
            return mVecRef_RW.extract(mIndexRef) ^
                x.mVecRef_RW.extract(mIndexRef);
        }

    private:
        // This object should be only constructible by the
        // vector type using it.
        IntermediateIndex() {}
        IntermediateIndex(IntermediateIndex const & x) : mIndexRef(x.mIndexRef), mVecRef_RW(x.mVecRef_RW) {}
        IntermediateIndex & operator= (IntermediateIndex const & x) {
            mIndexRef = x.mIndexRef;
            mVecRef_RW = x.mVecRef_RW;
        }

        friend VEC_TYPE;

        UME_FORCE_INLINE explicit IntermediateIndex(uint32_t index, VEC_TYPE & vec) : mIndexRef(index), mVecRef_RW(vec) {}

        uint32_t mIndexRef;
        VEC_TYPE & mVecRef_RW;
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

        // PREFETCH0
        static UME_FORCE_INLINE void prefetch0(SCALAR_TYPE const *p) {
            // DO NOTHING!
        }

        // PREFETCH1
        static UME_FORCE_INLINE void prefetch1(SCALAR_TYPE const *p) {
            // DO NOTHING!
        }

        // PREFETCH2
        static UME_FORCE_INLINE void prefetch2(SCALAR_TYPE const *p) {
            // DO NOTHING!
        }

        // ASSIGNV
        UME_FORCE_INLINE DERIVED_VEC_TYPE & assign (DERIVED_VEC_TYPE const & src) {
            return EMULATED_FUNCTIONS::assign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), src);
        }
        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator= (DERIVED_VEC_TYPE const & src) {
            return assign(src);
        }

        // MASSIGNV
        UME_FORCE_INLINE DERIVED_VEC_TYPE & assign (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & src) {
            return EMULATED_FUNCTIONS::assign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), src);
        }

        // ASSIGNS
        UME_FORCE_INLINE DERIVED_VEC_TYPE & assign (SCALAR_TYPE value) {
            return EMULATED_FUNCTIONS::assign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), value);
        }
        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator= (SCALAR_TYPE value) {
            return assign(value);
        }

        // MASSIGNS
        UME_FORCE_INLINE DERIVED_VEC_TYPE & assign (MASK_TYPE const & mask, SCALAR_TYPE value) {
            return EMULATED_FUNCTIONS::assign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), value);
        }

        // LOAD
        UME_FORCE_INLINE DERIVED_VEC_TYPE & load (SCALAR_TYPE const *p) {
            return EMULATED_FUNCTIONS::load<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }

        // MLOAD
        UME_FORCE_INLINE DERIVED_VEC_TYPE & load (MASK_TYPE const & mask, SCALAR_TYPE const * p) {
            return EMULATED_FUNCTIONS::load<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }

        // LOADA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & loada (SCALAR_TYPE const * p) {
            return EMULATED_FUNCTIONS::loadAligned<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }

        // MLOADA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & loada (MASK_TYPE const & mask, SCALAR_TYPE const *p) {
            return EMULATED_FUNCTIONS::loadAligned<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }

        // STORE
        UME_FORCE_INLINE SCALAR_TYPE* store (SCALAR_TYPE* p) const {
            return EMULATED_FUNCTIONS::store<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), p);
        }

        // MSTORE
        UME_FORCE_INLINE SCALAR_TYPE* store (MASK_TYPE const & mask, SCALAR_TYPE* p) const {
            return EMULATED_FUNCTIONS::store<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), p);
        }

        // STOREA
        UME_FORCE_INLINE SCALAR_TYPE* storea (SCALAR_TYPE* p) const {
            return EMULATED_FUNCTIONS::store<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), p);
        }

        // MSTOREA
        UME_FORCE_INLINE SCALAR_TYPE* storea (MASK_TYPE const & mask, SCALAR_TYPE* p) const {
           return EMULATED_FUNCTIONS::store<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), p);
        }

        // EXTRACT
        // This method should be provided for all derived classes and cannot be defined
        // as generic.
        UME_FORCE_INLINE SCALAR_TYPE extract(uint32_t index) const;

        // INSERT
        // This method should be provided for all derived classes and cannot be defined
        // as generic.
        UME_FORCE_INLINE DERIVED_VEC_TYPE & insert(uint32_t index, SCALAR_TYPE value);

        // BLENDV
        UME_FORCE_INLINE DERIVED_VEC_TYPE blend (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::blend<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BLENDS
        UME_FORCE_INLINE DERIVED_VEC_TYPE blend (MASK_TYPE const & mask, SCALAR_TYPE b) const {
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
        UME_FORCE_INLINE DERIVED_VEC_TYPE add (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::add<DERIVED_VEC_TYPE> ( static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE operator+ (DERIVED_VEC_TYPE const & b) const {
            return add(b);
        }

        // MADDV
        UME_FORCE_INLINE DERIVED_VEC_TYPE add (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::add<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ADDS
        UME_FORCE_INLINE DERIVED_VEC_TYPE add (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::addScalar<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE operator+ (SCALAR_TYPE b) const {
            return add(b);
        }

        // MADDS
        UME_FORCE_INLINE DERIVED_VEC_TYPE add(MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::addScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ADDVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & adda (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::addAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator+= (DERIVED_VEC_TYPE const & b) {
            return adda(b);
        }

        // MADDVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & adda (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::addAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // ADDSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & adda (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::addAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator+= (SCALAR_TYPE b) {
            return adda(b);
        }

        // MADDSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & adda (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::addAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // SADDV
        UME_FORCE_INLINE DERIVED_VEC_TYPE sadd(DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::addSaturated<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        } 

        // MSADDV
        UME_FORCE_INLINE DERIVED_VEC_TYPE sadd(MASK_TYPE const & mask, DERIVED_VEC_TYPE b) const {
            return EMULATED_FUNCTIONS::addSaturated<DERIVED_VEC_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SADDS
        UME_FORCE_INLINE DERIVED_VEC_TYPE sadd(SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::addSaturatedScalar<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MSADDS
        UME_FORCE_INLINE DERIVED_VEC_TYPE sadd(MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::addSaturatedScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SADDVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & sadda(DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::addSaturatedAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MSADDVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & sadda(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::addSaturatedAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // SADDSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & sadda(SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::addSaturatedScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MSADDSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & sadda(MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::addSaturatedScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // POSTINC
        UME_FORCE_INLINE DERIVED_VEC_TYPE postinc () {
            return EMULATED_FUNCTIONS::postfixIncrement<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE operator++ (int) {
            return postinc();
        }

        // MPOSTINC
        UME_FORCE_INLINE DERIVED_VEC_TYPE postinc (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::postfixIncrement<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // PREFINC
        UME_FORCE_INLINE DERIVED_VEC_TYPE & prefinc () {
            return EMULATED_FUNCTIONS::prefixIncrement<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator++ () {
            return prefinc();
        }

        // MPREFINC
        UME_FORCE_INLINE DERIVED_VEC_TYPE & prefinc (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::prefixIncrement<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // SUBV
        UME_FORCE_INLINE DERIVED_VEC_TYPE sub (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::sub<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MSUBV
        UME_FORCE_INLINE DERIVED_VEC_TYPE sub (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::sub<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SUBS
        UME_FORCE_INLINE DERIVED_VEC_TYPE sub (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::subScalar<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MSUBS
        UME_FORCE_INLINE DERIVED_VEC_TYPE sub (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::subScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SUBVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & suba (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::subAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator-= (DERIVED_VEC_TYPE const & b) {
            return suba(b);
        }

        // MSUBVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & suba (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::subAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // SUBSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & suba (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::subAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator-= (SCALAR_TYPE b) {
            return suba(b);
        }

        // MSUBSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & suba (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::subAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // SSUBV
        UME_FORCE_INLINE DERIVED_VEC_TYPE ssub (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::subSaturated<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MSSUBV
        UME_FORCE_INLINE DERIVED_VEC_TYPE ssub (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::subSaturated<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SSUBS
        UME_FORCE_INLINE DERIVED_VEC_TYPE ssub (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::subSaturated<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MSSUBS
        UME_FORCE_INLINE DERIVED_VEC_TYPE ssub (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::subSaturated<DERIVED_VEC_TYPE, SCALAR_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SSUBVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & ssuba (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::subSaturatedAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MSSUBVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & ssuba (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::subSaturatedAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // SSUBSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & ssuba (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::subSaturatedScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MSSUBSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & ssuba (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::subSaturatedScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // SUBFROMV
        UME_FORCE_INLINE DERIVED_VEC_TYPE subfrom (DERIVED_VEC_TYPE const & a) const {
            return EMULATED_FUNCTIONS::subFrom<DERIVED_VEC_TYPE>(a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MSUBFROMV
        UME_FORCE_INLINE DERIVED_VEC_TYPE subfrom (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & a) const {
            return EMULATED_FUNCTIONS::subFrom<DERIVED_VEC_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SUBFROMS
        UME_FORCE_INLINE DERIVED_VEC_TYPE subfrom (SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::subFromScalar<DERIVED_VEC_TYPE, SCALAR_TYPE>(a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MSUBFROMS
        UME_FORCE_INLINE DERIVED_VEC_TYPE subfrom (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::subFromScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SUBFROMVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & subfroma (DERIVED_VEC_TYPE const & a) {
            return EMULATED_FUNCTIONS::subFromAssign<DERIVED_VEC_TYPE>(a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MSUBFROMVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & subfroma (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & a) {
            return EMULATED_FUNCTIONS::subFromAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // SUBFROMSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & subfroma (SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::subFromScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MSUBFROMSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & subfroma (MASK_TYPE const & mask, SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::subFromScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // POSTDEC
        UME_FORCE_INLINE DERIVED_VEC_TYPE postdec () {
            return EMULATED_FUNCTIONS::postfixDecrement<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE operator-- (int) {
            return postdec();
        }

        // MPOSTDEC
        UME_FORCE_INLINE DERIVED_VEC_TYPE postdec (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::postfixDecrement<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // PREFDEC
        UME_FORCE_INLINE DERIVED_VEC_TYPE & prefdec() {
            return EMULATED_FUNCTIONS::prefixDecrement<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }
        
        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator-- () {
            return prefdec();
        }

        // MPREFDEC
        UME_FORCE_INLINE DERIVED_VEC_TYPE & prefdec (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::prefixDecrement<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MULV
        UME_FORCE_INLINE DERIVED_VEC_TYPE mul (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::mult<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE operator* (DERIVED_VEC_TYPE const & b) const {
            return mul(b);
        }

        // MMULV
        UME_FORCE_INLINE DERIVED_VEC_TYPE mul (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::mult<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MULS
        UME_FORCE_INLINE DERIVED_VEC_TYPE mul (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::mult<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE operator* (SCALAR_TYPE b) const {
            return mul(b);
        }

        // MMULS
        UME_FORCE_INLINE DERIVED_VEC_TYPE mul (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::mult<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MULVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & mula (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::multAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator*= (DERIVED_VEC_TYPE const & b) {
            return mula(b);
        }

        // MMULVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & mula (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::multAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MULSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & mula (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::multAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator*= (SCALAR_TYPE b) {
            return mula(b);
        }

        // MMULSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & mula (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::multAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // DIVV
        UME_FORCE_INLINE DERIVED_VEC_TYPE div (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::div<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE operator/ (DERIVED_VEC_TYPE const & b) const {
            return div(b);
        }

        // MDIVV
        UME_FORCE_INLINE DERIVED_VEC_TYPE div (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::div<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // DIVS
        UME_FORCE_INLINE DERIVED_VEC_TYPE div (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::div<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE operator/ (SCALAR_TYPE b) const {
            return div(b);
        }

        // MDIVS
        UME_FORCE_INLINE DERIVED_VEC_TYPE div (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::div<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // DIVVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & diva (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::divAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator/= (DERIVED_VEC_TYPE const & b) {
            return diva(b);
        }

        // MDIVVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & diva (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::divAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // DIVSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & diva (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::divAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator/= (SCALAR_TYPE b) {
            return diva(b);
        }

        // MDIVSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & diva (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::divAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // RCP
        UME_FORCE_INLINE DERIVED_VEC_TYPE rcp () const {
            return EMULATED_FUNCTIONS::rcp<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MRCP
        UME_FORCE_INLINE DERIVED_VEC_TYPE rcp (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::rcp<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // RCPS
        UME_FORCE_INLINE DERIVED_VEC_TYPE rcp (SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::rcpScalar<DERIVED_VEC_TYPE, SCALAR_TYPE> (a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MRCPS
        UME_FORCE_INLINE DERIVED_VEC_TYPE rcp (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::rcpScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // RCPA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & rcpa () {
            return EMULATED_FUNCTIONS::rcpAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MRCPA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & rcpa (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::rcpAssign<DERIVED_VEC_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // RCPSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & rcpa (SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::rcpScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MRCPSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & rcpa (MASK_TYPE const & mask, SCALAR_TYPE a) {
            return EMULATED_FUNCTIONS::rcpScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // CMPEQV
        UME_FORCE_INLINE MASK_TYPE cmpeq (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::isEqual<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE MASK_TYPE operator== (DERIVED_VEC_TYPE const & b) const {
            return cmpeq(b);
        }

        // CMPEQS
        UME_FORCE_INLINE MASK_TYPE cmpeq (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::isEqual<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE MASK_TYPE operator== (SCALAR_TYPE b) const {
            return cmpeq(b);
        }

        // CMPNEV
        UME_FORCE_INLINE MASK_TYPE cmpne (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::isNotEqual<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE MASK_TYPE operator!= (DERIVED_VEC_TYPE const & b) const {
            return cmpne(b);
        }

        // CMPNES
        UME_FORCE_INLINE MASK_TYPE cmpne (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::isNotEqual<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE MASK_TYPE operator!= (SCALAR_TYPE b) const {
            return cmpne(b);
        }

        // CMPGTV
        UME_FORCE_INLINE MASK_TYPE cmpgt (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::isGreater<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE MASK_TYPE operator> (DERIVED_VEC_TYPE const & b) const {
            return cmpgt(b);
        }

        // CMPGTS
        UME_FORCE_INLINE MASK_TYPE cmpgt (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::isGreater<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE MASK_TYPE operator> (SCALAR_TYPE b) const {
            return cmpgt(b);
        }

        // CMPLTV
        UME_FORCE_INLINE MASK_TYPE cmplt (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::isLesser<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE MASK_TYPE operator< (DERIVED_VEC_TYPE const & b) const {
            return cmplt(b);
        }

        // CMPLTS
        UME_FORCE_INLINE MASK_TYPE cmplt (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::isLesser<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE MASK_TYPE operator< (SCALAR_TYPE b) const {
            return cmplt(b);
        }

        // CMPGEV
        UME_FORCE_INLINE MASK_TYPE cmpge (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::isGreaterEqual<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE MASK_TYPE operator>= (DERIVED_VEC_TYPE const & b) const {
            return cmpge(b);
        }

        // CMPGES
        UME_FORCE_INLINE MASK_TYPE cmpge (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::isGreaterEqual<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE MASK_TYPE operator>= (SCALAR_TYPE b) const {
            return cmpge(b);
        }

        // CMPLEV
        UME_FORCE_INLINE MASK_TYPE cmple (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::isLesserEqual<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE MASK_TYPE operator<= (DERIVED_VEC_TYPE const & b) const {
            return cmple(b);
        }

        // CMPLES
        UME_FORCE_INLINE MASK_TYPE cmple (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::isLesserEqual<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE MASK_TYPE operator<= (SCALAR_TYPE b) const {
            return cmple(b);
        }

        // CMPEV
        UME_FORCE_INLINE bool cmpe (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::isExact<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // CMPES
        UME_FORCE_INLINE bool cmpe (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::isExact<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), DERIVED_VEC_TYPE(b));
        }

        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
            return EMULATED_FUNCTIONS::unique<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HADD
        UME_FORCE_INLINE SCALAR_TYPE hadd () const {
            return EMULATED_FUNCTIONS::reduceAdd<SCALAR_TYPE, DERIVED_VEC_TYPE>( static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHADD
        UME_FORCE_INLINE SCALAR_TYPE hadd (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::reduceAdd<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &> (*this));
        }

        // HADDS
        UME_FORCE_INLINE SCALAR_TYPE hadd (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::reduceAdd<SCALAR_TYPE, DERIVED_VEC_TYPE>(b, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHADDS
        UME_FORCE_INLINE SCALAR_TYPE hadd (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::reduceAdd<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, b, static_cast<DERIVED_VEC_TYPE const &> (*this));
        }

        // HMUL
        UME_FORCE_INLINE SCALAR_TYPE hmul () const {
            return EMULATED_FUNCTIONS::reduceMult<SCALAR_TYPE, DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHMUL
        UME_FORCE_INLINE SCALAR_TYPE hmul (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::reduceMult<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HMULS
        UME_FORCE_INLINE SCALAR_TYPE hmul (SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::reduceMultScalar<SCALAR_TYPE, DERIVED_VEC_TYPE>(a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHMULS
        UME_FORCE_INLINE SCALAR_TYPE hmul (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::reduceMultScalar<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ******************************************************************
        // * Fused arithmetics
        // ******************************************************************

        // FMULADDV
        UME_FORCE_INLINE DERIVED_VEC_TYPE fmuladd(DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            return EMULATED_FUNCTIONS::MATH::fmuladd<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // MFMULADDV
        UME_FORCE_INLINE DERIVED_VEC_TYPE fmuladd(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            return EMULATED_FUNCTIONS::MATH::fmuladd<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // FMULSUBV
        UME_FORCE_INLINE DERIVED_VEC_TYPE fmulsub(DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            return EMULATED_FUNCTIONS::MATH::fmulsub<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // MFMULSUBV
        UME_FORCE_INLINE DERIVED_VEC_TYPE fmulsub(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            return EMULATED_FUNCTIONS::MATH::fmulsub<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // FADDMULV
        UME_FORCE_INLINE DERIVED_VEC_TYPE faddmul(DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            return EMULATED_FUNCTIONS::MATH::faddmul<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // MFADDMULV
        UME_FORCE_INLINE DERIVED_VEC_TYPE faddmul(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            return EMULATED_FUNCTIONS::MATH::faddmul<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }
        
        // FSUBMULV
        UME_FORCE_INLINE DERIVED_VEC_TYPE fsubmul(DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            return EMULATED_FUNCTIONS::MATH::fsubmul<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // MFSUBMULV
        UME_FORCE_INLINE DERIVED_VEC_TYPE fsubmul(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            return EMULATED_FUNCTIONS::MATH::fsubmul<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // ******************************************************************
        // * Additional math functions
        // ******************************************************************

        // MAXV
        UME_FORCE_INLINE DERIVED_VEC_TYPE max (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::MATH::max<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMAXV
        UME_FORCE_INLINE DERIVED_VEC_TYPE max (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::MATH::max<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MAXS
        UME_FORCE_INLINE DERIVED_VEC_TYPE max (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::MATH::maxScalar<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMAXS
        UME_FORCE_INLINE DERIVED_VEC_TYPE max (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::MATH::maxScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MAXVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & maxa (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::MATH::maxAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MMAXVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & maxa (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::MATH::maxAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MAXSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & maxa (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::MATH::maxScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MMAXSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & maxa (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::MATH::maxScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MINV
        UME_FORCE_INLINE DERIVED_VEC_TYPE min (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::MATH::min<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMINV
        UME_FORCE_INLINE DERIVED_VEC_TYPE min (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::MATH::min<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MINS
        UME_FORCE_INLINE DERIVED_VEC_TYPE min (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::MATH::minScalar<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMINS
        UME_FORCE_INLINE DERIVED_VEC_TYPE min (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::MATH::minScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MINVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & mina (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::MATH::minAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MMINVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & mina (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::MATH::minAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MINSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & mina (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::MATH::minScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MMINSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & mina (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::MATH::minScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // HMAX
        UME_FORCE_INLINE SCALAR_TYPE hmax () const {
            return EMULATED_FUNCTIONS::MATH::reduceMax<SCALAR_TYPE, DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHMAX
        UME_FORCE_INLINE SCALAR_TYPE hmax (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::reduceMax<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // IMAX
        UME_FORCE_INLINE uint32_t imax() const {
            return EMULATED_FUNCTIONS::MATH::indexMax<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MIMAX
        UME_FORCE_INLINE uint32_t imax(MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::indexMax<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HMIN
        UME_FORCE_INLINE SCALAR_TYPE hmin() const {
            return EMULATED_FUNCTIONS::MATH::reduceMin<SCALAR_TYPE, DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHMIN
        UME_FORCE_INLINE SCALAR_TYPE hmin(MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::reduceMin<SCALAR_TYPE, DERIVED_VEC_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // IMIN
        UME_FORCE_INLINE uint32_t imin() const {
            return EMULATED_FUNCTIONS::MATH::indexMin<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MIMIN
        UME_FORCE_INLINE uint32_t imin(MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::indexMin<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
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
        UME_FORCE_INLINE VEC_TYPE & operator= (const int8_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int16_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int32_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int64_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint8_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint16_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint32_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint64_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const float & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const double & x) { }
 
    public:
        // BANDV
        UME_FORCE_INLINE DERIVED_VEC_TYPE band (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::binaryAnd<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE operator& (DERIVED_VEC_TYPE const & b) const {
            return band(b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE operator&& (DERIVED_VEC_TYPE const & b) const {
            return band(b);
        }

        // MBANDV
        UME_FORCE_INLINE DERIVED_VEC_TYPE band (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::binaryAnd<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BANDS
        UME_FORCE_INLINE DERIVED_VEC_TYPE band (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::binaryAnd<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE operator& (SCALAR_TYPE b) const {
            return band(b);
        }
        /*
        UME_FORCE_INLINE DERIVED_VEC_TYPE operator&& (SCALAR_TYPE b) const {
            return band(b);
        }*/

        // MBANDS
        UME_FORCE_INLINE DERIVED_VEC_TYPE band (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::binaryAnd<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BANDVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & banda (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryAndAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator&= (DERIVED_VEC_TYPE const & b) {
            return banda(b);
        }

        // MBANDVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & banda (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryAndAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        // BANDSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & banda (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryAndAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator&= (bool b) {
            return banda(b);
        }

        // MBANDSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & banda (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryAndAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // BORV
        UME_FORCE_INLINE DERIVED_VEC_TYPE bor ( DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE operator| ( DERIVED_VEC_TYPE const & b) const {
            return bor(b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE operator|| (DERIVED_VEC_TYPE const & b) const {
            return bor(b);
        }

        // MBORV
        UME_FORCE_INLINE DERIVED_VEC_TYPE bor ( MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BORS
        UME_FORCE_INLINE DERIVED_VEC_TYPE bor (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE operator| (SCALAR_TYPE b) const {
            return bor(b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE operator|| (SCALAR_TYPE b) const {
            return bor(b);
        }

        // MBORS
        UME_FORCE_INLINE DERIVED_VEC_TYPE bor (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::binaryOr<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BORVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & bora (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryOrAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator|= (DERIVED_VEC_TYPE const & b) {
            return bora(b);
        }

        // MBORVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & bora (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryOrAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // BORSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & bora (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryOrAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator|= (SCALAR_TYPE b) {
            return bora(b);
        }

        // MBORSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & bora (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryOrAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // BXORV
        UME_FORCE_INLINE DERIVED_VEC_TYPE bxor (DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::binaryXor<DERIVED_VEC_TYPE> ( static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        UME_FORCE_INLINE DERIVED_VEC_TYPE operator^ (DERIVED_VEC_TYPE const & b) const {
            return bxor(b);
        }

        // MBXORV
        UME_FORCE_INLINE DERIVED_VEC_TYPE bxor (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::binaryXor<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BXORS
        UME_FORCE_INLINE DERIVED_VEC_TYPE bxor (SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::binaryXor<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE operator^ (SCALAR_TYPE b) const {
            return bxor(b);
        }

        // MBXORS
        UME_FORCE_INLINE DERIVED_VEC_TYPE bxor (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            return EMULATED_FUNCTIONS::binaryXor<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BXORVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & bxora (DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryXorAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator^= (DERIVED_VEC_TYPE const & b) {
            return bxora(b);
        }

        // MBXORVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & bxora (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::binaryXorAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // BXORSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & bxora (SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryXorAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FORCE_INLINE DERIVED_VEC_TYPE & operator^= (SCALAR_TYPE b) {
            return bxora(b);
        }

        // MBXORSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & bxora (MASK_TYPE const & mask, SCALAR_TYPE b) {
            return EMULATED_FUNCTIONS::binaryXorAssign<DERIVED_VEC_TYPE,SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // BNOT
        UME_FORCE_INLINE DERIVED_VEC_TYPE bnot () const {
            return EMULATED_FUNCTIONS::binaryNot<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
    
        UME_FORCE_INLINE DERIVED_VEC_TYPE operator~ () const {
            return bnot();
        }

        // MBNOT
        UME_FORCE_INLINE DERIVED_VEC_TYPE bnot (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::binaryNot<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // BNOTA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & bnota () {
            return EMULATED_FUNCTIONS::binaryNotAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MBNOTA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & bnota (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::binaryNotAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // HBAND
        UME_FORCE_INLINE SCALAR_TYPE hband ()const  {
            return EMULATED_FUNCTIONS::reduceBinaryAnd<SCALAR_TYPE, DERIVED_VEC_TYPE>( static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBAND
        UME_FORCE_INLINE SCALAR_TYPE hband (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::reduceBinaryAnd<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HBANDS
        UME_FORCE_INLINE SCALAR_TYPE hband (SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::reduceBinaryAndScalar<SCALAR_TYPE, DERIVED_VEC_TYPE>(a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBANDS
        UME_FORCE_INLINE SCALAR_TYPE hband (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::reduceBinaryAndScalar<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HBOR
        UME_FORCE_INLINE SCALAR_TYPE hbor () const {
            return EMULATED_FUNCTIONS::reduceBinaryOr<SCALAR_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBOR
        UME_FORCE_INLINE SCALAR_TYPE hbor (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::reduceBinaryOr<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HBORS
        UME_FORCE_INLINE SCALAR_TYPE hbor (SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::reduceBinaryOrScalar<SCALAR_TYPE, DERIVED_VEC_TYPE> (a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBORS
        UME_FORCE_INLINE SCALAR_TYPE hbor (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::reduceBinaryOrScalar<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // HBXOR
        UME_FORCE_INLINE SCALAR_TYPE hbxor () const {
            return EMULATED_FUNCTIONS::reduceBinaryXor<SCALAR_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBXOR
        UME_FORCE_INLINE SCALAR_TYPE hbxor (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::reduceBinaryXor<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HBXORS
        UME_FORCE_INLINE SCALAR_TYPE hbxor (SCALAR_TYPE a) const {
            return EMULATED_FUNCTIONS::reduceBinaryXorScalar<SCALAR_TYPE, DERIVED_VEC_TYPE> (a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBXORS
        UME_FORCE_INLINE SCALAR_TYPE hbxor (MASK_TYPE const & mask, SCALAR_TYPE a) const {
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
             typename SCALAR_UINT_TYPE,
             typename MASK_TYPE>
    class SIMDVecGatherScatterInterface
    {
        typedef SIMDVecGatherScatterInterface< 
            DERIVED_VEC_TYPE, 
            DERIVED_UINT_VEC_TYPE,
            SCALAR_TYPE,
            SCALAR_UINT_TYPE,
            MASK_TYPE> VEC_TYPE;

    private:
        // Forbid assignment-initialization of vector using scalar values
        // TODO: is this necessary?
        UME_FORCE_INLINE VEC_TYPE & operator= (const int8_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int16_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int32_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int64_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint8_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint16_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint32_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint64_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const float & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const double & x) { }
 
    public:
        // GATHERS
        UME_FORCE_INLINE DERIVED_VEC_TYPE & gather (SCALAR_TYPE * baseAddr, SCALAR_UINT_TYPE* indices) {
            return EMULATED_FUNCTIONS::gather<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }

        // MGATHERS
        UME_FORCE_INLINE DERIVED_VEC_TYPE & gather (MASK_TYPE const & mask, SCALAR_TYPE* baseAddr, SCALAR_UINT_TYPE* indices) {
            return EMULATED_FUNCTIONS::gather<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }

        // GATHERV
        UME_FORCE_INLINE DERIVED_VEC_TYPE & gather (SCALAR_TYPE * baseAddr, DERIVED_UINT_VEC_TYPE const & indices) {
            return EMULATED_FUNCTIONS::gather<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }

        // MGATHERV
        UME_FORCE_INLINE DERIVED_VEC_TYPE & gather (MASK_TYPE const & mask, SCALAR_TYPE* baseAddr, DERIVED_UINT_VEC_TYPE const & indices) {
            return EMULATED_FUNCTIONS::gather<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }

        // SCATTERS
        UME_FORCE_INLINE SCALAR_TYPE* scatter (SCALAR_TYPE* baseAddr, SCALAR_UINT_TYPE* indices) {
            return EMULATED_FUNCTIONS::scatter<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }

        // MSCATTERS
        UME_FORCE_INLINE SCALAR_TYPE*  scatter (MASK_TYPE const & mask, SCALAR_TYPE* baseAddr, SCALAR_UINT_TYPE* indices) {
            return EMULATED_FUNCTIONS::scatter<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }

        // SCATTERV
        UME_FORCE_INLINE SCALAR_TYPE*  scatter (SCALAR_TYPE* baseAddr, DERIVED_UINT_VEC_TYPE const & indices) {
            return EMULATED_FUNCTIONS::scatter<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }

        // MSCATTERV
        UME_FORCE_INLINE SCALAR_TYPE*  scatter (MASK_TYPE const & mask, SCALAR_TYPE* baseAddr, DERIVED_UINT_VEC_TYPE const & indices) {
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
        UME_FORCE_INLINE VEC_TYPE & operator= (const int8_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int16_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int32_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int64_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint8_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint16_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint32_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint64_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const float & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const double & x) { }
 
    public:
        // LSHV
        UME_FORCE_INLINE DERIVED_VEC_TYPE lsh (DERIVED_UINT_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::shiftBitsLeft<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MLSHV
        UME_FORCE_INLINE DERIVED_VEC_TYPE lsh (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::shiftBitsLeft<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // LSHS
        UME_FORCE_INLINE DERIVED_VEC_TYPE lsh (SCALAR_UINT_TYPE b) const {
            return EMULATED_FUNCTIONS::shiftBitsLeftScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MLSHS
        UME_FORCE_INLINE DERIVED_VEC_TYPE lsh (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) const {
            return EMULATED_FUNCTIONS::shiftBitsLeftScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // LSHVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & lsha (DERIVED_UINT_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::shiftBitsLeftAssign<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MLSHVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & lsha (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::shiftBitsLeftAssign<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // LSHSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & lsha (SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::shiftBitsLeftAssignScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MLSHSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & lsha (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::shiftBitsLeftAssignScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // RSHV
        UME_FORCE_INLINE DERIVED_VEC_TYPE rsh (DERIVED_UINT_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::shiftBitsRight<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MRSHV
        UME_FORCE_INLINE DERIVED_VEC_TYPE rsh (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::shiftBitsRight<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // RSHS
        UME_FORCE_INLINE DERIVED_VEC_TYPE rsh (SCALAR_UINT_TYPE b) const {
            return EMULATED_FUNCTIONS::shiftBitsRightScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MRSHS
        UME_FORCE_INLINE DERIVED_VEC_TYPE rsh (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) const {
            return EMULATED_FUNCTIONS::shiftBitsRightScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // RSHVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & rsha (DERIVED_UINT_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::shiftBitsRightAssign<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MRSHVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & rsha (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::shiftBitsRightAssign<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // RSHSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & rsha (SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::shiftBitsRightAssignScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MRSHSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & rsha (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::shiftBitsRightAssignScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // ROLV
        UME_FORCE_INLINE DERIVED_VEC_TYPE rol (DERIVED_UINT_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::rotateBitsLeft<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, SCALAR_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MROLV
        UME_FORCE_INLINE DERIVED_VEC_TYPE rol (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::rotateBitsLeft<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ROLS
        UME_FORCE_INLINE DERIVED_VEC_TYPE rol (SCALAR_UINT_TYPE b) const {
            return EMULATED_FUNCTIONS::rotateBitsLeftScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MROLS
        UME_FORCE_INLINE DERIVED_VEC_TYPE rol (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) const {
            return EMULATED_FUNCTIONS::rotateBitsLeftScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ROLVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & rola (DERIVED_UINT_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::rotateBitsLeftAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MROLVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & rola (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::rotateBitsLeftAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // ROLSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & rola (SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::rotateBitsLeftAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MROLSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & rola (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::rotateBitsLeftAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // RORV
        UME_FORCE_INLINE DERIVED_VEC_TYPE ror (DERIVED_UINT_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::rotateBitsRight<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MRORV
        UME_FORCE_INLINE DERIVED_VEC_TYPE ror (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::rotateBitsRight<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // RORS
        UME_FORCE_INLINE DERIVED_VEC_TYPE ror (SCALAR_UINT_TYPE b) const {
            return EMULATED_FUNCTIONS::rotateBitsRightScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MRORS
        UME_FORCE_INLINE DERIVED_VEC_TYPE ror (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) const {
            return EMULATED_FUNCTIONS::rotateBitsRightScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // RORVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE rora (DERIVED_UINT_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::rotateBitsRightAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MRORVA
        UME_FORCE_INLINE DERIVED_VEC_TYPE rora (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::rotateBitsRightAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // RORSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE rora (SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::rotateBitsRightAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MRORSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE rora (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) {
            return EMULATED_FUNCTIONS::rotateBitsRightAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
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
        UME_FORCE_INLINE VEC_TYPE & operator= (const int8_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int16_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int32_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int64_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint8_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint16_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint32_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint64_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const float & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const double & x) { }

    public:

        // PACK
        DERIVED_VEC_TYPE & pack(DERIVED_HALF_VEC_TYPE const & a, DERIVED_HALF_VEC_TYPE const & b) {
            return EMULATED_FUNCTIONS::pack<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE>(
                static_cast<DERIVED_VEC_TYPE &>(*this),
                static_cast<DERIVED_HALF_VEC_TYPE const &>(a),
                static_cast<DERIVED_HALF_VEC_TYPE const &>(b)
                );
        }

        // PACKLO
        DERIVED_VEC_TYPE & packlo(DERIVED_HALF_VEC_TYPE const & a) {
            return EMULATED_FUNCTIONS::packLow<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE>(
                static_cast<DERIVED_VEC_TYPE &>(*this),
                static_cast<DERIVED_HALF_VEC_TYPE const &>(a)
                );
        }

        // PACKHI
        DERIVED_VEC_TYPE & packhi(DERIVED_HALF_VEC_TYPE const & a) {
            return EMULATED_FUNCTIONS::packHigh<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE>(
                static_cast<DERIVED_VEC_TYPE &>(*this),
                static_cast<DERIVED_HALF_VEC_TYPE const &>(a)
                );
        }

        // UNPACK
        void unpack(DERIVED_HALF_VEC_TYPE & a, DERIVED_HALF_VEC_TYPE & b) const {
            EMULATED_FUNCTIONS::unpack<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE>(
                static_cast<DERIVED_VEC_TYPE const &>(*this),
                static_cast<DERIVED_HALF_VEC_TYPE &>(a),
                static_cast<DERIVED_HALF_VEC_TYPE &>(b)
                );
        }

        // UNPACKLO
        DERIVED_HALF_VEC_TYPE unpacklo() const {
            return EMULATED_FUNCTIONS::unpackLow<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE>(
                static_cast<DERIVED_VEC_TYPE const &> (*this)
                );
        }

        // UNPACKHI
        DERIVED_HALF_VEC_TYPE unpackhi() const {
            return EMULATED_FUNCTIONS::unpackHigh<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE>(
                static_cast<DERIVED_VEC_TYPE const &> (*this)
                );
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
        UME_FORCE_INLINE VEC_TYPE & operator= (const int8_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int16_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int32_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int64_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint8_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint16_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint32_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint64_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const float & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const double & x) { }
 
    public:

        // NEG
        UME_FORCE_INLINE DERIVED_VEC_TYPE neg () const {
            return EMULATED_FUNCTIONS::unaryMinus<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MNEG
        UME_FORCE_INLINE DERIVED_VEC_TYPE neg (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::unaryMinus<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // NEGA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & nega() {
            return EMULATED_FUNCTIONS::unaryMinusAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MNEGA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & nega(MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::unaryMinusAssign<DERIVED_VEC_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // ABS
        UME_FORCE_INLINE DERIVED_VEC_TYPE abs () const {
            return EMULATED_FUNCTIONS::MATH::abs<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MABS
        UME_FORCE_INLINE DERIVED_VEC_TYPE abs (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::abs<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ABSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE absa () {
            return EMULATED_FUNCTIONS::MATH::absAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MABSA
        UME_FORCE_INLINE DERIVED_VEC_TYPE absa (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::MATH::absAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
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
 
        //SCALAR_UINT_TYPE operator[] (SCALAR_UINT_TYPE index) const; // Declaration only! This operator has to be implemented in derived class.
        UME_FORCE_INLINE DERIVED_UINT_VEC_TYPE & insert(uint32_t index, SCALAR_UINT_TYPE value); // Declaration only! This operator has to be implemented in derived class.

    protected:
            
        // Making destructor protected prohibits this class from being instantiated. Effectively this class can only be used as a base class.
        ~SIMDVecUnsignedInterface() {};
    public:
        // SUBV
        UME_FORCE_INLINE DERIVED_UINT_VEC_TYPE operator- (DERIVED_UINT_VEC_TYPE const & b) const {
            return this->sub(b);
        }
        // SUBS
        UME_FORCE_INLINE DERIVED_UINT_VEC_TYPE operator- (SCALAR_UINT_TYPE b) const {
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
            SCALAR_UINT_TYPE,
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
        UME_FORCE_INLINE VEC_TYPE & operator= (const int8_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int16_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int32_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int64_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint8_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint16_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint32_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint64_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const float & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const double & x) { }
        
        SCALAR_TYPE operator[] (SCALAR_UINT_TYPE index) const; // Declaration only! This operator has to be implemented in derived class.
        UME_FORCE_INLINE DERIVED_VEC_TYPE & insert (uint32_t index, SCALAR_TYPE value); // Declaration only! This operator has to be implemented in derived class.
    protected:
            
        // Making destructor protected prohibits this class from being instantiated. Effectively this class can only be used as a base class.
        ~SIMDVecSignedInterface() {};
    public:
        // Everything already handled by other interface classes

        // SUBV
        UME_FORCE_INLINE DERIVED_VEC_TYPE operator- (DERIVED_VEC_TYPE const & b) const {
            return this->sub(b);
        }

        // SUBS
        UME_FORCE_INLINE DERIVED_VEC_TYPE operator- (SCALAR_TYPE const & b) const {
            return this->sub(b);
        }

        // NEG
        UME_FORCE_INLINE DERIVED_VEC_TYPE operator- () const {
            return this->neg();
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
            SCALAR_UINT_TYPE,
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
        UME_FORCE_INLINE VEC_TYPE & operator= (const int8_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int16_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int32_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const int64_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint8_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint16_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint32_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const uint64_t & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const float & x) { }
        UME_FORCE_INLINE VEC_TYPE & operator= (const double & x) { }
 
    protected:
        // Making destructor protected prohibits this class from being instantiated. Effectively this class can only be used as a base class.
        ~SIMDVecFloatInterface() {};
        
        SCALAR_FLOAT_TYPE operator[] (SCALAR_UINT_TYPE index) const; // Declaration only! This operator has to be implemented in derived class.
        UME_FORCE_INLINE DERIVED_VEC_TYPE & insert(uint32_t index, SCALAR_FLOAT_TYPE value); // Declaration only! This operator has to be implemented in derived class.
    public:

        // SUBV
        UME_FORCE_INLINE DERIVED_VEC_TYPE operator- (DERIVED_VEC_TYPE const & b) const {
            return this->sub(b);
        }
        
        // SUBS
        UME_FORCE_INLINE DERIVED_VEC_TYPE operator- (SCALAR_FLOAT_TYPE b) const {
            return this->sub(b);
        }
        // NEG
        UME_FORCE_INLINE DERIVED_VEC_TYPE operator- () const {
            return EMULATED_FUNCTIONS::unaryMinus<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ********************************************************************
        // * MATH FUNCTIONS
        // ********************************************************************

        // SQR
        UME_FORCE_INLINE DERIVED_VEC_TYPE sqr () const {
            return EMULATED_FUNCTIONS::MATH::sqr<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MSQR
        UME_FORCE_INLINE DERIVED_VEC_TYPE sqr (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::sqr<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SQRA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & sqra () {
            return EMULATED_FUNCTIONS::MATH::sqrAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MSQRA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & sqra (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::MATH::sqrAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // SQRT
        UME_FORCE_INLINE DERIVED_VEC_TYPE sqrt () const {
            return EMULATED_FUNCTIONS::MATH::sqrt<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // MSQRT
        UME_FORCE_INLINE DERIVED_VEC_TYPE sqrt (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::sqrt<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // SQRTA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & sqrta () {
            return EMULATED_FUNCTIONS::MATH::sqrtAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }
        
        // MSQRTA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & sqrta (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::MATH::sqrtAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // RSQRT
        UME_FORCE_INLINE DERIVED_VEC_TYPE rsqrt () const {
            return EMULATED_FUNCTIONS::MATH::rsqrt<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MRSQRT
        UME_FORCE_INLINE DERIVED_VEC_TYPE rsqrt (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::rsqrt<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SQRTA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & rsqrta () {
            return EMULATED_FUNCTIONS::MATH::rsqrtAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MSQRTA
        UME_FORCE_INLINE DERIVED_VEC_TYPE & rsqrta (MASK_TYPE const & mask) {
            return EMULATED_FUNCTIONS::MATH::rsqrtAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }
        
        // POWV
        // Disabled, see Issue #10
        //UME_FORCE_INLINE DERIVED_VEC_TYPE pow (DERIVED_VEC_TYPE const & b) const {
        //    return EMULATED_FUNCTIONS::MATH::pow<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        // }

        // MPOWV    
        // Disabled, see Issue #10    
        //UME_FORCE_INLINE DERIVED_VEC_TYPE pow (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
        //    return EMULATED_FUNCTIONS::MATH::pow<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        //}

        // POWS
        // Disabled, see Issue #10
        //UME_FORCE_INLINE DERIVED_VEC_TYPE pow (SCALAR_FLOAT_TYPE b) const {
        //    return EMULATED_FUNCTIONS::MATH::pows<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        //}

        // MPOWS
        // Disabled, see Issue #10
        //UME_FORCE_INLINE DERIVED_VEC_TYPE pow (MASK_TYPE const & mask, SCALAR_FLOAT_TYPE b) const {
        //    return EMULATED_FUNCTIONS::MATH::pows<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        //}

        // ROUND
        UME_FORCE_INLINE DERIVED_VEC_TYPE round () const {
            return EMULATED_FUNCTIONS::MATH::round<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // MROUND
        UME_FORCE_INLINE DERIVED_VEC_TYPE round (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::round<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // TRUNC
        UME_FORCE_INLINE DERIVED_VEC_INT_TYPE trunc () const {
            return EMULATED_FUNCTIONS::MATH::truncToInt<DERIVED_VEC_TYPE, DERIVED_VEC_INT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MTRUNC
        UME_FORCE_INLINE DERIVED_VEC_INT_TYPE trunc (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::truncToInt<DERIVED_VEC_TYPE, DERIVED_VEC_INT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // FLOOR
        UME_FORCE_INLINE DERIVED_VEC_TYPE floor () const {
            return EMULATED_FUNCTIONS::MATH::floor<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MFLOOR
        UME_FORCE_INLINE DERIVED_VEC_TYPE floor (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::floor<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // CEIL
        UME_FORCE_INLINE DERIVED_VEC_TYPE ceil () const {
            return EMULATED_FUNCTIONS::MATH::ceil<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MCEIL
        UME_FORCE_INLINE DERIVED_VEC_TYPE ceil (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::ceil<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISFIN
        UME_FORCE_INLINE MASK_TYPE isfin () const {
            return EMULATED_FUNCTIONS::MATH::isfin<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISINF
        UME_FORCE_INLINE MASK_TYPE isinf () const {
            return EMULATED_FUNCTIONS::MATH::isinf<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISAN
        UME_FORCE_INLINE MASK_TYPE isan () const {
            return EMULATED_FUNCTIONS::MATH::isan<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISNAN
        UME_FORCE_INLINE MASK_TYPE isnan () const {
            return EMULATED_FUNCTIONS::MATH::isnan<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISNORM
        UME_FORCE_INLINE MASK_TYPE isnorm() const {
            return EMULATED_FUNCTIONS::MATH::isnorm<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISSUB
        UME_FORCE_INLINE MASK_TYPE issub () const {
            return EMULATED_FUNCTIONS::MATH::issub<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISZERO
        UME_FORCE_INLINE MASK_TYPE iszero () const {
            return EMULATED_FUNCTIONS::MATH::iszero<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISZEROSUB
        UME_FORCE_INLINE MASK_TYPE iszerosub () const {
            return EMULATED_FUNCTIONS::MATH::iszerosub<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // EXP
        UME_FORCE_INLINE DERIVED_VEC_TYPE exp () const {
            return EMULATED_FUNCTIONS::MATH::exp<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MEXP
        UME_FORCE_INLINE DERIVED_VEC_TYPE exp (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::exp<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // LOG
        UME_FORCE_INLINE DERIVED_VEC_TYPE log() const {
            return EMULATED_FUNCTIONS::MATH::log<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // LOG10
        UME_FORCE_INLINE DERIVED_VEC_TYPE log10() const {
            return EMULATED_FUNCTIONS::MATH::log10<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // LOG2
        UME_FORCE_INLINE DERIVED_VEC_TYPE log2() const {
            return EMULATED_FUNCTIONS::MATH::log2<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SIN
        UME_FORCE_INLINE DERIVED_VEC_TYPE sin () const {
            return EMULATED_FUNCTIONS::MATH::sin<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MSIN
        UME_FORCE_INLINE DERIVED_VEC_TYPE sin (MASK_TYPE const & mask)const  {
            return EMULATED_FUNCTIONS::MATH::sin<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // COS
        UME_FORCE_INLINE DERIVED_VEC_TYPE cos () const {
            return EMULATED_FUNCTIONS::MATH::cos<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MCOS
        UME_FORCE_INLINE DERIVED_VEC_TYPE cos (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::cos<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SINCOS
        UME_FORCE_INLINE void sincos(DERIVED_VEC_TYPE & sinvec, DERIVED_VEC_TYPE & cosvec) const {
            sinvec = EMULATED_FUNCTIONS::MATH::sin<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
            cosvec = EMULATED_FUNCTIONS::MATH::cos<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MSINCOS
        UME_FORCE_INLINE void sincos(MASK_TYPE const & mask, DERIVED_VEC_TYPE & sinvec, DERIVED_VEC_TYPE & cosvec) const {
            sinvec = EMULATED_FUNCTIONS::MATH::sin<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
            cosvec = EMULATED_FUNCTIONS::MATH::cos<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // TAN
        UME_FORCE_INLINE DERIVED_VEC_TYPE tan () const {
            return EMULATED_FUNCTIONS::MATH::tan<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MTAN
        UME_FORCE_INLINE DERIVED_VEC_TYPE tan (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::tan<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // CTAN
        UME_FORCE_INLINE DERIVED_VEC_TYPE ctan () const {
            return EMULATED_FUNCTIONS::MATH::ctan<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MCTAN
        UME_FORCE_INLINE DERIVED_VEC_TYPE ctan (MASK_TYPE const & mask) const {
            return EMULATED_FUNCTIONS::MATH::ctan<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ATAN
        UME_FORCE_INLINE DERIVED_VEC_TYPE atan() const {
            return EMULATED_FUNCTIONS::MATH::atan<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ATAN2
        UME_FORCE_INLINE DERIVED_VEC_TYPE atan2(DERIVED_VEC_TYPE const & b) const {
            return EMULATED_FUNCTIONS::MATH::atan2<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

    };

    // This is just an experimental setup! Providing functions like this to handle interface
    // is possible although it will be pretty extensive in number of necessary declarations.
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE addv (VEC_TYPE const & src1, VEC_TYPE const & src2) {
        return src1.add(src2);
    }

    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & addva (VEC_TYPE & src1, VEC_TYPE const & src2) {
        return src1.addva(src2);
    }

    // How to restrict template parameter resolution to certain types only?
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE adds (float src1, VEC_TYPE const & src2) {
        return src2.add(src1);
    }
    
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE adds (VEC_TYPE const & src1, float src2) {
        return src1.add(src2);
    }
} // namespace UME::SIMD
} // namespace UME

#endif
