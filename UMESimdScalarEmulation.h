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

#ifndef UME_SIMD_SCALAR_EMULATION_H_
#define UME_SIMD_SCALAR_EMULATION_H_

#include "UMEInline.h"
#include "UMEBasicTypes.h"

#include <algorithm>
#include <array>

namespace UME
{
namespace SIMD
{
//   All functions in this namespace will have one purpose: emulation of single function in different backends.
//   Scalar emulation plugin has to emulate all of these features using scalar values either way. Spliting
//   Functionality implementation from class implementation will allow re-use of the operator functions for 
//   other backends. This will decrease overall amount of code, and remove potential, repeated errors in plugins.
namespace SCALAR_EMULATION
{
    // ASSIGN
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & assign(VEC_TYPE & dst, VEC_TYPE const & src) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            dst.insert(i, src[i]);
        }
        return dst;
    }

    // MASSIGN
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & assign(MASK_TYPE const & mask, VEC_TYPE & dst, VEC_TYPE const & src) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) dst.insert(i, src[i]);
        }
        return dst;
    }

    // ASSIGNS
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & assign(VEC_TYPE & dst, SCALAR_TYPE src) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            dst.insert(i, src);
        }
        return dst;
    }

    // MASSIGNS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & assign(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE src) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) dst.insert(i, src);
        }
        return dst;
    }

    // LOAD
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & load(VEC_TYPE & dst, SCALAR_TYPE const * p) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            dst.insert(i, p[i]);
        }
        return dst;
    }

    // MLOAD
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & load(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE const * p) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++)
        {
            if (mask[i] == true) dst.insert(i, p[i]);
        }
        return dst;
    }

    // LOADA
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & loadAligned(VEC_TYPE & dst, SCALAR_TYPE const * p) {
        UME_ALIGNMENT_CHECK(p, VEC_TYPE::alignment());
        return SCALAR_EMULATION::load<VEC_TYPE, SCALAR_TYPE>(dst, p);
    }

    // MLOADA
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & loadAligned(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE const * p) {
        UME_ALIGNMENT_CHECK(p, VEC_TYPE::alignment());
        return SCALAR_EMULATION::load<VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, dst, p);
    }

    // STORE
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE* store(VEC_TYPE const & src, SCALAR_TYPE * p) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++)
        {
            p[i] = src[i];
        }
        return p;
    }

    // MSTORE
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE* store(MASK_TYPE const & mask, VEC_TYPE const & src, SCALAR_TYPE * p) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++)
        {
            if (mask[i] == true) p[i] = src[i];
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
    
    // GATHER
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & gatheru(VEC_TYPE & dst, SCALAR_TYPE* base, uint32_t stride) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            dst.insert(i, base[i*stride]);
        }
        return dst;
    }

    // MGATHER
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & gatheru(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE* base, uint32_t stride) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) dst.insert(i, base[i*stride]);
        }
        return dst;
    }
    
    // GATHERS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename SCALAR_UINT_TYPE>
    UME_FORCE_INLINE VEC_TYPE & gather(VEC_TYPE & dst, SCALAR_TYPE* base, SCALAR_UINT_TYPE* indices) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            dst.insert(i, base[indices[i]]);
        }
        return dst;
    }

    // MGATHERS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & gather(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE* base, SCALAR_UINT_TYPE* indices) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) dst.insert(i, base[indices[i]]);
        }
        return dst;
    }

    // GATHERV
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & gather(VEC_TYPE & dst, SCALAR_TYPE* base, UINT_VEC_TYPE const & indices) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            dst.insert(i, base[indices[i]]);
        }
        return dst;
    }

    // MGATHERV
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & gather(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE* base, UINT_VEC_TYPE const & indices) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) dst.insert(i, base[indices[i]]);
        }
        return dst;
    }

    // SCATTERS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename SCALAR_UINT_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE* scatter(VEC_TYPE const & src, SCALAR_TYPE* base, SCALAR_UINT_TYPE* indices) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            base[indices[i]] = src[i];
        }
        return base;
    }

    // MSCATTERS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE* scatter(MASK_TYPE const & mask, VEC_TYPE const & src, SCALAR_TYPE* base, SCALAR_UINT_TYPE* indices) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) base[indices[i]] = src[i];
        }
        return base;
    }

    // SCATTERV
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE* scatter(VEC_TYPE const & src, SCALAR_TYPE* base, UINT_VEC_TYPE const & indices) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++)
        {
            base[indices[i]] = src[i];
        }
        return base;
    }

    // MSCATTERV
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE* scatter(MASK_TYPE const & mask, VEC_TYPE const & src, SCALAR_TYPE* base, UINT_VEC_TYPE const & indices) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++)
        {
            if (mask[i] == true) base[indices[i]] = src[i];
        }
        return base;
    }

    // PACK
    template<typename VEC_TYPE, typename VEC_HALF_TYPE>
    UME_FORCE_INLINE VEC_TYPE & pack(VEC_TYPE & dst, VEC_HALF_TYPE const & src1, VEC_HALF_TYPE const & src2) {
        for (uint32_t i = 0; i < VEC_HALF_TYPE::length(); i++) {
            dst.insert(i, src1[i]);
            dst.insert(i + VEC_HALF_TYPE::length(), src2[i]);
        }
        return dst;
    }

    // PACKLO
    template<typename VEC_TYPE, typename VEC_HALF_TYPE>
    UME_FORCE_INLINE VEC_TYPE & packLow(VEC_TYPE & dst, VEC_HALF_TYPE const & src1) {
        for (uint32_t i = 0; i < VEC_HALF_TYPE::length(); i++) {
            dst.insert(i, src1[i]);
        }
        return dst;
    }

    // PACKHI
    template<typename VEC_TYPE, typename VEC_HALF_TYPE>
    UME_FORCE_INLINE VEC_TYPE & packHigh(VEC_TYPE & dst, VEC_HALF_TYPE const & src1) {
        for (uint32_t i = VEC_HALF_TYPE::length(); i < VEC_TYPE::length(); i++) {
            dst.insert(i, src1[i - VEC_HALF_TYPE::length()]);
        }
        return dst;
    }

    // UNPACK
    template<typename VEC_TYPE, typename VEC_HALF_TYPE>
    UME_FORCE_INLINE void unpack(VEC_TYPE const & src, VEC_HALF_TYPE & dst1, VEC_HALF_TYPE & dst2) {
        uint32_t halfLength = VEC_HALF_TYPE::length();
        for (uint32_t i = 0; i < halfLength; i++) {
            dst1.insert(i, src[i]);
            dst2.insert(i, src[i + halfLength]);
        }
    }

    // UNPACKLO
    template<typename VEC_TYPE, typename VEC_HALF_TYPE>
    UME_FORCE_INLINE VEC_HALF_TYPE unpackLow(VEC_TYPE const & src) {
        VEC_HALF_TYPE retval;
        for (uint32_t i = 0; i < VEC_HALF_TYPE::length(); i++) {
            retval.insert(i, src[i]);
        }
        return retval;
    }

    // UNPACKHI
    template<typename VEC_TYPE, typename VEC_HALF_TYPE>
    UME_FORCE_INLINE VEC_HALF_TYPE unpackHigh(VEC_TYPE const & src) {
        VEC_HALF_TYPE retval;
        for (uint32_t i = 0; i < VEC_HALF_TYPE::length(); i++) {
            retval.insert(i, src[i + VEC_HALF_TYPE::length()]);
        }
        return retval;
    }

    // ADDV
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE add(VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] + b[i]);
        }
        return retval;
    }

    // MADDV
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE add(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, mask[i] ? a[i] + b[i] : a[i]);
        }
        return retval;
    }

    // ADDS
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE addScalar(VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] + b);
        }
        return retval;
    }

    // MADDS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE addScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, mask[i] ? a[i] + b : a[i]);
        }
        return retval;
    }

    // ADDVA
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & addAssign(VEC_TYPE & a, VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) { a.insert(i, (a[i] + b[i])); }
        return a;
    }

    // MADDVA
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & addAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, (a[i] + b[i]));
        }
        return a;
    }

    // ADDSA
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & addAssignScalar(VEC_TYPE & a, SCALAR_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, (a[i] + b));
        }
        return a;
    }

    // MADDSA
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & addAssignScalar(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, a[i] + b);
        }
        return a;
    }

    // SADDV
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE addSaturated(VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        decltype(a.extract(0)) temp = 0;
        // maximum value
        decltype(a.extract(0)) maxValue = std::numeric_limits<decltype(a.extract(0))>::max();
        decltype(a.extract(0)) minValue = std::numeric_limits<decltype(a.extract(0))>::min();
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
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
    UME_FORCE_INLINE VEC_TYPE addSaturated(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        decltype(a.extract(0)) temp = 0;
        // maximum value
        decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::max();
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) {
                temp = (a[i] >(satValue - b[i])) ? satValue : (a[i] + b[i]);
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
    UME_FORCE_INLINE VEC_TYPE addSaturatedScalar(VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        decltype(a.extract(0)) temp = 0;
        // maximum value
        decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::max();
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            temp = (a[i] >(satValue - b)) ? satValue : (a[i] + b);
            retval.insert(i, temp);
        }
        return retval;
    }

    // MSADDS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE addSaturatedScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        decltype(a.extract(0)) temp = 0;
        // maximum value
        decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::max();
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) {
                temp = (a[i] >(satValue - b)) ? satValue : (a[i] + b);
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
        decltype(a.extract(0)) temp = 0;
        // maximum value
        decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::max();
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            temp = (a[i] >(satValue - b[i])) ? satValue : (a[i] + b[i]);
            a.insert(i, temp);
        }
        return a;
    }

    // MSADDVA
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & addSaturatedAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
        decltype(a.extract(0)) temp = 0;
        // maximum value
        decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::max();
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) {
                temp = (a[i] >(satValue - b[i])) ? satValue : (a[i] + b[i]);
                a.insert(i, temp);
            }
        }
        return a;
    }

    // SADDSA
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & addSaturatedScalarAssign(VEC_TYPE & a, SCALAR_TYPE b) {
        decltype(a.extract(0)) temp = 0;
        // maximum value
        decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::max();
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            temp = (a[i] >(satValue - b)) ? satValue : (a[i] + b);
            a.insert(i, temp);
        }
        return a;
    }

    // MSADDSA
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & addSaturatedScalarAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
        decltype(a.extract(0)) temp = 0;
        // maximum value
        decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::max();
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) {
                temp = (a[i] >(satValue - b)) ? satValue : (a[i] + b);
                a.insert(i, temp);
            }
        }
        return a;
    }

    // POSTINC
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE postfixIncrement(VEC_TYPE & a) {
        VEC_TYPE retval = a;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, a[i] + 1);
        }
        return retval;
    }

    // MPOSTINC
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE postfixIncrement(MASK_TYPE const & mask, VEC_TYPE & a) {
        VEC_TYPE retval = a;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, a[i] + 1);
        }
        return retval;
    }

    // PREFINC
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & prefixIncrement(VEC_TYPE & a) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++)
        {
            a.insert(i, a[i] + 1);
        }
        return a;
    }

    // MPREFINC
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & prefixIncrement(MASK_TYPE const & mask, VEC_TYPE & a) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++)
        {
            if (mask[i] == true) a.insert(i, a[i] + 1);
        }
        return a;
    }

    // SUBV
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE sub(VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] - b[i]);
        }
        return retval;
    }

    // MSUBV
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE sub(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval.insert(i, a[i] - b[i]);
            else retval.insert(i, a[i]);
        }
        return retval;
    }

    // SUBS
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE subScalar(VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (a[i] - b));
        }
        return retval;
    }

    // MSUBS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE subScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval.insert(i, (a[i] - b));
            else retval.insert(i, a[i]);
        }
        return retval;
    }

    // SUBFROMV
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE subFrom(VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] - b[i]);
        }
        return retval;
    }

    // MSUBFROMV
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE subFrom(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval.insert(i, a[i] - b[i]);
            else retval.insert(i, a[i]);
        }
        return retval;
    }

    // SUBFROMS
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE subFromScalar(SCALAR_TYPE a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a - b[i]);
        }
        return retval;
    }

    // MSUBFROMS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE subFromScalar(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval.insert(i, a - b[i]);
            else retval.insert(i, a);
        }
        return retval;
    }

    // SUBFROMVA
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & subFromAssign(VEC_TYPE const & a, VEC_TYPE & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            b.insert(i, a[i] - b[i]);
        }
        return b;
    }

    // MSUBFROMVA
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & subFromAssign(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) b.insert(i, a[i] - b[i]);
            else b.insert(i, a[i]);
        }
        return b;
    }

    // SUBFROMSA
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & subFromScalarAssign(SCALAR_TYPE a, VEC_TYPE & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            b.insert(i, a - b[i]);
        }
        return b;
    }

    // MSUBFROMSA
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & subFromScalarAssign(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) b.insert(i, a - b[i]);
            else b.insert(i, a);
        }
        return b;
    }

    // NEG
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE unaryMinus(VEC_TYPE const & a) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, -a[i]);
        }
        return retval;
    }

    // MNEG
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE unaryMinus(MASK_TYPE const & mask, VEC_TYPE const & a) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval.insert(i, -a[i]);
            else retval.insert(i, a[i]);
        }
        return retval;
    }

    // NEGA
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & unaryMinusAssign(VEC_TYPE & a) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, -a[i]);
        }
        return a;
    }

    // MNEGA
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & unaryMinusAssign(MASK_TYPE const & mask, VEC_TYPE & a) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, -a[i]);
        }
        return a;
    }

    // SUBVA
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & subAssign(VEC_TYPE & dst, VEC_TYPE const & b)
    {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            dst.insert(i, dst[i] - b[i]);
        }
        return dst;
    }

    // MSUBVA
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & subAssign(MASK_TYPE const & mask, VEC_TYPE & dst, VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) dst.insert(i, dst[i] - b[i]);
        }
        return dst;
    }

    // SUBSA
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & subAssign(VEC_TYPE & dst, SCALAR_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            dst.insert(i, dst[i] - b);
        }
        return dst;
    }

    // MSUBSA
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & subAssign(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) dst.insert(i, dst[i] - b);
        }
        return dst;
    }

    // SSUBV
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE subSaturated(VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        decltype(a.extract(0)) temp = 0;
        // maximum value
        decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::min();
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            temp = (a[i] < (satValue + b[i])) ? satValue : (a[i] - b[i]);
            retval.insert(i, temp);
        }
        return retval;
    }

    // MSSUBV
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE subSaturated(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        decltype(a.extract(0)) temp = 0;
        // maximum value
        decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::min();
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) {
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
    UME_FORCE_INLINE VEC_TYPE subSaturated(VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        decltype(a.extract(0)) temp = 0;
        // maximum value
        decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::min();
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            temp = (a[i] < (satValue + b)) ? satValue : (a[i] - b);
            retval.insert(i, temp);
        }
        return retval;
    }

    // MSSUBS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE subSaturated(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        decltype(a.extract(0)) temp = 0;
        // maximum value
        decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::min();
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) {
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
    UME_FORCE_INLINE VEC_TYPE & subSaturatedAssign(VEC_TYPE & a, VEC_TYPE const & b) {
        decltype(a.extract(0)) temp = 0;
        // maximum value
        decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::min();
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            temp = (a[i] < (satValue + b[i])) ? satValue : (a[i] - b[i]);
            a.insert(i, temp);
        }
        return a;
    }

    // MSSUBV
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & subSaturatedAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
        decltype(a.extract(0)) temp = 0;
        // maximum value
        decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::min();
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) {
                temp = (a[i] < (satValue + b[i])) ? satValue : (a[i] - b[i]);
                a.insert(i, temp);
            }
        }
        return a;
    }

    // SSUBS
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & subSaturatedScalarAssign(VEC_TYPE & a, SCALAR_TYPE b) {
        decltype(a.extract(0)) temp = 0;
        // maximum value
        decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::min();
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            temp = (a[i] < (satValue + b)) ? satValue : (a[i] - b);
            a.insert(i, temp);
        }
        return a;
    }

    // MSSUBS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & subSaturatedScalarAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
        decltype(a.extract(0)) temp = 0;
        // maximum value
        decltype(a.extract(0)) satValue = std::numeric_limits<decltype(a.extract(0))>::min();
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) {
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
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, a[i] - 1);
        }
        return retval;
    }

    // MPOSTDEC
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE postfixDecrement(MASK_TYPE const & mask, VEC_TYPE & a) {
        VEC_TYPE retval = a;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, a[i] - 1);
        }
        return retval;
    }

    // PREFDEC
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & prefixDecrement(VEC_TYPE & a) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, a[i] - 1);
        }
        return a;
    }

    // MPREFDEC
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & prefixDecrement(MASK_TYPE const & mask, VEC_TYPE & a) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, a[i] - 1);
        }
        return a;
    }

    // MULV
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE mult(VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++)
        {
            retval.insert(i, a[i] * b[i]);
        }
        return retval;
    }

    // MMULV
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE mult(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] == true) ? a[i] * b[i] : a[i]);
        }
        return retval;
    }

    // MULS
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE mult(VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] * b);
        }
        return retval;
    }

    // MMULS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE mult(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] == true) ? a[i] * b : a[i]);
        }
        return retval;
    }

    // MULVA
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & multAssign(VEC_TYPE & dst, VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            dst.insert(i, dst[i] * b[i]);
        }
        return dst;
    }

    // MMULVA
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & multAssign(MASK_TYPE const & mask, VEC_TYPE & dst, VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) dst.insert(i, dst[i] * b[i]);
        }
        return dst;
    }

    // MULSA
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & multAssign(VEC_TYPE & dst, SCALAR_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            dst.insert(i, dst[i] * b);
        }
        return dst;
    }

    // MMULSA
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & multAssign(MASK_TYPE const & mask, VEC_TYPE & dst, SCALAR_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) dst.insert(i, dst[i] * b);
        }
        return dst;
    }

    // DIVV
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE div(VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] / b[i]);
        }
        return retval;
    }

    // MDIVV
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE div(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] == true) ? a[i] / b[i] : a[i]);
        }
        return retval;
    }

    // DIVS
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE div(VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] / b);
        }
        return retval;
    }

    // MDIVS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE div(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] == true) ? (a[i] / b) : a[i]);
        }
        return retval;
    }

    // REMV
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE reminder(VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] % b[i]);
        }
        return retval;
    }

    // MREMV
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE reminder(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] == true) ? (a[i] % b[i]) : a[i]);
        }
        return retval;
    }

    // REMS
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE reminder(VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] % b);
        }
        return retval;
    }

    // MREMS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE reminder(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] == true) ? (a[i] % b) : a[i]);
        }
        return retval;
    }

    // REMVA
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & reminderAssign(VEC_TYPE & a, VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, a[i] % b[i]);
        }
        return a;
    }

    // MREMVA
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & reminderAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, (mask[i] == true) ? (a[i] % b[i]) : a[i]);
        }
        return a;
    }

    // REMSA
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & reminderAssign(VEC_TYPE & a, SCALAR_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, a[i] % b);
        }
        return a;
    }

    // MREMSA
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE reminderAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, (mask[i] == true) ? (a[i] % b) : a[i]);
        }
        return a;
    }

    // RCP
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE div(SCALAR_TYPE a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a / b[i]);
        }
        return retval;
    }

    // MRPC
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE div(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] == true) ? (a / b[i]) : a);
        }
        return retval;
    }

    // DIVVA
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & divAssign(VEC_TYPE & a, VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, a[i] / b[i]);
        }
        return a;
    }

    // MDIVVA
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & divAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, a[i] / b[i]);
        }
        return a;
    }

    // DIVSA
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & divAssign(VEC_TYPE & a, SCALAR_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, a[i] / b);
        }
        return a;
    }

    // MDIVSA
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & divAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, a[i] / b);
        }
        return a;
    }

    // RCP
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE rcp(VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, SCALAR_TYPE(1.0) / b[i]);
        }
        return retval;
    }

    // MRCP
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE rcp(MASK_TYPE const & mask, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval.insert(i, SCALAR_TYPE(1.0) / b[i]);
            else retval.insert(i, b[i]);
        }
        return retval;
    }

    // RCPS
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE rcpScalar(SCALAR_TYPE a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a / b[i]);
        }
        return retval;
    }

    // MRCPS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE rcpScalar(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval.insert(i, a / b[i]);
            else retval.insert(i, b[i]);
        }
        return retval;
    }

    // RCPA
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & rcpAssign(VEC_TYPE & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            b.insert(i, SCALAR_TYPE(1.0) / b[i]);
        }
        return b;
    }

    // MRCPA
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & rcpAssign(MASK_TYPE const & mask, VEC_TYPE & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) b.insert(i, SCALAR_TYPE(1.0) / b[i]);
        }
        return b;
    }

    // RCPSA
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & rcpScalarAssign(SCALAR_TYPE a, VEC_TYPE & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            b.insert(i, a / b[i]);
        }
        return b;
    }

    // MRCPSA
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & rcpScalarAssign(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) b.insert(i, a / b[i]);
        }
        return b;
    }

    // LSHV
    template<typename VEC_TYPE, typename UINT_VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE shiftBitsLeft(VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (a[i] << b[i]));
        }
        return retval;
    }

    // MLSHV
    template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE shiftBitsLeft(MASK_TYPE const & mask, VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] == true) ? (a[i] << b[i]) : a[i]);
        }
        return retval;
    }

    // LSHS
    template<typename VEC_TYPE, typename SCALAR_UINT_TYPE>
    UME_FORCE_INLINE VEC_TYPE shiftBitsLeftScalar(VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (a[i] << b));
        }
        return retval;
    }

    // MLSHS
    template<typename VEC_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE shiftBitsLeftScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] == true) ? (a[i] << b) : a[i]);
        }
        return retval;
    }

    // LSHVA
    template<typename VEC_TYPE, typename UINT_VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & shiftBitsLeftAssign(VEC_TYPE & a, UINT_VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, (a[i] << b[i]));
        }
        return a;
    }

    // MLSHVA
    template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & shiftBitsLeftAssign(MASK_TYPE const & mask, VEC_TYPE & a, UINT_VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, a[i] << b[i]);
        }
        return a;
    }

    // LSHSA
    template<typename VEC_TYPE, typename SCALAR_UINT_TYPE>
    UME_FORCE_INLINE VEC_TYPE & shiftBitsLeftAssignScalar(VEC_TYPE & a, SCALAR_UINT_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, (a[i] << b));
        }
        return a;
    }

    // MLSHSA
    template<typename VEC_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & shiftBitsLeftAssignScalar(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_UINT_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, (a[i] << b));
        }
        return a;
    }

    // RSHV
    template<typename VEC_TYPE, typename UINT_VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE shiftBitsRight(VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (a[i] >> b[i]));
        }
        return retval;
    }

    // MRSHV
    template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE shiftBitsRight(MASK_TYPE const & mask, VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] == true) ? (a[i] >> b[i]) : a[i]);
        }
        return retval;
    }

    // RSHS
    template<typename VEC_TYPE, typename SCALAR_UINT_TYPE>
    UME_FORCE_INLINE VEC_TYPE shiftBitsRightScalar(VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (a[i] >> b));
        }
        return retval;
    }

    // MRSHS
    template<typename VEC_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE shiftBitsRightScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] == true) ? (a[i] >> b) : a[i]);
        }
        return retval;
    }

    // RSHVA
    template<typename VEC_TYPE, typename UINT_VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & shiftBitsRightAssign(VEC_TYPE & a, UINT_VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, (a[i] >> b[i]));
        }
        return a;
    }

    // MRSHVA
    template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & shiftBitsRightAssign(MASK_TYPE const & mask, VEC_TYPE & a, UINT_VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, (a[i] >> b[i]));
        }
        return a;
    }

    // RSHSA
    template<typename VEC_TYPE, typename SCALAR_UINT_TYPE>
    UME_FORCE_INLINE VEC_TYPE & shiftBitsRightAssignScalar(VEC_TYPE & a, SCALAR_UINT_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, (a[i] >> b));
        }
        return a;
    }

    // MSRHSA
    template<typename VEC_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & shiftBitsRightAssignScalar(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_UINT_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, (a[i] >> b));
        }
        return a;
    }

    // ROLV
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE, typename SCALAR_UINT_TYPE>
    UME_FORCE_INLINE VEC_TYPE rotateBitsLeft(VEC_TYPE const & a, UINT_VEC_TYPE const & b) {
        VEC_TYPE retval;
        uint32_t bitLength = 8 * sizeof(SCALAR_TYPE);
        SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
        bool topBit;
        SCALAR_TYPE shifted;
        SCALAR_TYPE raw_a[VEC_TYPE::length()];
        SCALAR_UINT_TYPE raw_b[UINT_VEC_TYPE::length()];
        SCALAR_TYPE raw_retval[VEC_TYPE::length()];

        a.store(raw_a);
        b.store(raw_b);

        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            shifted = raw_a[i];
            // shift one bit at a time. This simplifies type dependency checks.
            for (uint32_t j = 0; j < raw_b[i]; j++) {
                if ((shifted & topBitMask) != 0) topBit = true;
                else topBit = false;

                shifted <<= 1;
                if (topBit == true) shifted |= SCALAR_TYPE(1);
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
        VEC_TYPE retval;
        uint32_t bitLength = 8 * sizeof(SCALAR_TYPE);
        SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
        bool topBit;
        SCALAR_TYPE shifted;

        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true)
            {
                shifted = a[i];
                // shift one bit at a time. This simplifies type dependency checks.
                for (uint32_t j = 0; j < b[i]; j++) {
                    if ((shifted & topBitMask) != 0) topBit = true;
                    else topBit = false;

                    shifted <<= 1;
                    if (topBit == true) shifted |= SCALAR_TYPE(1);
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
        VEC_TYPE retval;
        uint32_t bitLength = 8 * sizeof(SCALAR_TYPE);
        SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
        bool topBit;
        SCALAR_TYPE shifted;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            shifted = a[i];
            // shift one bit at a time. This simplifies type dependency checks.
            for (uint32_t j = 0; j < b; j++) {
                if ((shifted & topBitMask) != 0) topBit = true;
                else topBit = false;

                shifted <<= 1;
                if (topBit == true) shifted |= SCALAR_TYPE(1);
                else               shifted &= ~(SCALAR_TYPE(1));
            }
            retval.insert(i, shifted);
        }
        return retval;
    }

    // MROLS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE rotateBitsLeftScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
        VEC_TYPE retval;
        uint32_t bitLength = 8 * sizeof(SCALAR_TYPE);
        SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
        bool topBit;
        SCALAR_TYPE shifted;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true)
            {
                shifted = a[i];
                // shift one bit at a time. This simplifies type dependency checks.
                for (uint32_t j = 0; j < b; j++) {
                    if ((shifted & topBitMask) != 0) topBit = true;
                    else topBit = false;

                    shifted <<= 1;
                    if (topBit == true) shifted |= SCALAR_TYPE(1);
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
        VEC_TYPE retval;
        uint32_t bitLength = 8 * sizeof(SCALAR_TYPE);
        SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
        bool topBit;
        SCALAR_TYPE shifted;

        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            shifted = a[i];
            // shift one bit at a time. This simplifies type dependency checks.
            for (uint32_t j = 0; j < b[i]; j++) {
                if ((shifted & topBitMask) != 0) topBit = true;
                else topBit = false;

                shifted <<= 1;
                if (topBit == true) shifted |= SCALAR_TYPE(1);
                else               shifted &= ~(SCALAR_TYPE(1));
            }
            a.insert(i, shifted);
        }
        return a;
    }

    // MROLVA
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & rotateBitsLeftAssign(MASK_TYPE const & mask, VEC_TYPE & a, UINT_VEC_TYPE const & b) {
        VEC_TYPE retval;
        uint32_t bitLength = 8 * sizeof(SCALAR_TYPE);
        SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
        bool topBit;
        SCALAR_TYPE shifted;

        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true)
            {
                shifted = a[i];
                // shift one bit at a time. This simplifies type dependency checks.
                for (uint32_t j = 0; j < b[i]; j++) {
                    if ((shifted & topBitMask) != 0) topBit = true;
                    else topBit = false;

                    shifted <<= 1;
                    if (topBit == true) shifted |= SCALAR_TYPE(1);
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
        VEC_TYPE retval;
        SCALAR_TYPE bitLength = 8 * sizeof(SCALAR_UINT_TYPE);
        SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
        bool topBit;
        SCALAR_TYPE shifted;

        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            shifted = a[i];
            // shift one bit at a time. This simplifies type dependency checks.
            for (uint32_t j = 0; j < b; j++) {
                if ((shifted & topBitMask) != 0) topBit = true;
                else topBit = false;

                shifted <<= 1;
                if (topBit == true)  shifted |= SCALAR_TYPE(0x1);
                else                shifted &= ~(SCALAR_TYPE(1));
            }
            a.insert(i, shifted);
        }
        return a;
    }

    // MROLSA
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & rotateBitsLeftAssignScalar(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_UINT_TYPE b) {
        VEC_TYPE retval;
        uint32_t bitLength = 8 * sizeof(SCALAR_UINT_TYPE);
        SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
        bool topBit;
        SCALAR_TYPE shifted;

        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true)
            {
                shifted = a[i];
                // shift one bit at a time. This simplifies type dependency checks.
                for (uint32_t j = 0; j < b; j++) {
                    if ((shifted & topBitMask) != 0) topBit = true;
                    else topBit = false;

                    shifted <<= 1;
                    if (topBit == true)  shifted |= SCALAR_TYPE(0x1);
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
        VEC_TYPE retval;
        uint32_t bitLength = 8 * sizeof(SCALAR_TYPE);
        SCALAR_UINT_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
        bool bottomBit;
        SCALAR_UINT_TYPE shifted;

        SCALAR_TYPE raw_a[VEC_TYPE::length()];
        SCALAR_UINT_TYPE raw_b[VEC_TYPE::length()];
        SCALAR_TYPE raw_retval[VEC_TYPE::length()];

        a.store(raw_a);
        b.store(raw_b);

        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            shifted = raw_a[i];
            // shift one bit at a time. This simplifies type dependency checks.
            for (uint32_t j = 0; j < raw_b[i]; j++) {
                if ((shifted & 1) != 0) bottomBit = true;
                else bottomBit = false;

                shifted >>= 1;
                if (bottomBit == true) shifted |= topBitMask;
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
        VEC_TYPE retval;
        uint32_t bitLength = 8 * sizeof(SCALAR_TYPE);
        SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
        bool bottomBit;
        SCALAR_TYPE shifted;

        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            shifted = a[i];
            if (mask[i] == true)
            {
                // shift one bit at a time. This simplifies type dependency checks.
                for (uint32_t j = 0; j < b[i]; j++) {
                    if ((shifted & 1) != 0) bottomBit = true;
                    else bottomBit = false;

                    shifted >>= 1;
                    if (bottomBit == true) shifted |= topBitMask;
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
        VEC_TYPE retval;
        uint32_t bitLength = 8 * sizeof(SCALAR_TYPE);
        SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
        bool bottomBit;
        SCALAR_TYPE shifted;

        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            shifted = a[i];
            // shift one bit at a time. This simplifies type dependency checks.
            for (uint32_t j = 0; j < b; j++) {
                if ((shifted & 1) != 0) bottomBit = true;
                else bottomBit = false;

                shifted >>= 1;
                if (bottomBit == true) shifted |= topBitMask;
                else                  shifted &= ~topBitMask;
            }
            retval.insert(i, shifted);
        }
        return retval;
    }

    // MRORS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE rotateBitsRightScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_UINT_TYPE b) {
        VEC_TYPE retval;
        uint32_t bitLength = 8 * sizeof(SCALAR_TYPE);
        SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
        bool bottomBit;
        SCALAR_TYPE shifted;

        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            shifted = a[i];
            if (mask[i] == true)
            {
                // shift one bit at a time. This simplifies type dependency checks.
                for (uint32_t j = 0; j < b; j++) {
                    if ((shifted & 1) != 0) bottomBit = true;
                    else bottomBit = false;

                    shifted >>= 1;
                    if (bottomBit == true) shifted |= topBitMask;
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
        uint32_t bitLength = 8 * sizeof(SCALAR_TYPE);
        SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
        bool bottomBit;
        SCALAR_TYPE shifted;

        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            shifted = a[i];
            // shift one bit at a time. This simplifies type dependency checks.
            for (uint32_t j = 0; j < b[i]; j++) {
                if ((shifted & 1) != 0) bottomBit = true;
                else bottomBit = false;

                shifted >>= 1;
                if (bottomBit == true) shifted |= topBitMask;
                else                  shifted &= ~topBitMask;
            }
            a.insert(i, shifted);
        }
        return a;
    }

    // MRORVA
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename UINT_VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & rotateBitsRightAssign(MASK_TYPE const & mask, VEC_TYPE & a, UINT_VEC_TYPE const & b) {
        uint32_t bitLength = 8 * sizeof(SCALAR_TYPE);
        SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
        bool bottomBit;
        SCALAR_TYPE shifted;

        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            shifted = a[i];
            if (mask[i] == true)
            {
                // shift one bit at a time. This simplifies type dependency checks.
                for (uint32_t j = 0; j < b[i]; j++) {
                    if ((shifted & 1) != 0) bottomBit = true;
                    else bottomBit = false;

                    shifted >>= 1;
                    if (bottomBit == true) shifted |= topBitMask;
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
        uint32_t bitLength = 8 * sizeof(SCALAR_TYPE);
        SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
        bool bottomBit;
        SCALAR_TYPE shifted;

        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            shifted = a[i];
            // shift one bit at a time. This simplifies type dependency checks.
            for (uint32_t j = 0; j < b; j++) {
                if ((shifted & 1) != 0) bottomBit = true;
                else bottomBit = false;

                shifted >>= 1;
                if (bottomBit == true) shifted |= topBitMask;
                else                  shifted &= ~topBitMask;
            }
            a.insert(i, shifted);
        }
        return a;
    }

    // MRORSA
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename SCALAR_UINT_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & rotateBitsRightAssignScalar(MASK_TYPE const & mask, VEC_TYPE &  a, SCALAR_UINT_TYPE const & b) {
        uint32_t bitLength = 8 * sizeof(SCALAR_TYPE);
        SCALAR_TYPE topBitMask = SCALAR_TYPE(1) << (bitLength - 1);
        bool bottomBit;
        SCALAR_TYPE shifted;

        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            shifted = a[i];
            if (mask[i] == true)
            {
                // shift one bit at a time. This simplifies type dependency checks.
                for (uint32_t j = 0; j < b; j++) {
                    if ((shifted & 1) != 0) bottomBit = true;
                    else bottomBit = false;

                    shifted >>= 1;
                    if (bottomBit == true) shifted |= topBitMask;
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
        MASK_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] == b[i]);
        }
        return retval;
    }

    // CMPEQS
    template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE MASK_TYPE isEqual(VEC_TYPE const & a, SCALAR_TYPE b) {
        MASK_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] == b);
        }
        return retval;
    }

    // CMPNEV
    template<typename MASK_TYPE, typename VEC_TYPE>
    UME_FORCE_INLINE MASK_TYPE isNotEqual(VEC_TYPE const & a, VEC_TYPE const & b) {
        
        MASK_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] != b[i]);
        }
        return retval;
    }

    // CMPNES
    template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE MASK_TYPE isNotEqual(VEC_TYPE const & a, SCALAR_TYPE b) {
        MASK_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] != b);
        }
        return retval;
    }

    // CMPGTV
    template<typename MASK_TYPE, typename VEC_TYPE>
    UME_FORCE_INLINE MASK_TYPE isGreater(VEC_TYPE const & a, VEC_TYPE const & b) {
        MASK_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i]>b[i]);
        }
        return retval;
    }

    // CMPGTS
    template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE MASK_TYPE isGreater(VEC_TYPE const & a, SCALAR_TYPE b) {
        MASK_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i]>b);
        }
        return retval;
    }

    // CMPLTV
    template<typename MASK_TYPE, typename VEC_TYPE>
    UME_FORCE_INLINE MASK_TYPE isLesser(VEC_TYPE const & a, VEC_TYPE const & b) {
        MASK_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i]<b[i]);
        }
        return retval;
    }

    // CMPLTS
    template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE MASK_TYPE isLesser(VEC_TYPE const & a, SCALAR_TYPE b) {
        MASK_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i]<b);
        }
        return retval;
    }

    // CMPGEV
    template<typename MASK_TYPE, typename VEC_TYPE>
    UME_FORCE_INLINE MASK_TYPE isGreaterEqual(VEC_TYPE const & a, VEC_TYPE const & b) {
        MASK_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] >= b[i]);
        }
        return retval;
    }

    // CMPGES
    template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE MASK_TYPE isGreaterEqual(VEC_TYPE const & a, SCALAR_TYPE b) {
        MASK_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] >= b);
        }
        return retval;
    }

    // CMPLEV
    template<typename MASK_TYPE, typename VEC_TYPE>
    UME_FORCE_INLINE MASK_TYPE isLesserEqual(VEC_TYPE const & a, VEC_TYPE const & b) {
        MASK_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] <= b[i]);
        }
        return retval;
    }

    // CMPLES
    template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE MASK_TYPE isLesserEqual(VEC_TYPE const & a, SCALAR_TYPE b) {
        MASK_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] <= b);
        }
        return retval;
    }

    // CMPEV 
    template<typename VEC_TYPE>
    UME_FORCE_INLINE bool isExact(VEC_TYPE const & a, VEC_TYPE const & b) {
        bool retval = true;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (a[i] != b[i]) {
                retval = false;
                break;
            }
        }
        return retval;
    }

    // CMPEQRV
    template<typename MASK_TYPE, typename VEC_TYPE>
    UME_FORCE_INLINE MASK_TYPE isEqualInRange(VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & margin) {
        MASK_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if ((a[i] < b[i] + margin[i]) && (a[i] > b[i] - margin[i]))
                retval.insert(i, true);
            else
                retval.insert(i, false);
        }
        return retval;
    }

    // CMPEQRS
    template<typename MASK_TYPE, typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE MASK_TYPE isEqualInRange(VEC_TYPE const & a, VEC_TYPE const & b, SCALAR_TYPE margin) {
        MASK_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if ((a[i] < b[i] + margin) && (a[i] > b[i] - margin))
                retval.insert(i, true);
            else
                retval.insert(i, false);
        }
        return retval;
    }

    // UNIQUE
    template<typename VEC_TYPE>
    UME_FORCE_INLINE bool unique(VEC_TYPE const & a) {
        bool retval = true;
        for (uint32_t i = 0; i < VEC_TYPE::length() - 1; i++) {
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
    UME_FORCE_INLINE VEC_TYPE binaryAnd(VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] & b[i]);
        }
        return retval;
    }

    // MANDV
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryAnd(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] ? a[i] & b[i] : a[i]));
        }
        return retval;
    }

    // ANDS
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryAnd(VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] & b);
        }
        return retval;
    }

    // MANDS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryAnd(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] ? a[i] & b : a[i]));
        }
        return retval;
    }

    // binaryAnd (scalar, VEC) -> VEC
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryAnd(SCALAR_TYPE a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a & b[i]);
        }
        return retval;
    }

    // binaryAnd (MASK, scalar, VEC) -> VEC
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryAnd(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] ? a & b[i] : a));
        }
        return retval;
    }

    // ANDVA
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryAndAssign(VEC_TYPE & a, VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, a[i] & b[i]);
        }
        return a;
    }

    // MANDVA
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryAndAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, a[i] & b[i]);
        }
        return a;
    }

    // ANDSA
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryAndAssign(VEC_TYPE & a, SCALAR_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, a[i] & b);
        }
        return a;
    }

    // MANDSA 
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryAndAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, a[i] & b);
        }
        return a;
    }

    // ORV
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryOr(VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] | b[i]);
        }
        return retval;
    }

    // MORV
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryOr(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] ? (a[i] | b[i]) : a[i]));
        }
        return retval;
    }

    // ORS
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryOr(VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] | b);
        }
        return retval;
    }

    // MORS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryOr(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] ? (a[i] | b) : a[i]));
        }
        return retval;
    }

    // ORVA
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryOrAssign(VEC_TYPE & a, VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, a[i] | b[i]);
        }
        return a;
    }

    // MORVA
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryOrAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, a[i] | b[i]);
        }
        return a;
    }

    // ORSA
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryOrAssign(VEC_TYPE & a, SCALAR_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, a[i] | b);
        }
        return a;
    }

    // MORSA
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryOrAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, a[i] | b);
        }
        return a;
    }

    // XORV
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryXor(VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] ^ b[i]);
        }
        return retval;
    }

    // MXORV
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryXor(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] == true) ? (a[i] ^ b[i]) : a[i]);
        }
        return retval;
    }

    // XORS
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryXor(VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[i] ^ b);
        }
        return retval;
    }

    // MXORS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryXor(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] == true) ? (a[i] ^ b) : a[i]);
        }
        return retval;
    }

    // XORVA
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryXorAssign(VEC_TYPE & a, VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, a[i] ^ b[i]);
        }
        return a;
    }

    // MXORVA
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryXorAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, a[i] ^ b[i]);
        }
        return a;
    }

    // XORSA
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryXorAssign(VEC_TYPE & a, SCALAR_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, a[i] ^ b);
        }
        return a;
    }

    // MXORSA
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryXorAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, a[i] ^ b);
        }
        return a;
    }

    // BNOT
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryNot(VEC_TYPE const & a) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            SCALAR_TYPE temp = ~a[i];
            retval.insert(i, temp);
        }
        return retval;
    }

    // MBNOT
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryNot(MASK_TYPE const & mask, VEC_TYPE const & a) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, (mask[i] == true) ? (~a[i]) : (a[i]));
        }
        return retval;
    }

    // BNOTA
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryNotAssign(VEC_TYPE & a) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, ~a[i]);
        }
        return a;
    }

    // MBNOTA
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryNotAssign(MASK_TYPE const & mask, VEC_TYPE & a) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, ~a[i]);
        }
        return a;
    }

    // BANDNOTV
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryAndNot(VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, ((~a[i]) & b[i]));
        }
        return retval;
    }

    // MBANDNOTV
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryAndNot(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval = a;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if(mask[i] == true) retval.insert(i, ((~a[i]) & b[i]));
        }
        return retval;
    }

    // BANDNOTS
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryAndNot(VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, ((~a[i]) & b));
        }
        return retval;
    }

    // MBANDNOTS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE binaryAndNot(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval = a;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval.insert(i, ((~a[i]) & b));
        }
        return retval;
    }

    // BANDNOTVA
    template<typename VEC_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryAndNotAssign(VEC_TYPE & a, VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, ((~a[i]) & b[i]));
        }
        return a;
    }

    // MBANDNOTVA
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryAndNotAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, ((~a[i]) & b[i]));
        }
        return a;
    }

    // BANDNOTSA
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryAndNotAssign(VEC_TYPE & a, SCALAR_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, ((~a[i]) & b));
        }
        return a;
    }

    // MBANDNOTSA
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & binaryAndNotAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) a.insert(i, ((~a[i]) & b));
        }
        return a;
    }

    // LNOT
    template<typename MASK_TYPE>
    UME_FORCE_INLINE MASK_TYPE logicalNot(MASK_TYPE const & mask) {
        MASK_TYPE retval(false);
        for (uint32_t i = 0; i < MASK_TYPE::length(); i++) {
            if (mask[i] == false) retval.insert(i, true);
        }
        return retval;
    }

    // LNOTA
    template<typename MASK_TYPE>
    UME_FORCE_INLINE MASK_TYPE & logicalNotAssign(MASK_TYPE & mask) {
        for (uint32_t i = 0; i < MASK_TYPE::length(); i++) {
            mask.insert(i, !mask[i]);
        }
        return mask;
    }

    // BLENDV
    template<typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE blend(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, mask[i] ? b[i] : a[i]);
        }
        return retval;
    }

    // BLENDS
    template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE blend(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, mask[i] ? b[i] : a);
        }
        return retval;
    }

    // SWIZZLE
    template<typename VEC_TYPE, typename SWIZZLE_MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE swizzle(SWIZZLE_MASK_TYPE const & sMask, VEC_TYPE const & a) {
        VEC_TYPE retval;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, a[sMask[i]]);
        }
        return retval;
    }

    // SWIZZLEA
    template<typename VEC_TYPE, typename SWIZZLE_MASK_TYPE>
    UME_FORCE_INLINE VEC_TYPE & swizzleAssign(SWIZZLE_MASK_TYPE const & sMask, VEC_TYPE & a) {
        VEC_TYPE temp(a);
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            a.insert(i, temp[sMask[i]]);
        }
        return a;
    }

    // SORTA
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE sortAscending(VEC_TYPE const & a) {
        const uint32_t VEC_LEN = VEC_TYPE::length();
        std::array<SCALAR_TYPE, VEC_LEN> temp;
        VEC_TYPE retval;

        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            temp[i] = a[i];
        }

        std::sort(temp.begin(), temp.end());
        
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, temp[i]);
        }
        return retval;
    }
    
    // SORTD
    template<typename VEC_TYPE, typename SCALAR_TYPE>
    UME_FORCE_INLINE VEC_TYPE sortDescending(VEC_TYPE const & a) {
        const uint32_t VEC_LEN = VEC_TYPE::length();
        std::array<SCALAR_TYPE, VEC_LEN> temp;
        VEC_TYPE retval;

        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            temp[i] = a[i];
        }

        std::sort(temp.begin(), temp.end());

        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval.insert(i, temp[VEC_LEN - i - 1]);
        }
        return retval;
    }

    // reduceAdd(VEC) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceAdd(VEC_TYPE const & a) {
        SCALAR_TYPE retval = a[0];
        for (uint32_t i = 1; i < VEC_TYPE::length(); i++) {
            retval += a[i];
        }
        return retval;
    }

    // reduceAdd(MASK, VEC) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceAdd(MASK_TYPE const & mask, VEC_TYPE const & a) {
        SCALAR_TYPE retval = a[0];
        for (uint32_t i = 1; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval += a[i];
        }
        return retval;
    }

    // reduceAdd (scalar, VEC) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceAdd(SCALAR_TYPE & a, VEC_TYPE const & b) {
        SCALAR_TYPE retval = a;
        for (uint32_t i = 0; i <VEC_TYPE::length(); i++) {
            retval += b[i];
        }
        return retval;
    }

    // reduceAdd(MASK, scalar, VEC) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceAdd(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
        SCALAR_TYPE retval = a;
        for (uint32_t i = 0; i <VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval += b[i];
        }
        return retval;
    }

    // reduceMult(VEC) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceMult(VEC_TYPE const & a) {
        SCALAR_TYPE retval = a[0];
        for (uint32_t i = 1; i < VEC_TYPE::length(); i++) {
            retval *= a[i];
        }
        return retval;
    }

    // reduceMult(MASK, VEC) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceMult(MASK_TYPE const & mask, VEC_TYPE const & a) {
        SCALAR_TYPE retval = (mask[0] == true) ? a[0] : 0; // TODO: replace 0 with const expr returning zero depending on SCALAR type.
        for (uint32_t i = 1; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval *= a[i];
        }
        return retval;
    }

    // reduceMult(scalar, VEC) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceMultScalar(SCALAR_TYPE a, VEC_TYPE const & b) {
        SCALAR_TYPE retval = a;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval *= b[i];
        }
        return retval;
    }

    // reduceMult(MASK, scalar, VEC) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceMultScalar(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
        SCALAR_TYPE retval = a;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval *= b[i];
        }
        return retval;
    }

    // HLAND
    template<typename MASK_TYPE>
    UME_FORCE_INLINE bool reduceLogicalAnd(MASK_TYPE const & a) {
        bool retval = a[0];
        for (uint32_t i = 1; i < MASK_TYPE::length(); i++) {
            retval &= a[i];
        }
        return retval;
    }

    // HLOR
    template<typename MASK_TYPE>
    UME_FORCE_INLINE bool reduceLogicalOr(MASK_TYPE const & a) {
        bool retval = a[0];
        for (uint32_t i = 1; i < MASK_TYPE::length(); i++) {
            retval |= a[i];
        }
        return retval;
    }

    // HLXOR
    template<typename MASK_TYPE>
    UME_FORCE_INLINE bool reduceLogicalXor(MASK_TYPE const & a) {
        bool retval = a[0];
        for (uint32_t i = 1; i < MASK_TYPE::length(); i++) {
            retval ^= a[i];
        }
        return retval;
    }

    // reduceBinaryAnd (VEC) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceBinaryAnd(VEC_TYPE const & a) {
        SCALAR_TYPE retval = a[0];
        for (uint32_t i = 1; i < VEC_TYPE::length(); i++) {
            retval &= a[i];
        }
        return retval;
    }

    // reduceBinaryAnd (MASK, VEC) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceBinaryAnd(MASK_TYPE const & mask, VEC_TYPE const & a) {
        SCALAR_TYPE retval = (mask[0] == true) ? a[0] : (SCALAR_TYPE)-1;
        for (uint32_t i = 1; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval &= a[i];
        }
        return retval;
    }

    // reduceBinaryAnd (scalar, VEC) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceBinaryAndScalar(SCALAR_TYPE a, VEC_TYPE const & b) {
        SCALAR_TYPE retval = a;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval &= b[i];
        }
        return retval;
    }

    // reduceBinaryAnd (MASK, scalar, VEC) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceBinaryAndScalar(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
        SCALAR_TYPE retval = a;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval &= b[i];
        }
        return retval;
    }

    // reduceBinaryOr (VEC) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceBinaryOr(VEC_TYPE const & a) {
        SCALAR_TYPE retval = a[0];
        for (uint32_t i = 1; i < VEC_TYPE::length(); i++) {
            retval |= a[i];
        }
        return retval;
    }

    // reduceBinaryOr (MASK, VEC) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceBinaryOr(MASK_TYPE const & mask, VEC_TYPE const & a) {
        SCALAR_TYPE retval = (mask[0] == true) ? a[0] : 0; // TODO: 0-initializer of SCALAR_TYPE
        for (uint32_t i = 1; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval |= a[i];
        }
        return retval;
    }

    // reduceBinaryOr (scalar, VEC) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceBinaryOrScalar(SCALAR_TYPE a, VEC_TYPE const & b) {
        SCALAR_TYPE retval = a;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval |= b[i];
        }
        return retval;
    }

    // reduceBinaryOr (MASK, scalar, VEC) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceBinaryOrScalar(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
        SCALAR_TYPE retval = a;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval |= b[i];
        }
        return retval;
    }

    // reduceBinaryXor() -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceBinaryXor(VEC_TYPE const & a) {
        SCALAR_TYPE retval = 0;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval ^= a[i];
        }
        return retval;
    }

    // reduceBinaryXor(MASK) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceBinaryXor(MASK_TYPE const & mask, VEC_TYPE const & a) {
        SCALAR_TYPE retval = 0;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval ^= a[i];
        }
        return retval;
    }

    // reduceBinaryXor(scalar) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceBinaryXorScalar(SCALAR_TYPE a, VEC_TYPE const & b) {
        SCALAR_TYPE retval = a;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            retval ^= b[i];
        }
        return retval;
    }

    // reduceBinaryXor(MASK, scalar) -> scalar
    template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
    UME_FORCE_INLINE SCALAR_TYPE reduceBinaryXorScalar(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
        SCALAR_TYPE retval = a;
        for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
            if (mask[i] == true) retval ^= b[i];
        }
        return retval;
    }

    // xTOy (UTOI, ITOU, UTOF, FTOU, PROMOTE, DEGRADE)
    template<typename VEC_Y_TYPE, typename SCALAR_Y_TYPE, typename VEC_X_TYPE>
    UME_FORCE_INLINE VEC_Y_TYPE xtoy(VEC_X_TYPE const & a) {
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
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (a[i] > b[i] ? a[i] : b[i]));
            }
            return retval;
        }

        // MMAXV
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE max(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) retval.insert(i, (a[i] > b[i] ? a[i] : b[i]));
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // MAXS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE maxScalar(VEC_TYPE const & a, SCALAR_TYPE b) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (a[i] > b ? a[i] : b));
            }
            return retval;
        }

        // MMAXS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE maxScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) retval.insert(i, (a[i] > b ? a[i] : b));
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // MAXVA
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & maxAssign(VEC_TYPE & a, VEC_TYPE const & b) {
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (b[i] > a[i])a.insert(i, b[i]);
            }
            return a;
        }

        // MMAXVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & maxAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true && (b[i] > a[i]))a.insert(i, b[i]);
            }
            return a;
        }

        // MAXSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE & maxScalarAssign(VEC_TYPE & a, SCALAR_TYPE b) {
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (b > a[i]) a.insert(i, b);
            }
            return a;
        }

        // MMAXSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & maxScalarAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true && (b > a[i])) a.insert(i, b);
            }
            return a;
        }

        // MINS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE minScalar(VEC_TYPE const & a, SCALAR_TYPE b) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] < b ? a[i] : b);
            }
            return retval;
        }

        // MMINS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE minScalar(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            VEC_TYPE retval(std::numeric_limits<SCALAR_TYPE>::max());
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) retval.insert(i, a[i] < b ? a[i] : b);
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // MINV
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE min(VEC_TYPE const & a, VEC_TYPE const & b) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] < b[i] ? a[i] : b[i]);
            }
            return retval;
        }

        // MMINV
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE min(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            VEC_TYPE retval(std::numeric_limits<SCALAR_TYPE>::max());
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) retval.insert(i, a[i] < b[i] ? a[i] : b[i]);
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // MINSA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE & minScalarAssign(VEC_TYPE & a, SCALAR_TYPE b) {
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (b < a[i]) a.insert(i, b);
            }
            return a;
        }

        // MMINSA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & minScalarAssign(MASK_TYPE const & mask, VEC_TYPE & a, SCALAR_TYPE b) {
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true && (b < a[i])) a.insert(i, b);
            }
            return a;
        }

        // MINVA
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & minAssign(VEC_TYPE & a, VEC_TYPE const & b) {
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (b[i] < a[i]) a.insert(i, b[i]);
            }
            return a;
        }

        // MMINVA
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & minAssign(MASK_TYPE const & mask, VEC_TYPE & a, VEC_TYPE const & b) {
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true && (b[i] < a[i])) a.insert(i, b[i]);
            }
            return a;
        }

        // HMAX
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceMax(VEC_TYPE const & a) {
            SCALAR_TYPE retval = a[0];
            for (uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                if (a[i] > retval) retval = a[i];
            }
            return retval;
        }

        // MHMAX
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceMax(MASK_TYPE const & mask, VEC_TYPE const & a) {
            SCALAR_TYPE retval = std::numeric_limits<SCALAR_TYPE>::min();
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if ((mask[i] == true) && a[i] > retval) retval = a[i];
            }
            return retval;
        }

        // HMAXS
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceMax(SCALAR_TYPE a, VEC_TYPE const & b) {
            SCALAR_TYPE retval = a;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (b[i] > retval) retval = b[i];
            }
            return retval;
        }

        // MHMAXS
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceMax(MASK_TYPE const & mask, SCALAR_TYPE a, VEC_TYPE const & b) {
            SCALAR_TYPE retval = a;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if ((mask[i] == true) && (a[i] > retval)) retval = a[i];
            }
            return retval;
        }

        // IMAX
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE uint32_t indexMax(VEC_TYPE const & a) {
            uint32_t indexMax = 0;
            SCALAR_TYPE maxVal = a[0];
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (a[i] > maxVal) {
                    maxVal = a[i];
                    indexMax = i;
                }
            }
            return indexMax;
        }

        // MIMAX
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE uint32_t indexMax(MASK_TYPE const & mask, VEC_TYPE const & a) {
            uint32_t indexMax = 0xFFFFFFFF;
            SCALAR_TYPE maxVal = std::numeric_limits<SCALAR_TYPE>::min();
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true && a[i] > maxVal) {
                    maxVal = a[i];
                    indexMax = i;
                }
            }
            return indexMax;
        }

        // HMIN
        template<typename SCALAR_TYPE, typename VEC_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceMin(VEC_TYPE const & a) {
            SCALAR_TYPE retval = a[0];
            for (uint32_t i = 1; i < VEC_TYPE::length(); i++) {
                if (a[i] < retval) retval = a[i];
            }
            return retval;
        }

        // MHMIN
        template<typename SCALAR_TYPE, typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE SCALAR_TYPE reduceMin(MASK_TYPE const & mask, VEC_TYPE const & a) {
            SCALAR_TYPE retval = std::numeric_limits<SCALAR_TYPE>::max();
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if ((mask[i] == true) && a[i] < retval) retval = a[i];
            }
            return retval;
        }

        // IMIN
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE uint32_t indexMin(VEC_TYPE const & a) {
            uint32_t indexMin = 0;
            SCALAR_TYPE minVal = std::numeric_limits<SCALAR_TYPE>::max();
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (a[i] < minVal) {
                    minVal = a[i];
                    indexMin = i;
                }
            }
            return indexMin;
        }

        // MIMIN
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE uint32_t indexMin(MASK_TYPE const & mask, VEC_TYPE const & a) {
            uint32_t indexMin = 0xFFFFFFFF;
            SCALAR_TYPE minVal = std::numeric_limits<SCALAR_TYPE>::max();
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true && a[i] < minVal) {
                    minVal = a[i];
                    indexMin = i;
                }
            }
            return indexMin;
        }

        // ABS
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE abs(VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                // abs for floating point numbers is non-trivial. Using std::abs for reliability.
                retval.insert(i, std::abs(a[i]));
            }
            return retval;
        }

        // MABS
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE abs(MASK_TYPE const & mask, VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                // abs for floating point numbers is non-trivial. Using std::abs for reliability.
                retval.insert(i, (mask[i] == true ? std::abs(a[i]) : a[i]));
            }
            return retval;
        }

        // ABSA
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & absAssign(VEC_TYPE & a) {
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                // abs for floating point numbers is non-trivial. Using std::abs for reliability.
                a.insert(i, std::abs(a[i]));
            }
            return a;
        }

        // MABSA
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE absAssign(MASK_TYPE const & mask, VEC_TYPE & a) {
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                // abs for floating point numbers is non-trivial. Using std::abs for reliability.
                a.insert(i, (mask[i] == true ? std::abs(a[i]) : a[i]));
            }
            return a;
        }

        // COPYSIGN
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE copySign(VEC_TYPE const & a, VEC_TYPE const & b) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                // Can't use std::copysign because this has to work also for integers.
                // typesafe sign: ((x > 0) ? 1 : ((x < 0) ? -1 : 0)))
                SCALAR_TYPE sign = (b[i] > SCALAR_TYPE(0)) ? SCALAR_TYPE(1) : ((b[i] < SCALAR_TYPE(0)) ? SCALAR_TYPE(-1) : SCALAR_TYPE(0));
                retval.insert(i, std::abs(a[i]) * sign);
            }
            return retval;
        }

        // MCOPYSIGN
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE copySign(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                // Can't use std::copysign because this has to work also for integers.
                // typesafe sign: ((x > 0) ? 1 : ((x < 0) ? -1 : 0)))
                SCALAR_TYPE sign = (b[i] > SCALAR_TYPE(0)) ? SCALAR_TYPE(1) : ((b[i] < SCALAR_TYPE(0)) ? SCALAR_TYPE(-1) : SCALAR_TYPE(0));
                retval.insert(i, (mask[i] == true ? std::abs(a[i]) * sign : a[i]) );
            }
            return retval;
        }

        // SQR
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE sqr(VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, a[i] * a[i]);
            }
            return retval;
        }

        // MSQR
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE sqr(MASK_TYPE const & mask, VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) retval.insert(i, a[i] * a[i]);
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // SQRA
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & sqrAssign(VEC_TYPE & a) {
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, a[i] * a[i]);
            }
            return a;
        }

        // MSQRA
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & sqrAssign(MASK_TYPE const & mask, VEC_TYPE & a) {
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) a.insert(i, a[i] * a[i]);
            }
            return a;
        }

        // SQRT
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE sqrt(VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, std::sqrt(a[i]));
            }
            return retval;
        }

        // MSQRT
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE sqrt(MASK_TYPE const & mask, VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? std::sqrt(a[i]) : a[i]);
            }
            return retval;
        }

        // SQRTA
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE & sqrtAssign(VEC_TYPE & a) {
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                a.insert(i, std::sqrt(a[i]));
            }
            return a;
        }

        // MSQRTA
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & sqrtAssign(MASK_TYPE const & mask, VEC_TYPE & a) {
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) a.insert(i, std::sqrt(a[i]));
            }
            return a;
        }

        // RSQRT
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE rsqrt(VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, SCALAR_TYPE(1.0) / std::sqrt(a[i]));
            }
            return retval;
        }
        // MRSQRT
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE rsqrt(MASK_TYPE const & mask, VEC_TYPE const & a) {
            VEC_TYPE retval;
            decltype(retval.extract(0)) temp;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = SCALAR_TYPE(1.0) / std::sqrt(a[i]);
                retval.insert(i, (mask[i] == true) ? temp : a[i]);
            }
            return retval;
        }
        // RSQRTA
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE & rsqrtAssign(VEC_TYPE & a) {
            decltype(a.extract(0)) temp;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = SCALAR_TYPE(1.0) / std::sqrt(a[i]);
                a.insert(i, temp);
            }
            return a;
        }
        // MRSQRTA
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE & rsqrtAssign(MASK_TYPE const & mask, VEC_TYPE & a) {
            decltype(a.extract(0)) temp;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                temp = SCALAR_TYPE(1.0) / std::sqrt(a[i]);
                if (mask[i] == true) a.insert(i, temp);
            }
            return a;
        }

        // POWV
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE pow(VEC_TYPE const & a, VEC_TYPE const & b) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, std::pow(a[i], b[i]));
            }
            return retval;
        }

        // MPOWV
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE pow(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) retval.insert(i, std::pow(a[i], b[i]));
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // POWS
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE pows(VEC_TYPE const & a, SCALAR_TYPE b) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, std::pow(a[i], b));
            }
            return retval;
        }

        // MPOWS
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE pows(MASK_TYPE const & mask, VEC_TYPE const & a, SCALAR_TYPE b) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) retval.insert(i, std::pow(a[i], b));
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // ROUND
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE round(VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, std::round(a[i]));
            }
            return retval;
        }

        // MROUND
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE round(MASK_TYPE const & mask, VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) retval.insert(i, std::round(a[i]));
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // TRUNC
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename INT_VEC_TYPE>
        UME_FORCE_INLINE INT_VEC_TYPE truncToInt(VEC_TYPE const & a) {
            INT_VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, SCALAR_TYPE(std::trunc(a[i])));
            }
            return retval;
        }

        // MTRUNC
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename INT_VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE INT_VEC_TYPE truncToInt(MASK_TYPE const & mask, VEC_TYPE const & a) {
            INT_VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) retval.insert(i, SCALAR_TYPE(std::trunc(a[i])));
                else retval.insert(i, 0);
            }
            return retval;
        }

        // FLOOR
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE floor(VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length();i++) {
                retval.insert(i, std::floor(a[i]));
            }
            return retval;
        }

        // MFLOOR
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE floor(MASK_TYPE const & mask, VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length();i++) {
                if (mask[i] == true) retval.insert(i, std::floor(a[i]));
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // CEIL
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE ceil(VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length();i++) {
                retval.insert(i, std::ceil(a[i]));
            }
            return retval;
        }

        // MCEIL
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE ceil(MASK_TYPE const & mask, VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length();i++) {
                if (mask[i] == true) retval.insert(i, std::ceil(a[i]));
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // FMULADDV
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE fmuladd(VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (a[i] * b[i]) + c[i]);
            }
            return retval;
        }

        // MFMULADDV
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE fmuladd(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) retval.insert(i, (a[i] * b[i]) + c[i]);
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // FADDMULV
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE faddmul(VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (a[i] + b[i]) * c[i]);
            }
            return retval;
        }

        // MFADDMULV
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE faddmul(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) retval.insert(i, (a[i] + b[i]) * c[i]);
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // FMULSUBV
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE fmulsub(VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (a[i] * b[i]) - c[i]);
            }
            return retval;
        }

        // MFMULSUBV
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE fmulsub(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) retval.insert(i, (a[i] * b[i]) - c[i]);
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // FSUBMULV
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE fsubmul(VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (a[i] - b[i]) * c[i]);
            }
            return retval;
        }

        // MFSUBMULV
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE fsubmul(MASK_TYPE const & mask, VEC_TYPE const & a, VEC_TYPE const & b, VEC_TYPE const & c) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) retval.insert(i, (a[i] - b[i]) * c[i]);
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // ISFIN
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE MASK_TYPE isfin(VEC_TYPE const & a) {
            MASK_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, std::isfinite(a[i]));
            }
            return retval;
        }

        // ISINF
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE MASK_TYPE isinf(VEC_TYPE const & a) {
            MASK_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, std::isinf(a[i]));
            }
            return retval;
        }

        // ISAN
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE MASK_TYPE isan(VEC_TYPE const & a) {
            MASK_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (!std::isnan(a[i]) && !std::isinf(a[i])));
            }
            return retval;
        }

        // ISNAN
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE MASK_TYPE isnan(VEC_TYPE const & a) {
            MASK_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, std::isnan(a[i]));
            }
            return retval;
        }

        // ISNORM
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE MASK_TYPE isnorm(VEC_TYPE const & a) {
            MASK_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, std::isnormal(a[i]));
            }
            return retval;
        }

        // ISSUB
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE MASK_TYPE issub(VEC_TYPE const & a) {
            MASK_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
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
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE MASK_TYPE iszero(VEC_TYPE const & a) {
            MASK_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (a[i] == SCALAR_TYPE(0.0)));
            }
            return retval;
        }

        // ISZEROSUB
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE MASK_TYPE iszerosub(VEC_TYPE const & a) {
            MASK_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                bool isZero = (a[i] == SCALAR_TYPE(0.0));
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
        UME_FORCE_INLINE VEC_TYPE exp(VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, std::exp(a[i]));
            }
            return retval;
        }

        // MEXP
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE exp(MASK_TYPE const & mask, VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? std::exp(a[i]) : a[i]);
            }
            return retval;
        }

        // SIN
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE sin(VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, std::sin(a[i]));
            }
            return retval;
        }

        // MSIN
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE sin(MASK_TYPE const & mask, VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? std::sin(a[i]) : a[i]);
            }
            return retval;
        }

        // COS
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE cos(VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, std::cos(a[i]));
            }
            return retval;
        }

        // MCOS
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE cos(MASK_TYPE const & mask, VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, (mask[i] == true) ? std::cos(a[i]) : a[i]);
            }
            return retval;
        }

        // TAN
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE tan(VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, std::tan(a[i]));
            }
            return retval;
        }

        // MTAN
        template<typename VEC_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE tan(MASK_TYPE const & mask, VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) retval.insert(i, std::tan(a[i]));
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // CTAN
        template<typename VEC_TYPE, typename SCALAR_TYPE>
        UME_FORCE_INLINE VEC_TYPE ctan(VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, SCALAR_TYPE(1.0) / std::tan(a[i]));
            }
            return retval;
        }

        // MCTAN
        template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
        UME_FORCE_INLINE VEC_TYPE ctan(MASK_TYPE const & mask, VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                if (mask[i] == true) retval.insert(i, SCALAR_TYPE(1.0) / std::tan(a[i]));
                else retval.insert(i, a[i]);
            }
            return retval;
        }

        // ATAN
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE atan(VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, std::atan(a[i]));
            }
            return retval;
        }

        // ATAN2
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE atan2(VEC_TYPE const & a, VEC_TYPE const & b) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, std::atan2(a[i], b[i]));
            }
            return retval;
        }

        // LOG
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE log(VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, std::log(a.extract(i)));
            }
            return retval;
        }

        // LOG10
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE log10(VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, std::log10(a.extract(i)));
            }
            return retval;
        }

        // LOG2
        template<typename VEC_TYPE>
        UME_FORCE_INLINE VEC_TYPE log2(VEC_TYPE const & a) {
            VEC_TYPE retval;
            for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
                retval.insert(i, std::log2(a.extract(i)));
            }
            return retval;
        }

    } // UME::SIMD::SCALAR_EMULATION::MATH
} // namespace UME::SIMD::SCALAR_EMULATION

} // UME::SIMD
} // UME

#endif
