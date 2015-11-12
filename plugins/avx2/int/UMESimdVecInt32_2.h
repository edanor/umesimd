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

#ifndef UME_SIMD_VEC_INT32_2_H_
#define UME_SIMD_VEC_INT32_2_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int32_t, 2> :
        public SIMDVecSignedInterface<
        SIMDVec_i<int32_t, 2>,
        SIMDVec_u<uint32_t, 2>,
        int32_t,
        2,
        uint32_t,
        SIMDVecMask<2>,
        SIMDVecSwizzle<2 >> ,
        public SIMDVecPackableInterface<
        SIMDVec_i<int32_t, 2>,
        SIMDVec_i<int32_t, 1 >>
    {
        friend class SIMDVec_u<uint32_t, 2>;
        friend class SIMDVec_f<float, 2>;
        friend class SIMDVec_f<double, 2>;

    private:
        int32_t mVec[2];

    public:
        inline SIMDVec_i() {};

        inline explicit SIMDVec_i(int32_t i) {
            mVec[0] = i;
            mVec[1] = i;
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_i(int32_t const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
        };

        inline SIMDVec_i(int32_t i0, int32_t i1)
        {
            mVec[0] = i0;
            mVec[1] = i1;
        }

        inline int32_t extract(uint32_t index) const {
            return mVec[index & 1];
        }

        // Override Access operators
        inline int32_t operator[] (uint32_t index) const {
            return mVec[index & 1];
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_i, SIMDVecMask<2>> operator[] (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_i, SIMDVecMask<2>>(mask, static_cast<SIMDVec_i &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVec_i & insert(uint32_t index, int32_t value) {
            mVec[index & 1] = value;
            return *this;
        }

        // UNIQUE
        inline bool unique() const {
            return mVec[0] != mVec[1];
        }

        // ITOU
        inline SIMDVec_u<uint32_t, 2> itou() {
            uint32_t t0 = uint32_t(mVec[0]);
            uint32_t t1 = uint32_t(mVec[1]);
            return SIMDVec_u<uint32_t, 2>(t0, t1);
        }
        inline  operator SIMDVec_u<uint32_t, 2> const ();
    };

}
}

#endif
