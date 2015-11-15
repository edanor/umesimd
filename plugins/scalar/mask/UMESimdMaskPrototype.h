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

#ifndef UME_SIMD_MASK_PROTOTYPE_H_
#define UME_SIMD_MASK_PROTOTYPE_H_

#include <type_traits>
#include "../../../UMESimdInterface.h"
#include <immintrin.h>

namespace UME {
namespace SIMD {

    template<uint32_t VEC_LEN>
    struct SIMDVecMask_traits {};

    // MASK_BASE_TYPE is the type of element that will represent single entry in
    //                mask register. This can be for examle a 'bool' or 'unsigned int' or 'float'
    //                The actual representation depends on how the underlying instruction
    //                set handles the masks/mask registers. For scalar emulation the mask vetor should
    //                be represented using a boolean values. Bool in C++ has one disadventage: it is possible
    //                for the compiler to implicitly cast it to integer. To forbid this casting operations from
    //                happening the default type has to be wrapped into a class. 
    template<uint32_t VEC_LEN>
    class SIMDVecMask : public SIMDMaskBaseInterface<
        SIMDVecMask<VEC_LEN>,
        bool,
        VEC_LEN>
    {
        typedef SIMDVecMask_traits<VEC_LEN> MASK_TRAITS;
    private:
        bool mMask[VEC_LEN]; // each entry represents single mask element. For real SIMD vectors, mMask will be of mask intrinsic type.
    public:
        inline SIMDVecMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecMask(bool m) {
            for (int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = m;
            }
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecMask(bool const * p) { this->load(p); }

        // TODO: this should be handled using variadic templates, but unfortunatelly Visual Studio does not support this feature...
        inline SIMDVecMask(bool m0, bool m1)
        {
            mMask[0] = m0;
            mMask[1] = m1;
        }

        inline SIMDVecMask(bool m0, bool m1, bool m2, bool m3)
        {
            mMask[0] = m0;
            mMask[1] = m1;
            mMask[2] = m2;
            mMask[3] = m3;
        }

        inline SIMDVecMask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7)
        {
            mMask[0] = m0; mMask[1] = m1;
            mMask[2] = m2; mMask[3] = m3;
            mMask[4] = m4; mMask[5] = m5;
            mMask[6] = m6; mMask[7] = m7;
        }

        inline SIMDVecMask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7,
            bool m8, bool m9, bool m10, bool m11,
            bool m12, bool m13, bool m14, bool m15)
        {
            mMask[0] = m0;   mMask[1] = m1;
            mMask[2] = m2;   mMask[3] = m3;
            mMask[4] = m4;   mMask[5] = m5;
            mMask[6] = m6;   mMask[7] = m7;
            mMask[8] = m8;   mMask[9] = m9;
            mMask[10] = m10; mMask[11] = m11;
            mMask[12] = m12; mMask[13] = m13;
            mMask[14] = m14; mMask[15] = m15;
        }

        inline SIMDVecMask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7,
            bool m8, bool m9, bool m10, bool m11,
            bool m12, bool m13, bool m14, bool m15,
            bool m16, bool m17, bool m18, bool m19,
            bool m20, bool m21, bool m22, bool m23,
            bool m24, bool m25, bool m26, bool m27,
            bool m28, bool m29, bool m30, bool m31)
        {
            mMask[0] = m0;   mMask[1] = m1;
            mMask[2] = m2;   mMask[3] = m3;
            mMask[4] = m4;   mMask[5] = m5;
            mMask[6] = m6;   mMask[7] = m7;
            mMask[8] = m8;   mMask[9] = m9;
            mMask[10] = m10; mMask[11] = m11;
            mMask[12] = m12; mMask[13] = m13;
            mMask[14] = m14; mMask[15] = m15;
            mMask[16] = m16; mMask[17] = m17;
            mMask[18] = m18; mMask[19] = m19;
            mMask[20] = m20; mMask[21] = m21;
            mMask[22] = m22; mMask[23] = m23;
            mMask[24] = m24; mMask[25] = m25;
            mMask[26] = m26; mMask[27] = m27;
            mMask[28] = m28; mMask[29] = m29;
            mMask[30] = m30; mMask[31] = m31;
        }

        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const { return mMask[index]; }

        inline bool extract(uint32_t index)
        {
            return mMask[index];
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) {
            mMask[index] = x;
        }

        inline SIMDVecMask(SIMDVecMask const & mask) {
            for (int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = mask.mMask[i];
            }
        }
    };
}
}

#endif
