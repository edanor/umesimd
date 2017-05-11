// The MIT License (MIT)
//
// Copyright (c) 2015-2017 CERN
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

namespace UME {
namespace SIMD {

    template<uint32_t VEC_LEN>
    struct SIMDVecMask_traits {};

    // No specialized traits

    // MASK_BASE_TYPE is the type of element that will represent single entry in
    //                mask register. This can be for examle a 'bool' or 'unsigned int' or 'float'
    //                The actual representation depends on how the underlying instruction
    //                set handles the masks/mask registers. For scalar emulation the mask vetor should
    //                be represented using a boolean values. Bool in C++ has one disadventage: it is possible
    //                for the compiler to implicitly cast it to integer. To forbid this casting operations from
    //                happening the default type has to be wrapped into a class. 
    template<uint32_t VEC_LEN>
    class SIMDVecMask :
        public SIMDMaskBaseInterface<
            SIMDVecMask<VEC_LEN>,
            bool,
            VEC_LEN>
    {
    private:
        bool mMask[VEC_LEN]; // each entry represents single mask element. For real SIMD vectors, mMask will be of mask intrinsic type.

    public:
        constexpr static uint32_t alignment() {
            return VEC_LEN*sizeof(bool) > 16 ? 16 : VEC_LEN*sizeof(bool);
        }

        UME_FORCE_INLINE SIMDVecMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        UME_FORCE_INLINE SIMDVecMask(bool m) {
            for (unsigned int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = m;
            }
        }

        // LOAD-CONSTR - Construct by loading from memory
        UME_FORCE_INLINE explicit SIMDVecMask(bool const * p) { this->load(p); }

        // TODO: this should be handled using variadic templates, but unfortunatelly Visual Studio does not support this feature...
        UME_FORCE_INLINE SIMDVecMask(bool m0, bool m1)
        {
            mMask[0] = m0;
            mMask[1] = m1;
        }

        UME_FORCE_INLINE SIMDVecMask(bool m0, bool m1, bool m2, bool m3)
        {
            mMask[0] = m0;
            mMask[1] = m1;
            mMask[2] = m2;
            mMask[3] = m3;
        }

        UME_FORCE_INLINE SIMDVecMask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7)
        {
            mMask[0] = m0; mMask[1] = m1;
            mMask[2] = m2; mMask[3] = m3;
            mMask[4] = m4; mMask[5] = m5;
            mMask[6] = m6; mMask[7] = m7;
        }

        UME_FORCE_INLINE SIMDVecMask(bool m0, bool m1, bool m2, bool m3,
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

        UME_FORCE_INLINE SIMDVecMask(bool m0, bool m1, bool m2, bool m3,
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

        UME_FORCE_INLINE SIMDVecMask(
            bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7,
            bool m8, bool m9, bool m10, bool m11,
            bool m12, bool m13, bool m14, bool m15,
            bool m16, bool m17, bool m18, bool m19,
            bool m20, bool m21, bool m22, bool m23,
            bool m24, bool m25, bool m26, bool m27,
            bool m28, bool m29, bool m30, bool m31,
            bool m32, bool m33, bool m34, bool m35,
            bool m36, bool m37, bool m38, bool m39,
            bool m40, bool m41, bool m42, bool m43,
            bool m44, bool m45, bool m46, bool m47,
            bool m48, bool m49, bool m50, bool m51,
            bool m52, bool m53, bool m54, bool m55,
            bool m56, bool m57, bool m58, bool m59,
            bool m60, bool m61, bool m62, bool m63)
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
            mMask[32] = m32; mMask[33] = m33;
            mMask[34] = m34; mMask[35] = m35;
            mMask[36] = m36; mMask[37] = m37;
            mMask[38] = m38; mMask[39] = m39;
            mMask[40] = m40; mMask[41] = m41;
            mMask[42] = m42; mMask[43] = m43;
            mMask[44] = m44; mMask[45] = m45;
            mMask[46] = m46; mMask[47] = m47;
            mMask[48] = m48; mMask[49] = m49;
            mMask[50] = m50; mMask[51] = m51;
            mMask[52] = m52; mMask[53] = m53;
            mMask[54] = m54; mMask[55] = m55;
            mMask[56] = m56; mMask[57] = m57;
            mMask[58] = m58; mMask[59] = m59;
            mMask[60] = m60; mMask[61] = m61;
            mMask[62] = m62; mMask[63] = m63;
        }

        UME_FORCE_INLINE SIMDVecMask(
            bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7,
            bool m8, bool m9, bool m10, bool m11,
            bool m12, bool m13, bool m14, bool m15,
            bool m16, bool m17, bool m18, bool m19,
            bool m20, bool m21, bool m22, bool m23,
            bool m24, bool m25, bool m26, bool m27,
            bool m28, bool m29, bool m30, bool m31,
            bool m32, bool m33, bool m34, bool m35,
            bool m36, bool m37, bool m38, bool m39,
            bool m40, bool m41, bool m42, bool m43,
            bool m44, bool m45, bool m46, bool m47,
            bool m48, bool m49, bool m50, bool m51,
            bool m52, bool m53, bool m54, bool m55,
            bool m56, bool m57, bool m58, bool m59,
            bool m60, bool m61, bool m62, bool m63,
            bool m64, bool m65, bool m66, bool m67,
            bool m68, bool m69, bool m70, bool m71,
            bool m72, bool m73, bool m74, bool m75,
            bool m76, bool m77, bool m78, bool m79,
            bool m80, bool m81, bool m82, bool m83,
            bool m84, bool m85, bool m86, bool m87,
            bool m88, bool m89, bool m90, bool m91,
            bool m92, bool m93, bool m94, bool m95,
            bool m96, bool m97, bool m98, bool m99,
            bool m100, bool m101, bool m102, bool m103,
            bool m104, bool m105, bool m106, bool m107,
            bool m108, bool m109, bool m110, bool m111,
            bool m112, bool m113, bool m114, bool m115,
            bool m116, bool m117, bool m118, bool m119,
            bool m120, bool m121, bool m122, bool m123,
            bool m124, bool m125, bool m126, bool m127)
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
            mMask[32] = m32; mMask[33] = m33;
            mMask[34] = m34; mMask[35] = m35;
            mMask[36] = m36; mMask[37] = m37;
            mMask[38] = m38; mMask[39] = m39;
            mMask[40] = m40; mMask[41] = m41;
            mMask[42] = m42; mMask[43] = m43;
            mMask[44] = m44; mMask[45] = m45;
            mMask[46] = m46; mMask[47] = m47;
            mMask[48] = m48; mMask[49] = m49;
            mMask[50] = m50; mMask[51] = m51;
            mMask[52] = m52; mMask[53] = m53;
            mMask[54] = m54; mMask[55] = m55;
            mMask[56] = m56; mMask[57] = m57;
            mMask[58] = m58; mMask[59] = m59;
            mMask[60] = m60; mMask[61] = m61;
            mMask[62] = m62; mMask[63] = m63;
            mMask[64] = m64; mMask[65] = m65;
            mMask[66] = m66; mMask[67] = m67;
            mMask[68] = m68; mMask[69] = m69;
            mMask[70] = m70; mMask[71] = m71;
            mMask[72] = m72; mMask[73] = m73;
            mMask[74] = m74; mMask[75] = m75;
            mMask[76] = m76; mMask[77] = m77;
            mMask[78] = m78; mMask[79] = m79;
            mMask[80] = m80; mMask[81] = m81;
            mMask[82] = m82; mMask[83] = m83;
            mMask[84] = m84; mMask[85] = m85;
            mMask[86] = m86; mMask[87] = m87;
            mMask[88] = m88; mMask[89] = m89;
            mMask[90] = m90; mMask[91] = m91;
            mMask[92] = m92; mMask[93] = m93;
            mMask[94] = m94; mMask[95] = m95;
            mMask[96] = m96; mMask[97] = m97;
            mMask[98] = m98; mMask[99] = m99;
            mMask[100] = m100; mMask[101] = m101;
            mMask[102] = m102; mMask[103] = m103;
            mMask[104] = m104; mMask[105] = m105;
            mMask[106] = m106; mMask[107] = m107;
            mMask[108] = m108; mMask[109] = m109;
            mMask[110] = m110; mMask[111] = m111;
            mMask[112] = m112; mMask[113] = m113;
            mMask[114] = m114; mMask[115] = m115;
            mMask[116] = m116; mMask[117] = m117;
            mMask[118] = m118; mMask[119] = m119;
            mMask[120] = m120; mMask[121] = m121;
            mMask[122] = m122; mMask[123] = m123;
            mMask[124] = m124; mMask[125] = m125;
            mMask[126] = m126; mMask[127] = m127;
        }
        // A non-modifying element-wise access operator
        UME_FORCE_INLINE bool operator[] (uint32_t index) const { return mMask[index]; }

        UME_FORCE_INLINE bool extract(uint32_t index)
        {
            return mMask[index];
        }

        // Element-wise modification operator
        UME_FORCE_INLINE void insert(uint32_t index, bool x) {
            mMask[index] = x;
        }

        UME_FORCE_INLINE SIMDVecMask(SIMDVecMask const & mask) {
            for (unsigned int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = mask.mMask[i];
            }
        }
    };

}
}

#endif
