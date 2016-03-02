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
// This piece of code was developed as part of ICE-DIP project at CERN.
//  "ICE-DIP is a European Industrial Doctorate project funded by the European Community's 
//  7th Framework programme Marie Curie Actions under grant PITN-GA-2012-316596".
//

#include "UMEUnitTestDataSets64.h"

#include <limits>

// Below are pre-computed values for unit-tests.
// inputA, inputB, inputC, scalar and mask are used as inputs.
// Other arrays are used as model values used in comparison
const uint64_t DataSet_1_64u::inputs::inputA[16] = {
    0x68aa1ac70f070230, 0x29ee1c3601447519, 0x6c90197b74392d2f, 0x1daa5553417b7843,
    0x041d2a2f0c901357, 0x3232453f61ae5e52, 0x3a8117cf31c477f8, 0x2c1272d64fbf5618,
    0x19c300cc53062e07, 0x30e911bd0346414e, 0x233a6d9c317562a0, 0x67eb20d92211470f,
    0x181577356ca9777f, 0x6a5d18d6749f765a, 0x11d47d965f2c6d85, 0x041c2aa12aa47893
};  

const uint64_t DataSet_1_64u::inputs::inputB[16] = {
    0x68db109125c10425, 0x064d024b37a34994, 0x340a313e6b8e2fc4, 0x2edd1ed655ce702b,
    0x277a61c977d475bd, 0x4f13505551747955, 0x12261fc912513d57, 0x75991d467f952c93,
    0x4a0863c761e5606d, 0x295e335f0f94523b, 0x72a24dc267163bef, 0x38147c074db86bf6,
    0x537921ac068f1060, 0x14b300db75ae3d40, 0x3e9413eb239c3256, 0x45f4202a150623eb
};

const uint64_t DataSet_1_64u::inputs::inputC[16] = {
    0x692f54343c161013, 0x1de0111a73be2046, 0x40ff40d45c89343d, 0x77582f5323b25111,
    0x4b2e311931c34af6, 0x5676749d466541c0, 0x5d4302f1067746a7, 0x6c8233a2305d0f47,
    0x50e717e06a9b59eb, 0x7a2720de60a2670a, 0x1e68057c21e92c12, 0x31f52b2477402365,
    0x073715f122625da1, 0x2ae124173bc93825, 0x506a187f22c65cd1, 0x62c038b133d06f0c
};

const uint64_t DataSet_1_64u::inputs::inputShiftA[16] = {
    4,  4,  6,  28, 21, 24, 31, 28,
    30, 23, 26, 22, 8,  26, 8,  7
};

const uint64_t DataSet_1_64u::inputs::scalarA = 636364;
const uint64_t DataSet_1_64u::inputs::inputShiftScalarA = 27;

const bool DataSet_1_64u::inputs::maskA[16] = {
    false,   false,  false,  true,  // 4
    false,  true,   true,   false,  // 8

    false,  true,   true,   false,
    true,   false,  false,  true    // 16
};

const uint64_t DataSet_1_64u::outputs::ADDV[16] = {
    0xd1852b5834c80655,     0x303b1e8138e7bead,     0xa09a4ab9dfc75cf3,     0x4c8774299749e86e,
    0x2b978bf884648914,     0x81459594b322d7a7,     0x4ca737984415b54f,     0xa1ab901ccf5482ab,
    0x63cb6493b4eb8e74,     0x5a47451c12da9389,     0x95dcbb5e988b9e8f,     0x9fff9ce06fc9b305,
    0x6b8e98e1733887df,     0x7f1019b1ea4db39a,     0x5068918182c89fdb,     0x4a104acb3faa9c7e
};

const uint64_t DataSet_1_64u::outputs::MADDV[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x4c8774299749e86e,
    0x041d2a2f0c901357,     0x81459594b322d7a7,     0x4ca737984415b54f,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x5a47451c12da9389,     0x95dcbb5e988b9e8f,     0x67eb20d92211470f,
    0x6b8e98e1733887df,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x4a104acb3faa9c7e
};

const uint64_t DataSet_1_64u::outputs::ADDS[16] = {
    0x68aa1ac70f10b7fc,     0x29ee1c36014e2ae5,     0x6c90197b7442e2fb,     0x1daa555341852e0f,
    0x041d2a2f0c99c923,     0x3232453f61b8141e,     0x3a8117cf31ce2dc4,     0x2c1272d64fc90be4,
    0x19c300cc530fe3d3,     0x30e911bd034ff71a,     0x233a6d9c317f186c,     0x67eb20d9221afcdb,
    0x181577356cb32d4b,     0x6a5d18d674a92c26,     0x11d47d965f362351,     0x041c2aa12aae2e5f
};

const uint64_t DataSet_1_64u::outputs::MADDS[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x1daa555341852e0f,
    0x041d2a2f0c901357,     0x3232453f61b8141e,     0x3a8117cf31ce2dc4,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x30e911bd034ff71a,     0x233a6d9c317f186c,     0x67eb20d92211470f,
    0x181577356cb32d4b,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x041c2aa12aae2e5f
};

const uint64_t DataSet_1_64u::outputs::POSTPREFINC[16] = {
    0x68aa1ac70f070231,     0x29ee1c360144751a,     0x6c90197b74392d30,     0x1daa5553417b7844,
    0x041d2a2f0c901358,     0x3232453f61ae5e53,     0x3a8117cf31c477f9,     0x2c1272d64fbf5619,
    0x19c300cc53062e08,     0x30e911bd0346414f,     0x233a6d9c317562a1,     0x67eb20d922114710,
    0x181577356ca97780,     0x6a5d18d6749f765b,     0x11d47d965f2c6d86,     0x041c2aa12aa47894
};

const uint64_t DataSet_1_64u::outputs::MPOSTPREFINC[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x1daa5553417b7844,
    0x041d2a2f0c901357,     0x3232453f61ae5e53,     0x3a8117cf31c477f9,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x30e911bd0346414f,     0x233a6d9c317562a1,     0x67eb20d92211470f,
    0x181577356ca97780,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x041c2aa12aa47894
};

const uint64_t DataSet_1_64u::outputs::SUBV[16] = {
    0xffcf0a35e945fe0b,     0x23a119eac9a12b85,     0x3885e83d08aafd6b,     0xeecd367cebad0818,
    0xdca2c86594bb9d9a,     0xe31ef4ea1039e4fd,     0x285af8061f733aa1,     0xb679558fd02a2985,
    0xcfba9d04f120cd9a,     0x078ade5df3b1ef13,     0xb0981fd9ca5f26b1,     0x2fd6a4d1d458db19,
    0xc49c5589661a671f,     0x55aa17fafef1391a,     0xd34069ab3b903b2f,     0xbe280a77159e54a8
};

const uint64_t DataSet_1_64u::outputs::MSUBV[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0xeecd367cebad0818,
    0x041d2a2f0c901357,     0xe31ef4ea1039e4fd,     0x285af8061f733aa1,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x078ade5df3b1ef13,     0xb0981fd9ca5f26b1,     0x67eb20d92211470f,
    0xc49c5589661a671f,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0xbe280a77159e54a8
};

const uint64_t DataSet_1_64u::outputs::SUBS[16] = {
    0x68aa1ac70efd4c64,     0x29ee1c36013abf4d,     0x6c90197b742f7763,     0x1daa55534171c277,
    0x041d2a2f0c865d8b,     0x3232453f61a4a886,     0x3a8117cf31bac22c,     0x2c1272d64fb5a04c,
    0x19c300cc52fc783b,     0x30e911bd033c8b82,     0x233a6d9c316bacd4,     0x67eb20d922079143,
    0x181577356c9fc1b3,     0x6a5d18d67495c08e,     0x11d47d965f22b7b9,     0x041c2aa12a9ac2c7
};

const uint64_t DataSet_1_64u::outputs::MSUBS[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x1daa55534171c277,
    0x041d2a2f0c901357,     0x3232453f61a4a886,     0x3a8117cf31bac22c,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x30e911bd033c8b82,     0x233a6d9c316bacd4,     0x67eb20d92211470f,
    0x181577356c9fc1b3,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x041c2aa12a9ac2c7
};

const uint64_t DataSet_1_64u::outputs::SUBFROMV[16] = {
    0x0030f5ca16ba01f5,     0xdc5ee615365ed47b,     0xc77a17c2f7550295,     0x1132c9831452f7e8,
    0x235d379a6b446266,     0x1ce10b15efc61b03,     0xd7a507f9e08cc55f,     0x4986aa702fd5d67b,
    0x304562fb0edf3266,     0xf87521a20c4e10ed,     0x4f67e02635a0d94f,     0xd0295b2e2ba724e7,
    0x3b63aa7699e598e1,     0xaa55e805010ec6e6,     0x2cbf9654c46fc4d1,     0x41d7f588ea61ab58
};

const uint64_t DataSet_1_64u::outputs::MSUBFROMV[16] = {
    0x68db109125c10425,     0x064d024b37a34994,     0x340a313e6b8e2fc4,     0x1132c9831452f7e8,
    0x277a61c977d475bd,     0x1ce10b15efc61b03,     0xd7a507f9e08cc55f,     0x75991d467f952c93,
    0x4a0863c761e5606d,     0xf87521a20c4e10ed,     0x4f67e02635a0d94f,     0x38147c074db86bf6,
    0x3b63aa7699e598e1,     0x14b300db75ae3d40,     0x3e9413eb239c3256,     0x41d7f588ea61ab58
};

const uint64_t DataSet_1_64u::outputs::SUBFROMS[16] = {
    0x9755e538f102b39c,     0xd611e3c9fec540b3,     0x936fe6848bd0889d,     0xe255aaacbe8e3d89,
    0xfbe2d5d0f379a275,     0xcdcdbac09e5b577a,     0xc57ee830ce453dd4,     0xd3ed8d29b04a5fb4,
    0xe63cff33ad0387c5,     0xcf16ee42fcc3747e,     0xdcc59263ce94532c,     0x9814df26ddf86ebd,
    0xe7ea88ca93603e4d,     0x95a2e7298b6a3f72,     0xee2b8269a0dd4847,     0xfbe3d55ed5653d39
};

const uint64_t DataSet_1_64u::outputs::MSUBFROMS[16] = {
    0x000000000009b5cc,     0x000000000009b5cc,     0x000000000009b5cc,     0xe255aaacbe8e3d89,
    0x000000000009b5cc,     0xcdcdbac09e5b577a,     0xc57ee830ce453dd4,     0x000000000009b5cc,
    0x000000000009b5cc,     0xcf16ee42fcc3747e,     0xdcc59263ce94532c,     0x000000000009b5cc,
    0xe7ea88ca93603e4d,     0x000000000009b5cc,     0x000000000009b5cc,     0xfbe3d55ed5653d39
};

const uint64_t DataSet_1_64u::outputs::POSTPREFDEC[16] = {
    0x68aa1ac70f07022f,     0x29ee1c3601447518,     0x6c90197b74392d2e,     0x1daa5553417b7842,
    0x041d2a2f0c901356,     0x3232453f61ae5e51,     0x3a8117cf31c477f7,     0x2c1272d64fbf5617,
    0x19c300cc53062e06,     0x30e911bd0346414d,     0x233a6d9c3175629f,     0x67eb20d92211470e,
    0x181577356ca9777e,     0x6a5d18d6749f7659,     0x11d47d965f2c6d84,     0x041c2aa12aa47892
};

const uint64_t DataSet_1_64u::outputs::MPOSTPREFDEC[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x1daa5553417b7842,
    0x041d2a2f0c901357,     0x3232453f61ae5e51,     0x3a8117cf31c477f7,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x30e911bd0346414d,     0x233a6d9c3175629f,     0x67eb20d92211470f,
    0x181577356ca9777e,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x041c2aa12aa47892
};

const uint64_t DataSet_1_64u::outputs::MULV[16] = {
    0x106a8fcade3c10f0,     0x6ad9b9a72ee2d374,     0xb6647ad7302438fc,     0xdfd2d3ed09448341,
    0x18862c1694410a3b,     0x1681534b88a2133a,     0xdb333f971fd2dd48,     0x8a0b5f67b1a28fc8,
    0x9e620e8a7e2738fb,     0xf09569ff953408fa,     0x1100ac3d1411f360,     0x300c36178b158d6a,
    0xb92e5e9e10f5bfa0,     0x3899b3deba3d0880,     0xc55c093d925cc4ae,     0xf11fd0a383e8c7f1
};

const uint64_t DataSet_1_64u::outputs::MMULV[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0xdfd2d3ed09448341,
    0x041d2a2f0c901357,     0x1681534b88a2133a,     0xdb333f971fd2dd48,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0xf09569ff953408fa,     0x1100ac3d1411f360,     0x67eb20d92211470f,
    0xb92e5e9e10f5bfa0,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0xf11fd0a383e8c7f1
};

const uint64_t DataSet_1_64u::outputs::MULS[16] = {
    0x9d7b9b7f01d1ae40,     0xc196b5568738fcec,     0x4a2fad8fa1293c74,     0x11fd28fbb4c63464,
    0x61b852708e8aec54,     0xfa3f6db370852358,     0xbafd3733c919f1a0,     0x336dd6e3db2f9320,
    0x6e2404bd25b6a094,     0x62e9e36793e63028,     0x3c8c848fdfe5b780,     0x3539e4b8dc893af4,
    0x8f44965b9bff0434,     0x744984f4d6e9f1b8,     0xdc68bdae1a834efc,     0xb0406b5c817b0424
};

const uint64_t DataSet_1_64u::outputs::MMULS[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x11fd28fbb4c63464,
    0x041d2a2f0c901357,     0xfa3f6db370852358,     0xbafd3733c919f1a0,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x62e9e36793e63028,     0x3c8c848fdfe5b780,     0x67eb20d92211470f,
    0x8f44965b9bff0434,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0xb0406b5c817b0424
};

const uint64_t DataSet_1_64u::outputs::DIVV[16] = {
    0x0000000000000000,     0x0000000000000006,     0x0000000000000002,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000003,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000001,     0x0000000000000000,     0x0000000000000001,
    0x0000000000000000,     0x0000000000000005,     0x0000000000000000,     0x0000000000000000
};

const uint64_t DataSet_1_64u::outputs::MDIVV[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x0000000000000000,
    0x041d2a2f0c901357,     0x0000000000000000,     0x0000000000000003,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x0000000000000001,     0x0000000000000000,     0x67eb20d92211470f,
    0x0000000000000000,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x0000000000000000
};

const uint64_t DataSet_1_64u::outputs::DIVS[16] = {
    0x00000ac764a31049,     0x000004517407938b,     0x00000b2e2bf4eafd,     0x0000030e1a61af67,
    0x0000006c75d3d23c,     0x0000052b62dd237c,     0x000006066acc65aa,     0x00000489ec85b345,
    0x000002a72fdbf67e,     0x000005097bbe50ee,     0x000003a0c37a1eb6,     0x00000ab3b9b41756,
    0x0000027af374a8eb,     0x00000af430db2c65,     0x000001d6130eda89,     0x0000006c5b8257d9
};

const uint64_t DataSet_1_64u::outputs::MDIVS[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x0000030e1a61af67,
    0x041d2a2f0c901357,     0x0000052b62dd237c,     0x000006066acc65aa,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x000005097bbe50ee,     0x000003a0c37a1eb6,     0x67eb20d92211470f,
    0x0000027af374a8eb,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x0000006c5b8257d9
};

const uint64_t DataSet_1_64u::outputs::RCP[16] = {
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000
};

const uint64_t DataSet_1_64u::outputs::MRCP[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x0000000000000000,
    0x041d2a2f0c901357,     0x0000000000000000,     0x0000000000000000,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x0000000000000000,     0x0000000000000000,     0x67eb20d92211470f,
    0x0000000000000000,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x0000000000000000
};

const uint64_t DataSet_1_64u::outputs::RCPS[16] = {
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000
};

const uint64_t DataSet_1_64u::outputs::MRCPS[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x0000000000000000,
    0x041d2a2f0c901357,     0x0000000000000000,     0x0000000000000000,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x0000000000000000,     0x0000000000000000,     0x67eb20d92211470f,
    0x0000000000000000,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x0000000000000000
};

const bool DataSet_1_64u::outputs::CMPEQV[16] = {
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false
};

const bool DataSet_1_64u::outputs::CMPEQS[16] = {
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false
};

const bool DataSet_1_64u::outputs::CMPNEV[16] = {
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true
};

const bool DataSet_1_64u::outputs::CMPNES[16] = {
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true
};

const bool DataSet_1_64u::outputs::CMPGTV[16] = {
    false,  true,   true,   false,
    false,  false,  true,   false,
    false,  true,   false,  true,
    false,  true,   false,  false
};

const bool DataSet_1_64u::outputs::CMPGTS[16] = {
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true
};

const bool DataSet_1_64u::outputs::CMPLTV[16] = {
    true,   false,  false,  true,
    true,   true,   false,  true,
    true,   false,  true,   false,
    true,   false,  true,   true
};

const bool DataSet_1_64u::outputs::CMPLTS[16] = {
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false
};

const bool DataSet_1_64u::outputs::CMPGEV[16] = {
    false,  true,   true,   false,
    false,  false,  true,   false,
    false,  true,   false,  true,
    false,  true,   false,  false
};

const bool DataSet_1_64u::outputs::CMPGES[16] = {
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true
};

const bool DataSet_1_64u::outputs::CMPLEV[16] = {
    true,   false,  false,  true,
    true,   true,   false,  true,
    true,   false,  true,   false,
    true,   false,  true,   true
};

const bool DataSet_1_64u::outputs::CMPLES[16] = {
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false
};

const bool  DataSet_1_64u::outputs::CMPEV = false;
const bool  DataSet_1_64u::outputs::CMPES = false;


const uint64_t DataSet_1_64u::outputs::HADD[16] = {
    0x68aa1ac70f070230,     0x929836fd104b7749,     0xff2850788484a478,     0x1cd2a5cbc6001cbb,
    0x20efcffad2903012,     0x5322153a343e8e64,     0x8da32d096603065c,     0xb9b59fdfb5c25c74,
    0xd378a0ac08c88a7b,     0x0461b2690c0ecbc9,     0x279c20053d842e69,     0x8f8740de5f957578,
    0xa79cb813cc3eecf7,     0x11f9d0ea40de6351,     0x23ce4e80a00ad0d6,     0x27ea7921caaf4969
};

const uint64_t DataSet_1_64u::outputs::MHADD[16] = {
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x1daa5553417b7843,
    0x1daa5553417b7843,     0x4fdc9a92a329d695,     0x8a5db261d4ee4e8d,     0x8a5db261d4ee4e8d,
    0x8a5db261d4ee4e8d,     0xbb46c41ed8348fdb,     0xde8131bb09a9f27b,     0xde8131bb09a9f27b,
    0xf696a8f0765369fa,     0xf696a8f0765369fa,     0xf696a8f0765369fa,     0xfab2d391a0f7e28d
};

const uint64_t DataSet_1_64u::outputs::HMUL[16] = {
    0x68aa1ac70f070230,     0x8320524d706f26b0,     0x0ec5e5d28b650a50,     0x2a62608c1cb732f0,
    0x6defad3f420a1f90,     0x368705c9e2b4fc20,     0x889368a8e9071f00,     0xaeab86ff5e14e800,
    0x3f186f98c4425800,     0x20ad338cb48ed000,     0x2d70b02494e20000,     0xec3a5e71673e0000,
    0xb51ce12e09c20000,     0x0c225a2cda340000,     0x5164bf2581040000,     0x5d28a592f54c0000
};

const uint64_t DataSet_1_64u::outputs::MHMUL[16] = {
    0x0000000000000001,     0x0000000000000001,     0x0000000000000001,     0x1daa5553417b7843,
    0x1daa5553417b7843,     0x8e2cd8b0703f1f76,     0x4611feddc11e5450,     0x4611feddc11e5450,
    0x4611feddc11e5450,     0x88e5adf188860060,     0xac3953092bc4fc00,     0xac3953092bc4fc00,
    0x127aa03fa3dd0400,     0x127aa03fa3dd0400,     0x127aa03fa3dd0400,     0xf1d6999941c94c00
};

const uint64_t DataSet_1_64u::outputs::FMULADDV[16] = {
    0x7999e3ff1a522103,     0x88b9cac1a2a0f3ba,     0xf763bbab8cad6d39,     0x572b03402cf6d452,
    0x63b45d2fc6045531,     0x6cf7c7e8cf0754fa,     0x38764288264a23ef,     0xf68d9309e1ff9f0f,
    0xef49266ae8c292e6,     0x6abc8addf5d67004,     0x2f68b1b935fb1f72,     0x6201613c0255b0cf,
    0xc065748f33581d41,     0x637ad7f5f60640a5,     0x15c621bcb523217f,     0x53e00954b7b936fd
};

const uint64_t DataSet_1_64u::outputs::MFMULADDV[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x572b03402cf6d452,
    0x041d2a2f0c901357,     0x6cf7c7e8cf0754fa,     0x38764288264a23ef,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x6abc8addf5d67004,     0x2f68b1b935fb1f72,     0x67eb20d92211470f,
    0xc065748f33581d41,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x53e00954b7b936fd
};

const uint64_t DataSet_1_64u::outputs::FMULSUBV[16] = {
    0xa73b3b96a22600dd,     0x4cf9a88cbb24b32e,     0x75653a02d39b04bf,     0x687aa499e5923230,
    0xcd57fafd627dbf45,     0xc00adeae423cd17a,     0x7df03ca6195b96a1,     0x1d892bc581458081,
    0x4d7af6aa138bdf10,     0x766e49213491a1f0,     0xf298a6c0f228c74e,     0xfe170af313d56a05,
    0xb1f748acee9361ff,     0x0db88fc77e73d05b,     0x74f1f0be6f9667dd,     0x8e5f97f2501858e5
};

const uint64_t DataSet_1_64u::outputs::MFMULSUBV[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x687aa499e5923230,
    0x041d2a2f0c901357,     0xc00adeae423cd17a,     0x7df03ca6195b96a1,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x766e49213491a1f0,     0xf298a6c0f228c74e,     0x67eb20d92211470f,
    0xb1f748acee9361ff,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x8e5f97f2501858e5
};

const uint64_t DataSet_1_64u::outputs::FADDMULV[16] = {
    0xa6db52c8e28bc84f,     0xe005afabc299c34e,     0xe438b012e36d81e7,     0x0bccb97614ef3d4e,
    0x8b60a4d986778138,     0x72e11223f1c62440,     0x8e305ca87b75e089,     0x2db6938cfc37426d,
    0xa299d29840fe187c,     0xe648f5216a97e25a,     0xfe4ddda67838ba0e,     0x80964c2cc24d4ff9,
    0x4f19de43a147763f,     0x293a4c345e6ea542,     0x4864f7c7989f35cb,     0xbdb61d0c3639f7e8
};

const uint64_t DataSet_1_64u::outputs::MFADDMULV[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x0bccb97614ef3d4e,
    0x041d2a2f0c901357,     0x72e11223f1c62440,     0x8e305ca87b75e089,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0xe648f5216a97e25a,     0xfe4ddda67838ba0e,     0x67eb20d92211470f,
    0x4f19de43a147763f,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0xbdb61d0c3639f7e8
};

const uint64_t DataSet_1_64u::outputs::FSUBMULV[16] = {
    0xf0e7ca5f19048ad1,     0xe9c1bc145338865e,     0xfd656f7dde7b1e7f,     0xdec8b3104ebd2198,
    0x085b00b8ae25f5fc,     0x8e116c486460fac0,     0xf9d13e190d0a4507,     0x166e745939714ee3,
    0x43dd2c1c00d5465e,     0x29d01f728629fbbe,     0xf83b536b9d702472,     0x379b1d645242dbdd,
    0xb986ec7d65ef1d7f,     0x8b35ebf070c4f0c2,     0x5978cbee7a5f355f,     0xe377839eeaa0cfe0
};

const uint64_t DataSet_1_64u::outputs::MFSUBMULV[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0xdec8b3104ebd2198,
    0x041d2a2f0c901357,     0x8e116c486460fac0,     0xf9d13e190d0a4507,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x29d01f728629fbbe,     0xf83b536b9d702472,     0x67eb20d92211470f,
    0xb986ec7d65ef1d7f,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0xe377839eeaa0cfe0
};

const uint64_t DataSet_1_64u::outputs::MAXV[16] = {
    0x68db109125c10425,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x2edd1ed655ce702b,
    0x277a61c977d475bd,     0x4f13505551747955,     0x3a8117cf31c477f8,     0x75991d467f952c93,
    0x4a0863c761e5606d,     0x30e911bd0346414e,     0x72a24dc267163bef,     0x67eb20d92211470f,
    0x537921ac068f1060,     0x6a5d18d6749f765a,     0x3e9413eb239c3256,     0x45f4202a150623eb
};

const uint64_t DataSet_1_64u::outputs::MMAXV[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x2edd1ed655ce702b,
    0x041d2a2f0c901357,     0x4f13505551747955,     0x3a8117cf31c477f8,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x30e911bd0346414e,     0x72a24dc267163bef,     0x67eb20d92211470f,
    0x537921ac068f1060,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x45f4202a150623eb
};

const uint64_t DataSet_1_64u::outputs::MAXS[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x1daa5553417b7843,
    0x041d2a2f0c901357,     0x3232453f61ae5e52,     0x3a8117cf31c477f8,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x30e911bd0346414e,     0x233a6d9c317562a0,     0x67eb20d92211470f,
    0x181577356ca9777f,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x041c2aa12aa47893
};

const uint64_t DataSet_1_64u::outputs::MMAXS[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x1daa5553417b7843,
    0x041d2a2f0c901357,     0x3232453f61ae5e52,     0x3a8117cf31c477f8,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x30e911bd0346414e,     0x233a6d9c317562a0,     0x67eb20d92211470f,
    0x181577356ca9777f,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x041c2aa12aa47893
};

const uint64_t DataSet_1_64u::outputs::MINV[16] = {
    0x68aa1ac70f070230,     0x064d024b37a34994,     0x340a313e6b8e2fc4,     0x1daa5553417b7843,
    0x041d2a2f0c901357,     0x3232453f61ae5e52,     0x12261fc912513d57,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x295e335f0f94523b,     0x233a6d9c317562a0,     0x38147c074db86bf6,
    0x181577356ca9777f,     0x14b300db75ae3d40,     0x11d47d965f2c6d85,     0x041c2aa12aa47893
};

const uint64_t DataSet_1_64u::outputs::MMINV[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x1daa5553417b7843,
    0x041d2a2f0c901357,     0x3232453f61ae5e52,     0x12261fc912513d57,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x295e335f0f94523b,     0x233a6d9c317562a0,     0x67eb20d92211470f,
    0x181577356ca9777f,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x041c2aa12aa47893
};

const uint64_t DataSet_1_64u::outputs::MINS[16] = {
    0x000000000009b5cc,     0x000000000009b5cc,     0x000000000009b5cc,     0x000000000009b5cc,
    0x000000000009b5cc,     0x000000000009b5cc,     0x000000000009b5cc,     0x000000000009b5cc,
    0x000000000009b5cc,     0x000000000009b5cc,     0x000000000009b5cc,     0x000000000009b5cc,
    0x000000000009b5cc,     0x000000000009b5cc,     0x000000000009b5cc,     0x000000000009b5cc
};

const uint64_t DataSet_1_64u::outputs::MMINS[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x000000000009b5cc,
    0x041d2a2f0c901357,     0x000000000009b5cc,     0x000000000009b5cc,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x000000000009b5cc,     0x000000000009b5cc,     0x67eb20d92211470f,
    0x000000000009b5cc,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x000000000009b5cc
};

const uint64_t DataSet_1_64u::outputs::HMAX[16] = {
    0x68aa1ac70f070230,     0x68aa1ac70f070230,     0x6c90197b74392d2f,     0x6c90197b74392d2f,
    0x6c90197b74392d2f,     0x6c90197b74392d2f,     0x6c90197b74392d2f,     0x6c90197b74392d2f,
    0x6c90197b74392d2f,     0x6c90197b74392d2f,     0x6c90197b74392d2f,     0x6c90197b74392d2f,
    0x6c90197b74392d2f,     0x6c90197b74392d2f,     0x6c90197b74392d2f,     0x6c90197b74392d2f
};

const uint64_t DataSet_1_64u::outputs::MHMAX[16] = {
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x1daa5553417b7843,
    0x1daa5553417b7843,     0x3232453f61ae5e52,     0x3a8117cf31c477f8,     0x3a8117cf31c477f8,
    0x3a8117cf31c477f8,     0x3a8117cf31c477f8,     0x3a8117cf31c477f8,     0x3a8117cf31c477f8,
    0x3a8117cf31c477f8,     0x3a8117cf31c477f8,     0x3a8117cf31c477f8,     0x3a8117cf31c477f8
};

const uint64_t DataSet_1_64u::outputs::HMIN[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x29ee1c3601447519,     0x1daa5553417b7843,
    0x041d2a2f0c901357,     0x041d2a2f0c901357,     0x041d2a2f0c901357,     0x041d2a2f0c901357,
    0x041d2a2f0c901357,     0x041d2a2f0c901357,     0x041d2a2f0c901357,     0x041d2a2f0c901357,
    0x041d2a2f0c901357,     0x041d2a2f0c901357,     0x041d2a2f0c901357,     0x041c2aa12aa47893
};

const uint64_t DataSet_1_64u::outputs::MHMIN[16] = {
    0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,     0x1daa5553417b7843,
    0x1daa5553417b7843,     0x1daa5553417b7843,     0x1daa5553417b7843,     0x1daa5553417b7843,
    0x1daa5553417b7843,     0x1daa5553417b7843,     0x1daa5553417b7843,     0x1daa5553417b7843,
    0x181577356ca9777f,     0x181577356ca9777f,     0x181577356ca9777f,     0x041c2aa12aa47893
};

const uint64_t DataSet_1_64u::outputs::BANDV[16] = {
    0x688a108105010020,     0x004c000201004110,     0x2400113a60082d04,     0x0c881452414a7003,
    0x0418200904901115,     0x0212401541245850,     0x120017c910403550,     0x241010464f950410,
    0x080000c441042005,     0x2048111d0304400a,     0x22224d80211422a0,     0x2000200100104306,
    0x1011212404891060,     0x001100d2748e3440,     0x10941182030c2004,     0x0414202000042083
};

const uint64_t DataSet_1_64u::outputs::MBANDV[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x0c881452414a7003,
    0x041d2a2f0c901357,     0x0212401541245850,     0x120017c910403550,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x2048111d0304400a,     0x22224d80211422a0,     0x67eb20d92211470f,
    0x1011212404891060,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x0414202000042083
};

const uint64_t DataSet_1_64u::outputs::BANDS[16] = {
    0x0000000000010000,     0x0000000000003508,     0x000000000009250c,     0x0000000000093040,
    0x0000000000001144,     0x0000000000081440,     0x00000000000035c8,     0x0000000000091408,
    0x0000000000002404,     0x000000000000014c,     0x0000000000012080,     0x000000000001050c,
    0x000000000009354c,     0x0000000000093448,     0x0000000000082584,     0x0000000000003080
};

const uint64_t DataSet_1_64u::outputs::MBANDS[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x0000000000093040,
    0x041d2a2f0c901357,     0x0000000000081440,     0x00000000000035c8,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x000000000000014c,     0x0000000000012080,     0x67eb20d92211470f,
    0x000000000009354c,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x0000000000003080
};

const uint64_t DataSet_1_64u::outputs::BORV[16] = {
    0x68fb1ad72fc70635,     0x2fef1e7f37e77d9d,     0x7c9a397f7fbf2fef,     0x3fff5fd755ff786b,
    0x277f6bef7fd477ff,     0x7f33557f71fe7f57,     0x3aa71fcf33d57fff,     0x7d9b7fd67fbf7e9b,
    0x5bcb63cf73e76e6f,     0x39ff33ff0fd6537f,     0x73ba6dde77777bef,     0x7fff7cdf6fb96fff,
    0x5b7d77bd6eaf777f,     0x7eff18df75bf7f5a,     0x3fd47fff7fbc7fd7,     0x45fc2aab3fa67bfb
};

const uint64_t DataSet_1_64u::outputs::MBORV[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x3fff5fd755ff786b,
    0x041d2a2f0c901357,     0x7f33557f71fe7f57,     0x3aa71fcf33d57fff,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x39ff33ff0fd6537f,     0x73ba6dde77777bef,     0x67eb20d92211470f,
    0x5b7d77bd6eaf777f,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x45fc2aab3fa67bfb
};

const uint64_t DataSet_1_64u::outputs::BORS[16] = {
    0x68aa1ac70f0fb7fc,     0x29ee1c36014df5dd,     0x6c90197b7439bdef,     0x1daa5553417bfdcf,
    0x041d2a2f0c99b7df,     0x3232453f61afffde,     0x3a8117cf31cdf7fc,     0x2c1272d64fbff7dc,
    0x19c300cc530fbfcf,     0x30e911bd034ff5ce,     0x233a6d9c317df7ec,     0x67eb20d92219f7cf,
    0x181577356ca9f7ff,     0x6a5d18d6749ff7de,     0x11d47d965f2dfdcd,     0x041c2aa12aadfddf
};

const uint64_t DataSet_1_64u::outputs::MBORS[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x1daa5553417bfdcf,
    0x041d2a2f0c901357,     0x3232453f61afffde,     0x3a8117cf31cdf7fc,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x30e911bd034ff5ce,     0x233a6d9c317df7ec,     0x67eb20d92211470f,
    0x181577356ca9f7ff,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x041c2aa12aadfddf
};

const uint64_t DataSet_1_64u::outputs::BXORV[16] = {
    0x00710a562ac60615,     0x2fa31e7d36e73c8d,     0x589a28451fb702eb,     0x33774b8514b50868,
    0x23674be67b4466ea,     0x7d21156a30da2707,     0x28a7080623954aaf,     0x598b6f90302a7a8b,
    0x53cb630b32e34e6a,     0x19b722e20cd21375,     0x5198205e5663594f,     0x5fff5cde6fa92cf9,
    0x4b6c56996a26671f,     0x7eee180d01314b1a,     0x2f406e7d7cb05fd3,     0x41e80a8b3fa25b78
};

const uint64_t DataSet_1_64u::outputs::MBXORV[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x33774b8514b50868,
    0x041d2a2f0c901357,     0x7d21156a30da2707,     0x28a7080623954aaf,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x19b722e20cd21375,     0x5198205e5663594f,     0x67eb20d92211470f,
    0x4b6c56996a26671f,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x41e80a8b3fa25b78
};

const uint64_t DataSet_1_64u::outputs::BXORS[16] = {
    0x68aa1ac70f0eb7fc,     0x29ee1c36014dc0d5,     0x6c90197b743098e3,     0x1daa55534172cd8f,
    0x041d2a2f0c99a69b,     0x3232453f61a7eb9e,     0x3a8117cf31cdc234,     0x2c1272d64fb6e3d4,
    0x19c300cc530f9bcb,     0x30e911bd034ff482,     0x233a6d9c317cd76c,     0x67eb20d92218f2c3,
    0x181577356ca0c2b3,     0x6a5d18d67496c396,     0x11d47d965f25d849,     0x041c2aa12aadcd5f
};

const uint64_t DataSet_1_64u::outputs::MBXORS[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x1daa55534172cd8f,
    0x041d2a2f0c901357,     0x3232453f61a7eb9e,     0x3a8117cf31cdc234,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x30e911bd034ff482,     0x233a6d9c317cd76c,     0x67eb20d92211470f,
    0x181577356ca0c2b3,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x041c2aa12aadcd5f
};

const uint64_t DataSet_1_64u::outputs::BNOT[16] = {
    0x9755e538f0f8fdcf,     0xd611e3c9febb8ae6,     0x936fe6848bc6d2d0,     0xe255aaacbe8487bc,
    0xfbe2d5d0f36feca8,     0xcdcdbac09e51a1ad,     0xc57ee830ce3b8807,     0xd3ed8d29b040a9e7,
    0xe63cff33acf9d1f8,     0xcf16ee42fcb9beb1,     0xdcc59263ce8a9d5f,     0x9814df26ddeeb8f0,
    0xe7ea88ca93568880,     0x95a2e7298b6089a5,     0xee2b8269a0d3927a,     0xfbe3d55ed55b876c
};

const uint64_t DataSet_1_64u::outputs::MBNOT[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0xe255aaacbe8487bc,
    0x041d2a2f0c901357,     0xcdcdbac09e51a1ad,     0xc57ee830ce3b8807,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0xcf16ee42fcb9beb1,     0xdcc59263ce8a9d5f,     0x67eb20d92211470f,
    0xe7ea88ca93568880,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0xfbe3d55ed55b876c
};

const uint64_t DataSet_1_64u::outputs::HBAND[16] = {
    0x68aa1ac70f070230,     0x28aa180601040010,     0x2880180200000000,     0x0880100200000000,
    0x0000000200000000,     0x0000000200000000,     0x0000000200000000,     0x0000000200000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000
};

const uint64_t DataSet_1_64u::outputs::MHBAND[16] = {
    0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,     0x1daa5553417b7843,
    0x1daa5553417b7843,     0x10224513412a5842,     0x1000050301005040,     0x1000050301005040,
    0x1000050301005040,     0x1000010101004040,     0x0000010001004000,     0x0000010001004000,
    0x0000010000004000,     0x0000010000004000,     0x0000010000004000,     0x0000000000004000
};

const uint64_t DataSet_1_64u::outputs::HBANDS[16] = {
    0x0000000000010000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000
};

const uint64_t DataSet_1_64u::outputs::MHBANDS[16] = {
    0x000000000009b5cc,     0x000000000009b5cc,     0x000000000009b5cc,     0x0000000000093040,
    0x0000000000093040,     0x0000000000081040,     0x0000000000001040,     0x0000000000001040,
    0x0000000000001040,     0x0000000000000040,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000
};

const uint64_t DataSet_1_64u::outputs::HBOR[16] = {
    0x68aa1ac70f070230,     0x69ee1ef70f477739,     0x6dfe1fff7f7f7f3f,     0x7dfe5fff7f7f7f7f,
    0x7dff7fff7fff7f7f,     0x7fff7fff7fff7f7f,     0x7fff7fff7fff7fff,     0x7fff7fff7fff7fff,
    0x7fff7fff7fff7fff,     0x7fff7fff7fff7fff,     0x7fff7fff7fff7fff,     0x7fff7fff7fff7fff,
    0x7fff7fff7fff7fff,     0x7fff7fff7fff7fff,     0x7fff7fff7fff7fff,     0x7fff7fff7fff7fff
};

const uint64_t DataSet_1_64u::outputs::MHBOR[16] = {
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x1daa5553417b7843,
    0x1daa5553417b7843,     0x3fba557f61ff7e53,     0x3fbb57ff71ff7ffb,     0x3fbb57ff71ff7ffb,
    0x3fbb57ff71ff7ffb,     0x3ffb57ff73ff7fff,     0x3ffb7fff73ff7fff,     0x3ffb7fff73ff7fff,
    0x3fff7fff7fff7fff,     0x3fff7fff7fff7fff,     0x3fff7fff7fff7fff,     0x3fff7fff7fff7fff
};

const uint64_t DataSet_1_64u::outputs::HBORS[16] = {
    0x68aa1ac70f0fb7fc,     0x69ee1ef70f4ff7fd,     0x6dfe1fff7f7fffff,     0x7dfe5fff7f7fffff,
    0x7dff7fff7fffffff,     0x7fff7fff7fffffff,     0x7fff7fff7fffffff,     0x7fff7fff7fffffff,
    0x7fff7fff7fffffff,     0x7fff7fff7fffffff,     0x7fff7fff7fffffff,     0x7fff7fff7fffffff,
    0x7fff7fff7fffffff,     0x7fff7fff7fffffff,     0x7fff7fff7fffffff,     0x7fff7fff7fffffff
};

const uint64_t DataSet_1_64u::outputs::MHBORS[16] = {
    0x000000000009b5cc,     0x000000000009b5cc,     0x000000000009b5cc,     0x1daa5553417bfdcf,
    0x1daa5553417bfdcf,     0x3fba557f61ffffdf,     0x3fbb57ff71ffffff,     0x3fbb57ff71ffffff,
    0x3fbb57ff71ffffff,     0x3ffb57ff73ffffff,     0x3ffb7fff73ffffff,     0x3ffb7fff73ffffff,
    0x3fff7fff7fffffff,     0x3fff7fff7fffffff,     0x3fff7fff7fffffff,     0x3fff7fff7fffffff
};

const uint64_t DataSet_1_64u::outputs::HBXOR[16] = {
    0x68aa1ac70f070230,     0x414406f10e437729,     0x2dd41f8a7a7a5a06,     0x307e4ad93b012245,
    0x346360f637913112,     0x065125c9563f6f40,     0x3cd0320667fb18b8,     0x10c240d028444ea0,
    0x0901401c7b4260a7,     0x39e851a1780421e9,     0x1ad23c3d49714349,     0x7d391ce46b600446,
    0x652c6bd107c97339,     0x0f71730773560563,     0x1ea50e912c7a68e6,     0x1ab9243006de1075
};

const uint64_t DataSet_1_64u::outputs::MHBXOR[16] = {
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x1daa5553417b7843,
    0x1daa5553417b7843,     0x2f98106c20d52611,     0x151907a3111151e9,     0x151907a3111151e9,
    0x151907a3111151e9,     0x25f0161e125710a7,     0x06ca7b8223227207,     0x06ca7b8223227207,
    0x1edf0cb74f8b0578,     0x1edf0cb74f8b0578,     0x1edf0cb74f8b0578,     0x1ac32616652f7deb
};

const uint64_t DataSet_1_64u::outputs::HBXORS[16] = {
    0x68aa1ac70f0eb7fc,     0x414406f10e4ac2e5,     0x2dd41f8a7a73efca,     0x307e4ad93b089789,
    0x346360f6379884de,     0x065125c95636da8c,     0x3cd0320667f2ad74,     0x10c240d0284dfb6c,
    0x0901401c7b4bd56b,     0x39e851a1780d9425,     0x1ad23c3d4978f685,     0x7d391ce46b69b18a,
    0x652c6bd107c0c6f5,     0x0f717307735fb0af,     0x1ea50e912c73dd2a,     0x1ab9243006d7a5b9
};

const uint64_t DataSet_1_64u::outputs::MHBXORS[16] = {
    0x000000000009b5cc,     0x000000000009b5cc,     0x000000000009b5cc,     0x1daa55534172cd8f,
    0x1daa55534172cd8f,     0x2f98106c20dc93dd,     0x151907a31118e425,     0x151907a31118e425,
    0x151907a31118e425,     0x25f0161e125ea56b,     0x06ca7b82232bc7cb,     0x06ca7b82232bc7cb,
    0x1edf0cb74f82b0b4,     0x1edf0cb74f82b0b4,     0x1edf0cb74f82b0b4,     0x1ac326166526c827
};

const uint64_t DataSet_1_64u::outputs::LSHV[16] = {
    0x8aa1ac70f0702300,     0x9ee1c36014475190,     0x24065edd0e4b4bc0,     0x3417b78430000000,
    0x45e192026ae00000,     0x3f61ae5e52000000,     0x98e23bfc00000000,     0x64fbf56180000000,
    0x14c18b81c0000000,     0xde81a320a7000000,     0x70c5d58a80000000,     0x36488451c3c00000,
    0x1577356ca9777f00,     0x59d27dd968000000,     0xd47d965f2c6d8500,     0x0e155095523c4980
};

const uint64_t DataSet_1_64u::outputs::MLSHV[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x3417b78430000000,
    0x041d2a2f0c901357,     0x3f61ae5e52000000,     0x98e23bfc00000000,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0xde81a320a7000000,     0x70c5d58a80000000,     0x67eb20d92211470f,
    0x1577356ca9777f00,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x0e155095523c4980
};

const uint64_t DataSet_1_64u::outputs::LSHS[16] = {
    0x3878381180000000,     0xb00a23a8c8000000,     0xdba1c96978000000,     0x9a0bdbc218000000,
    0x7864809ab8000000,     0xfb0d72f290000000,     0x798e23bfc0000000,     0xb27dfab0c0000000,
    0x6298317038000000,     0xe81a320a70000000,     0xe18bab1500000000,     0xc9108a3878000000,
    0xab654bbbf8000000,     0xb3a4fbb2d0000000,     0xb2f9636c28000000,     0x095523c498000000
};

const uint64_t DataSet_1_64u::outputs::MLSHS[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x9a0bdbc218000000,
    0x041d2a2f0c901357,     0xfb0d72f290000000,     0x798e23bfc0000000,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0xe81a320a70000000,     0xe18bab1500000000,     0x67eb20d92211470f,
    0xab654bbbf8000000,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x095523c498000000
};

const uint64_t DataSet_1_64u::outputs::RSHV[16] = {
    0x068aa1ac70f07023,     0x029ee1c360144751,     0x01b24065edd0e4b4,     0x00000001daa55534,
    0x00000020e9517864,     0x0000003232453f61,     0x0000000075022f9e,     0x00000002c1272d64,
    0x00000000670c0331,     0x00000061d2237a06,     0x00000008ce9b670c,     0x0000019fac836488,
    0x00181577356ca977,     0x0000001a9746359d,     0x0011d47d965f2c6d,     0x00083855425548f1
};

const uint64_t DataSet_1_64u::outputs::MRSHV[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x00000001daa55534,
    0x041d2a2f0c901357,     0x0000003232453f61,     0x0000000075022f9e,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x00000061d2237a06,     0x00000008ce9b670c,     0x67eb20d92211470f,
    0x00181577356ca977,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x00083855425548f1
};

const uint64_t DataSet_1_64u::outputs::RSHS[16] = {
    0x0000000d154358e1,     0x000000053dc386c0,     0x0000000d92032f6e,     0x00000003b54aaa68,
    0x0000000083a545e1,     0x000000064648a7ec,     0x000000075022f9e6,     0x00000005824e5ac9,
    0x000000033860198a,     0x000000061d2237a0,     0x00000004674db386,     0x0000000cfd641b24,
    0x0000000302aee6ad,     0x0000000d4ba31ace,     0x000000023a8fb2cb,     0x0000000083855425
};

const uint64_t DataSet_1_64u::outputs::MRSHS[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x00000003b54aaa68,
    0x041d2a2f0c901357,     0x000000064648a7ec,     0x000000075022f9e6,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x000000061d2237a0,     0x00000004674db386,     0x67eb20d92211470f,
    0x0000000302aee6ad,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x0000000083855425
};

const uint64_t DataSet_1_64u::outputs::ROLV[16] = {
    0x8aa1ac70f0702306,     0x9ee1c36014475192,     0x24065edd0e4b4bdb,     0x3417b78431daa555,
    0x45e192026ae083a5,     0x3f61ae5e52323245,     0x98e23bfc1d408be7,     0x64fbf56182c1272d,
    0x14c18b81c670c033,     0xde81a320a7187488,     0x70c5d58a808ce9b6,     0x36488451c3d9fac8,
    0x1577356ca9777f18,     0x59d27dd969a97463,     0xd47d965f2c6d8511,     0x0e155095523c4982
};

const uint64_t DataSet_1_64u::outputs::MROLV[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x3417b78431daa555,
    0x041d2a2f0c901357,     0x3f61ae5e52323245,     0x98e23bfc1d408be7,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0xde81a320a7187488,     0x70c5d58a808ce9b6,     0x67eb20d92211470f,
    0x1577356ca9777f18,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x0e155095523c4982
};

const uint64_t DataSet_1_64u::outputs::ROLS[16] = {
    0x38783811834550d6,     0xb00a23a8c94f70e1,     0xdba1c9697b6480cb,     0x9a0bdbc218ed52aa,
    0x7864809ab820e951,     0xfb0d72f291919229,     0x798e23bfc1d408be,     0xb27dfab0c1609396,
    0x6298317038ce1806,     0xe81a320a7187488d,     0xe18bab150119d36c,     0xc9108a387b3f5906,
    0xab654bbbf8c0abb9,     0xb3a4fbb2d352e8c6,     0xb2f9636c288ea3ec,     0x095523c49820e155
};

const uint64_t DataSet_1_64u::outputs::MROLS[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x9a0bdbc218ed52aa,
    0x041d2a2f0c901357,     0xfb0d72f291919229,     0x798e23bfc1d408be,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0xe81a320a7187488d,     0xe18bab150119d36c,     0x67eb20d92211470f,
    0xab654bbbf8c0abb9,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x095523c49820e155
};

const uint64_t DataSet_1_64u::outputs::RORV[16] = {
    0x068aa1ac70f07023,     0x929ee1c360144751,     0xbdb24065edd0e4b4,     0x17b78431daa55534,
    0x809ab820e9517864,     0xae5e523232453f61,     0x6388eff075022f9e,     0xfbf56182c1272d64,
    0x4c18b81c670c0331,     0x8c829c61d2237a06,     0x5d58a808ce9b670c,     0x451c3d9fac836488,
    0x7f181577356ca977,     0x27dd969a9746359d,     0x8511d47d965f2c6d,     0x26083855425548f1
};

const uint64_t DataSet_1_64u::outputs::MRORV[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x17b78431daa55534,
    0x041d2a2f0c901357,     0xae5e523232453f61,     0x6388eff075022f9e,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x8c829c61d2237a06,     0x5d58a808ce9b670c,     0x67eb20d92211470f,
    0x7f181577356ca977,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x26083855425548f1
};

const uint64_t DataSet_1_64u::outputs::RORS[16] = {
    0xe0e0460d154358e1,     0x288ea3253dc386c0,     0x8725a5ed92032f6e,     0x2f6f0863b54aaa68,
    0x92026ae083a545e1,     0x35cbca464648a7ec,     0x388eff075022f9e6,     0xf7eac305824e5ac9,
    0x60c5c0e33860198a,     0x68c829c61d2237a0,     0x2eac5404674db386,     0x4228e1ecfd641b24,
    0x952eefe302aee6ad,     0x93eecb4d4ba31ace,     0xe58db0a23a8fb2cb,     0x548f126083855425
};

const uint64_t DataSet_1_64u::outputs::MRORS[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x2f6f0863b54aaa68,
    0x041d2a2f0c901357,     0x35cbca464648a7ec,     0x388eff075022f9e6,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x68c829c61d2237a0,     0x2eac5404674db386,     0x67eb20d92211470f,
    0x952eefe302aee6ad,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x548f126083855425
};

const int64_t DataSet_1_64u::outputs::UTOI[16] = {
    0x68aa1ac70f070230,     0x29ee1c3601447519,     0x6c90197b74392d2f,     0x1daa5553417b7843,
    0x041d2a2f0c901357,     0x3232453f61ae5e52,     0x3a8117cf31c477f8,     0x2c1272d64fbf5618,
    0x19c300cc53062e07,     0x30e911bd0346414e,     0x233a6d9c317562a0,     0x67eb20d92211470f,
    0x181577356ca9777f,     0x6a5d18d6749f765a,     0x11d47d965f2c6d85,     0x041c2aa12aa47893
};


const double DataSet_1_64u::outputs::UTOF[16]{
    7.541869968238249e+018,         3.0213834182872571e+018,        7.8227805707641272e+018,        2.1376147892099789e+018,
    2.9643953203891898e+017,        3.6170295892560031e+018,        4.2156769048561234e+018,        3.1757269520361283e+018,
    1.8563283489730186e+018,        3.524367686895551e+018,         2.538461857588798e+018,         7.4881149223851315e+018,
    1.7354243027612116e+018,        7.6643094502131282e+018,        1.2847898785027476e+018,        2.9615854719313318e+017
};

const int64_t DataSet_1_64i::inputs::inputA[16] = {
    0x0e550ac1262a6992, 0x475a3b35688074ba, 0x47a80c3c59d764b5, 0x57fc35ae5c75353e,
    0x5d793f3374314a60, 0x1d4a455947367e2e, 0x638814e41c0536b5, 0x115e106678b9182f,
    0x566b140f478f277a, 0x19674bc524a570c8, 0x540a2f7739020697, 0x662919af46d76ce4,
    0x24232c674b04449b, 0x247b781e57de4b53, 0x10c6192b0d300338, 0x7dc83f5051801748
};

const int64_t DataSet_1_64i::inputs::inputB[16] = {
    0x0ead786002475380, 0x0738595b30c43ffe, 0x151d03ce4a3d695b, 0x76f206cd010959de,
    0x69ef567d35477b7e, 0x5145261b76cf6201, 0x67b0233f63680061, 0x2f507662353b4d73,
    0x13b52c9e15871bca, 0x72396eb6079728da, 0x7c9229567ff12759, 0x29a70a9c7bd148eb,
    0x28a378d62cd50afd, 0x0a48668e0cfa4af1, 0x148727c46ec7657c, 0x744d2c7a5dc8654d
};

const int64_t DataSet_1_64i::inputs::inputC[16] = {
    0x0ef51411017b6771, 0x7da02acf3c3e446e, 0x201473753d876d24, 0x658114c0742d2bdc,
    0x6a9b05e7529b6125, 0x264a587f01253cd2, 0x4ea2599c0a9c0842, 0x0a154e3a371f65e4,
    0x40e74edb093529f9, 0x22e306711119309c, 0x354d587d0a555500, 0x7d17327977926dfa,
    0x19570c7430af73bf, 0x61f95cfc171032ee, 0x5eb10440643930d1, 0x2a347276469b24e0
};

const uint64_t DataSet_1_64i::inputs::inputShiftA[16] = {
    21,27,12,23, 18,12,29,30,
    10,23,10,23, 8, 7, 6, 22
};

const int64_t DataSet_1_64i::inputs::scalarA = -274929821;
const uint64_t DataSet_1_64i::inputs::inputShiftScalarA = 27;

const bool    DataSet_1_64i::inputs::maskA[16] = {
    false,   false,  false,  true,   // 4
    false,  true,   true,   false,  // 8

    false,  true,   true,   false,
    true,   false,  false,  true   // 16
};

/*
const int64_t DataSet_1_64i::outputs::ADDV[16] = {
    0x1d0283212871bd12ll,   0x4e9294909944b4b8ll,   0x5cc5100aa414ce10ll,   0xceee3c7b5d7e8f1cll,
    0xc76895b0a978c5dell,   0x6e8f6b74be05e02fll,   0xcb3838237f6d3716ll,   0x40ae86c8adf465a2ll,
    0x6a2040ad5d164344ll,   0x8ba0ba7b2c3c99a2ll,   0xd09c58cdb8f32df0ll,   0x8fd0244bc2a8b5cfll,
    0x4cc6a53d77d94f98ll,   0x2ec3deac64d89644ll,   0x254d40ef7bf768b4ll,   0xf2156bcaaf487c95ll
};
*/
const int64_t DataSet_1_64i::outputs::ADDV[16] = {
    2090377355489033490,    5661751030325753016,    6684766859789913616,    -3535822156892041444,
    -4077844876593936930,   7966704414990131247,    -3803228160201509098,   4660810860845098402,
    7647183280610362180,    -8385497467977950814,   -3414756776859193872,   -8083921423323056689,
    5532290875693551512,    3369781778210920004,    2687875949931227316,    -1002776823751738219
};

const int64_t DataSet_1_64i::outputs::MADDV[16] = {
    (int64_t)1032743514236676498l,   (int64_t)5141487025169396922l,   (int64_t)5163390426125132981l,   (int64_t)-3535822156892041444l,
    (int64_t)6735484207934556768l,   (int64_t)7966704414990131247l,   (int64_t)-3803228160201509098l,  (int64_t)1251455778753681455l,
    (int64_t)6227092965627471738l,   (int64_t)-8385497467977950814l,  (int64_t)-3414756776859193872l,  (int64_t)7361443306512280804l,
    (int64_t)5532290875693551512l,   (int64_t)2628826879219354451l,   (int64_t)1208681222691095352l,   (int64_t)-1002776823751738219l
};

const int64_t DataSet_1_64i::outputs::ADDS[16] = {
    (int64_t)1032743513961746677,   (int64_t)5141487024894467101,   (int64_t)5163390425850203160,   (int64_t)6340001398147652769,
    (int64_t)6735484207659626947,   (int64_t)2110575624850728337,   (int64_t)7172005376267787800,   (int64_t)1251455778478751634,
    (int64_t)6227092965352541917,   (int64_t)1830515083369928747,   (int64_t)6055704837782171130,   (int64_t)7361443306237350983,
    (int64_t)2603973831427173374,   (int64_t)2628826878944424630,   (int64_t)1208681222416165531,   (int64_t)9063563864005476011
};

const int64_t DataSet_1_64i::outputs::MADDS[16] = {
    (int64_t)1032743514236676498,   (int64_t)5141487025169396922,   (int64_t)5163390426125132981,   (int64_t)6340001398147652769,
    (int64_t)6735484207934556768,   (int64_t)2110575624850728337,   (int64_t)7172005376267787800,   (int64_t)1251455778753681455,
    (int64_t)6227092965627471738,   (int64_t)1830515083369928747,   (int64_t)6055704837782171130,   (int64_t)7361443306512280804,
    (int64_t)2603973831427173374,   (int64_t)2628826879219354451,   (int64_t)1208681222691095352,   (int64_t)9063563864005476011
};

const int64_t DataSet_1_64i::outputs::POSTPREFINC[16] = {
    1032743514236676499,    5141487025169396923,    5163390426125132982,    6340001398422582591,
    6735484207934556769,    2110575625125658159,    7172005376542717622,    1251455778753681456,
    6227092965627471739,    1830515083644858569,    6055704838057100952,    7361443306512280805,
    2603973831702103196,    2628826879219354452,    1208681222691095353,    9063563864280405833
};

const int64_t DataSet_1_64i::outputs::MPOSTPREFINC[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    6340001398422582591,
    6735484207934556768,    2110575625125658159,    7172005376542717622,    1251455778753681455,
    6227092965627471738,    1830515083644858569,    6055704838057100952,    7361443306512280804,
    2603973831702103196,    2628826879219354451,    1208681222691095352,    9063563864280405833
};

const int64_t DataSet_1_64i::outputs::SUBV[16] = {
    -24890327015680494,     4621223020013040828,    3642013992460352346,    -2230919119972344992,
    -897930781246501150,    -3745553164738814931,   -299505160422607276,    -2157899303337735492,
    4807002650644581296,    -6400216438441883666,   -2920577620736155842,   4360063962638066681,
    -324343212289345122,    1887871980227788898,    -270513504549036612,    683160478602998267
};

const int64_t DataSet_1_64i::outputs::MSUBV[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    -2230919119972344992,
    6735484207934556768,    -3745553164738814931,   -299505160422607276,    1251455778753681455,
    6227092965627471738,    -6400216438441883666,   -2920577620736155842,   7361443306512280804,
    -324343212289345122,    2628826879219354451,    1208681222691095352,    683160478602998267
};

const int64_t DataSet_1_64i::outputs::SUBS[16] = {
    1032743514511606319,    5141487025444326743,    5163390426400062802,    6340001398697512411,
    6735484208209486589,    2110575625400587979,    7172005376817647442,    1251455779028611276,
    6227092965902401559,    1830515083919788389,    6055704838332030772,    7361443306787210625,
    2603973831977033016,    2628826879494284272,    1208681222966025173,    9063563864555335653
};

const int64_t DataSet_1_64i::outputs::MSUBS[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    6340001398697512411,
    6735484207934556768,    2110575625400587979,    7172005376817647442,    1251455778753681455,
    6227092965627471738,    1830515083919788389,    6055704838332030772,    7361443306512280804,
    2603973831977033016,    2628826879219354451,    1208681222691095352,    9063563864555335653
};

const int64_t DataSet_1_64i::outputs::SUBFROMV[16] = {
    24890327015680494,      -4621223020013040828,   -3642013992460352346,   2230919119972344992,
    897930781246501150,     3745553164738814931,    299505160422607276,     2157899303337735492,
    -4807002650644581296,   6400216438441883666,    2920577620736155842,    -4360063962638066681,
    324343212289345122,     -1887871980227788898,   270513504549036612,     -683160478602998267
};

const int64_t DataSet_1_64i::outputs::MSUBFROMV[16] = {
    1057633841252356992,    520264005156356094,     1521376433664780635,    2230919119972344992,
    7633414989181057918,    3745553164738814931,    299505160422607276,     3409355082091416947,
    1420090314982890442,    6400216438441883666,    2920577620736155842,    3001379343874214123,
    324343212289345122,     740954898991565553,     1479194727240131964,    -683160478602998267
};

const int64_t DataSet_1_64i::outputs::SUBFROMS[16] = {
    -1032743514511606319,   -5141487025444326743,   -5163390426400062802,   -6340001398697512411,
    -6735484208209486589,   -2110575625400587979,   -7172005376817647442,   -1251455779028611276,
    -6227092965902401559,   -1830515083919788389,   -6055704838332030772,   -7361443306787210625,
    -2603973831977033016,   -2628826879494284272,   -1208681222966025173,   -9063563864555335653
};

const int64_t DataSet_1_64i::outputs::MSUBFROMS[16] = {
    -274929821,     -274929821,     -274929821,     -6340001398697512411,
    -274929821,     -2110575625400587979,   -7172005376817647442,   -274929821,
    -274929821,     -1830515083919788389,   -6055704838332030772,   -274929821,
    -2603973831977033016,   -274929821,     -274929821,     -9063563864555335653
};

const int64_t DataSet_1_64i::outputs::POSTPREFDEC[16] = {
    1032743514236676497,    5141487025169396921,    5163390426125132980,    6340001398422582589,
    6735484207934556767,    2110575625125658157,    7172005376542717620,    1251455778753681454,
    6227092965627471737,    1830515083644858567,    6055704838057100950,    7361443306512280803,
    2603973831702103194,    2628826879219354450,    1208681222691095351,    9063563864280405831
};

const int64_t DataSet_1_64i::outputs::MPOSTPREFDEC[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    6340001398422582589,
    6735484207934556768,    2110575625125658157,    7172005376542717620,    1251455778753681455,
    6227092965627471738,    1830515083644858567,    6055704838057100950,    7361443306512280804,
    2603973831702103194,    2628826879219354451,    1208681222691095352,    9063563864280405831
};

const int64_t DataSet_1_64i::outputs::MULV[16] = {
    -4616768746060701952,   -4214485177120483700,   -3090333137041880745,   4509501062932380100,
    -8355352582018909376,   -1172544834931123666,   8450026409593780885,    -3567623860658307043,
    931753234814796868,     3148743423733353040,    -494063311225336961,    -6315882913254271668,
    7331881002590853935,    9177753979853530915,    -3185155843178977504,   -701321123556071256
};

const int64_t DataSet_1_64i::outputs::MMULV[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    4509501062932380100,
    6735484207934556768,    -1172544834931123666,   8450026409593780885,    1251455778753681455,
    6227092965627471738,    3148743423733353040,    -494063311225336961,    7361443306512280804,
    7331881002590853935,    2628826879219354451,    1208681222691095352,    -701321123556071256
};

const int64_t DataSet_1_64i::outputs::MULS[16] = {
    126631817890599286,     -8363961338319799826,   -3758638746237975297,   -4821764487429846790,
    -6445100941578575072,   -5982707076668437046,   -2477362378464986369,   -3991855307762777299,
    4616652614679091758,    2090947455458678104,    -5574741278760972955,   -7265162993392887764,
    -1611779943291525903,   -5491403644211100135,   -5211388695628167512,   -7702811104884623144
};

const int64_t DataSet_1_64i::outputs::MMULS[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    -4821764487429846790,
    6735484207934556768,    -5982707076668437046,   -2477362378464986369,   1251455778753681455,
    6227092965627471738,    2090947455458678104,    -5574741278760972955,   7361443306512280804,
    -1611779943291525903,   2628826879219354451,    1208681222691095352,    -7702811104884623144
};

const int64_t DataSet_1_64i::outputs::DIVV[16] = {
    0,      9,      3,      0,
    0,      0,      0,      0,
    4,      0,      0,      2,
    0,      3,      0,      1
};

const int64_t DataSet_1_64i::outputs::MDIVV[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    0,
    6735484207934556768,    0,      0,      1251455778753681455,
    6227092965627471738,    0,      0,      7361443306512280804,
    0,      2628826879219354451,    1208681222691095352,    1
};

const int64_t DataSet_1_64i::outputs::DIVS[16] = {
    -3756389577ll,    -18701088905,   -18780757967,   -23060435478,
    -24498921882,   -7676779541,    -26086676776,   -4551909917,
    -22649754555,   -6658117613,    -22026365914,   -26775717816,
    -9471412821,    -9561810609,    -4396326372,    -32966827066
};

const int64_t DataSet_1_64i::outputs::MDIVS[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    -23060435478,
    6735484207934556768,    -7676779541,    -26086676776,   1251455778753681455,
    6227092965627471738,    -6658117613,    -22026365914,   7361443306512280804,
    -9471412821,    2628826879219354451,    1208681222691095352,    -32966827066
};

const int64_t DataSet_1_64i::outputs::RCP[16] = {
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0
};

const int64_t DataSet_1_64i::outputs::MRCP[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    0,
    6735484207934556768,    0,      0,      1251455778753681455,
    6227092965627471738,    0,      0,      7361443306512280804,
    0,      2628826879219354451,    1208681222691095352,    0
};

const int64_t DataSet_1_64i::outputs::RCPS[16] = {
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0
};

const int64_t DataSet_1_64i::outputs::MRCPS[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    0,
    6735484207934556768,    0,      0,      1251455778753681455,
    6227092965627471738,    0,      0,      7361443306512280804,
    0,      2628826879219354451,    1208681222691095352,    0
};

const bool DataSet_1_64i::outputs::CMPEQV[16] = {
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false
};

const bool DataSet_1_64i::outputs::CMPEQS[16] = {
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false
};

const bool DataSet_1_64i::outputs::CMPNEV[16] = {
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true
};

const bool DataSet_1_64i::outputs::CMPNES[16] = {
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true
};

const bool DataSet_1_64i::outputs::CMPGTV[16] = {
    false,  true,   true,   false,
    false,  false,  false,  false,
    true,   false,  false,  true,
    false,  true,   false,  true
};

const bool DataSet_1_64i::outputs::CMPGTS[16] = {
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true
};

const bool DataSet_1_64i::outputs::CMPLTV[16] = {
    true,   false,  false,  true,
    true,   true,   true,   true,
    false,  true,   true,   false,
    true,   false,  true,   false
};

const bool DataSet_1_64i::outputs::CMPLTS[16] = {
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false
};

const bool DataSet_1_64i::outputs::CMPGEV[16] = {
    false,  true,   true,   false,
    false,  false,  false,  false,
    true,   false,  false,  true,
    false,  true,   false,  true
};

const bool DataSet_1_64i::outputs::CMPGES[16] = {
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true
};

const bool DataSet_1_64i::outputs::CMPLEV[16] = {
    true,   false,  false,  true,
    true,   true,   true,   true,
    false,  true,   true,   false,
    true,   false,  true,   false
};

const bool DataSet_1_64i::outputs::CMPLES[16] = {
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false
};

const bool  DataSet_1_64i::outputs::CMPEV = false;
const bool  DataSet_1_64i::outputs::CMPES = false;


const int64_t DataSet_1_64i::outputs::HADD[16] = {
    1032743514236676498,    6174230539406073420,    -7109123108178345215,   -769121709755762625,
    5966362498178794143,    8076938123304452301,    -3197800573862381694,   -1946344795108700239,
    4280748170518771499,    6111263254163630067,    -6279775981488820598,   1081667325023460206,
    3685641156725563401,    6314468035944917852,    7523149258636013204,    -1860030950793132580
};

const int64_t DataSet_1_64i::outputs::MHADD[16] = {
    0,      0,      0,      6340001398422582590,
    6340001398422582590,    8450577023548240748,    -2824161673618593247,   -2824161673618593247,
    -2824161673618593247,   -993646589973734679,    5062058248083366272,    5062058248083366272,
    7666032079785469467,    7666032079785469467,    7666032079785469467,    -1717148129643676317
};

const int64_t DataSet_1_64i::outputs::HMUL[16] = {
    1032743514236676498,    1382084960963714068,    5949478008837204516,    -6637472432988018504,
    -4151973528349149952,   1080835075233482240,    9128782934570384896,    -5132568264794938880,
    3051122567841887232,    4557374540206612480,    -7874891803985289216,   4791387871600541696,
    -3466208858008158208,   3877749311395430400,    -8746227168344539136,   -3598538683011039232
};

const int64_t DataSet_1_64i::outputs::MHMUL[16] = {
    1,      1,      1,      6340001398422582590,
    6340001398422582590,    -4166197022219037404,   2417672136483179124,    2417672136483179124,
    2417672136483179124,    3876760725745756832,    -1080114154415053728,   -1080114154415053728,
    4892438263075669536,    4892438263075669536,    4892438263075669536,    8925886136776825088
};

const int64_t DataSet_1_64i::outputs::FMULADDV[16] = {
    -3538978996963735951,   4837797143451523834,    -778733680913320325,    -6623092924135381600,
    -673612501526897563,    1586570189290034944,    -4330527856202497321,   -2841050933828229631,
    5608546675521629757,    5662603523424156396,    3346760040059084927,    2697845763229582150,
    9157822869685423854,    -2209213965187605999,   3638083741869922289,    2339860377142136200
};

const int64_t DataSet_1_64i::outputs::MFMULADDV[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    -6623092924135381600,
    6735484207934556768,    1586570189290034944,    -4330527856202497321,   1251455778753681455,
    6227092965627471738,    5662603523424156396,    3346760040059084927,    7361443306512280804,
    9157822869685423854,    2628826879219354451,    1208681222691095352,    2339860377142136200
};

const int64_t DataSet_1_64i::outputs::FMULSUBV[16] = {
    -5694558495157667953,   5179976576017060382,    -5401932593170441165,   -2804649023709409816,
    2409651411198630427,    -3931659859152282276,   2783836601680507475,    -4294196787488384455,
    -3745040205892036021,   634883324042549684,     -4334886662509758849,   3117132483971426130,
    5505939135496284016,    2117977851185116213,    8438348645481674319,    -3742502624254278712
};

const int64_t DataSet_1_64i::outputs::MFMULSUBV[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    -2804649023709409816,
    6735484207934556768,    -3931659859152282276,   2783836601680507475,    1251455778753681455,
    6227092965627471738,    634883324042549684,     -4334886662509758849,   7361443306512280804,
    5505939135496284016,    2628826879219354451,    1208681222691095352,    -3742502624254278712
};

const int64_t DataSet_1_64i::outputs::FADDMULV[16] = {
    8323232093143085810,    -8027727225537788144,   7031142381946260032,    -8347703588541648880,
    -6028420416683919594,   6702304401822313102,    -1334271384695020628,   8136364598461820488,
    -231093509564575452,    5611397153643429560,    -2973460824419422208,   6431845535190658854,
    7118153671439526504,    3081814482397494072,    5327519486934399732,    6391887297925150304
};

const int64_t DataSet_1_64i::outputs::MFADDMULV[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    -8347703588541648880,
    6735484207934556768,    6702304401822313102,    -1334271384695020628,   1251455778753681455,
    6227092965627471738,    5611397153643429560,    -2973460824419422208,   7361443306512280804,
    7118153671439526504,    2628826879219354451,    1208681222691095352,    6391887297925150304
};

const int64_t DataSet_1_64i::outputs::FSUBMULV[16] = {
    3869914469769673714,    1858889912353855688,    3041976320794995368,    -2372801743732234624,
    -1768573316647057238,   -495934842673780502,    9054598483308880296,    -2695892771976725648,
    -5300031147426542032,   -7073844524674550520,   5746956011239216640,    -4192288620346726870,
    -2406845784812161310,   -8174801302548938980,   -1874952771034675588,   -328995798484514912
};

const int64_t DataSet_1_64i::outputs::MFSUBMULV[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    -2372801743732234624,
    6735484207934556768,    -495934842673780502,    9054598483308880296,    1251455778753681455,
    6227092965627471738,    -7073844524674550520,   5746956011239216640,    7361443306512280804,
    -2406845784812161310,   2628826879219354451,    1208681222691095352,    -328995798484514912
};

const int64_t DataSet_1_64i::outputs::MAXV[16] = {
    1057633841252356992,    5141487025169396922,    5163390426125132981,    8570920518394927582,
    7633414989181057918,    5856128789864473089,    7471510536965324897,    3409355082091416947,
    6227092965627471738,    8230731522086742234,    8976282458793256793,    7361443306512280804,
    2928317043991448317,    2628826879219354451,    1479194727240131964,    9063563864280405832
};

const int64_t DataSet_1_64i::outputs::MMAXV[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    8570920518394927582,
    6735484207934556768,    5856128789864473089,    7471510536965324897,    1251455778753681455,
    6227092965627471738,    8230731522086742234,    8976282458793256793,    7361443306512280804,
    2928317043991448317,    2628826879219354451,    1208681222691095352,    9063563864280405832
};

const int64_t DataSet_1_64i::outputs::MAXS[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    6340001398422582590,
    6735484207934556768,    2110575625125658158,    7172005376542717621,    1251455778753681455,
    6227092965627471738,    1830515083644858568,    6055704838057100951,    7361443306512280804,
    2603973831702103195,    2628826879219354451,    1208681222691095352,    9063563864280405832
};

const int64_t DataSet_1_64i::outputs::MMAXS[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    6340001398422582590,
    6735484207934556768,    2110575625125658158,    7172005376542717621,    1251455778753681455,
    6227092965627471738,    1830515083644858568,    6055704838057100951,    7361443306512280804,
    2603973831702103195,    2628826879219354451,    1208681222691095352,    9063563864280405832
};

const int64_t DataSet_1_64i::outputs::MINV[16] = {
    1032743514236676498,    520264005156356094,     1521376433664780635,    6340001398422582590,
    6735484207934556768,    2110575625125658158,    7172005376542717621,    1251455778753681455,
    1420090314982890442,    1830515083644858568,    6055704838057100951,    3001379343874214123,
    2603973831702103195,    740954898991565553,     1208681222691095352,    8380403385677407565
};

const int64_t DataSet_1_64i::outputs::MMINV[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    6340001398422582590,
    6735484207934556768,    2110575625125658158,    7172005376542717621,    1251455778753681455,
    6227092965627471738,    1830515083644858568,    6055704838057100951,    7361443306512280804,
    2603973831702103195,    2628826879219354451,    1208681222691095352,    8380403385677407565
};

const int64_t DataSet_1_64i::outputs::MINS[16] = {
    -274929821,     -274929821,     -274929821,     -274929821,
    -274929821,     -274929821,     -274929821,     -274929821,
    -274929821,     -274929821,     -274929821,     -274929821,
    -274929821,     -274929821,     -274929821,     -274929821
};

const int64_t DataSet_1_64i::outputs::MMINS[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    -274929821,
    6735484207934556768,    -274929821,     -274929821,     1251455778753681455,
    6227092965627471738,    -274929821,     -274929821,     7361443306512280804,
    -274929821,     2628826879219354451,    1208681222691095352,    -274929821
};

const int64_t DataSet_1_64i::outputs::HMAX[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    6340001398422582590,
    6735484207934556768,    6735484207934556768,    7172005376542717621,    7172005376542717621,
    7172005376542717621,    7172005376542717621,    7172005376542717621,    7361443306512280804,
    7361443306512280804,    7361443306512280804,    7361443306512280804,    9063563864280405832
};

const int64_t DataSet_1_64i::outputs::MHMAX[16] = {
    -9223372036854775807,   -9223372036854775807,   -9223372036854775807,   6340001398422582590,
    6340001398422582590,    6340001398422582590,    7172005376542717621,    7172005376542717621,
    7172005376542717621,    7172005376542717621,    7172005376542717621,    7172005376542717621,
    7172005376542717621,    7172005376542717621,    7172005376542717621,    9063563864280405832
};

const int64_t DataSet_1_64i::outputs::HMIN[16] = {
    1032743514236676498,    1032743514236676498,    1032743514236676498,    1032743514236676498,
    1032743514236676498,    1032743514236676498,    1032743514236676498,    1032743514236676498,
    1032743514236676498,    1032743514236676498,    1032743514236676498,    1032743514236676498,
    1032743514236676498,    1032743514236676498,    1032743514236676498,    1032743514236676498
};

const int64_t DataSet_1_64i::outputs::MHMIN[16] = {
    9223372036854775807,    9223372036854775807,    9223372036854775807,    6340001398422582590,
    6340001398422582590,    2110575625125658158,    2110575625125658158,    2110575625125658158,
    2110575625125658158,    1830515083644858568,    1830515083644858568,    1830515083644858568,
    1830515083644858568,    1830515083644858568,    1830515083644858568,    1830515083644858568
};

const int64_t DataSet_1_64i::outputs::BANDV[16] = {
    1010222762419175808,    511186119056962746,     362539822752292881,     6264512081014362398,
    5289783637905066592,    1242998003749773824,    7169730761392652321,    94593606076663843,
    1306329825182942026,    1162292109710270664,    6053446299439597073,    2315141081954601184,
    2315738914645868697,    20371811650390609,      1190640251199029560,    8378995830199158088
};

const int64_t DataSet_1_64i::outputs::MBANDV[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    6264512081014362398,
    6735484207934556768,    1242998003749773824,    7169730761392652321,    1251455778753681455,
    6227092965627471738,    1162292109710270664,    6053446299439597073,    7361443306512280804,
    2315738914645868697,    2628826879219354451,    1208681222691095352,    8378995830199158088
};

const int64_t DataSet_1_64i::outputs::BANDS[16] = {
    1032743514234446082,    5141487025169392674,    5163390425852306465,    6340001398147786018,
    6735484207663956576,    2110575625123423778,    7172005376274212385,    1251455778483077155,
    6227092965627275106,    1830515083642691648,    6055704837788534275,    7361443306507887712,
    2603973831702103043,    2628826878946591555,    1208681222688998176,    9063563864011966272
};

const int64_t DataSet_1_64i::outputs::MBANDS[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    6340001398147786018,
    6735484207934556768,    2110575625123423778,    7172005376274212385,    1251455778753681455,
    6227092965627471738,    1830515083642691648,    6055704837788534275,    7361443306512280804,
    2603973831702103043,    2628826879219354451,    1208681222691095352,    9063563864011966272
};

const int64_t DataSet_1_64i::outputs::BORV[16] = {
    1080154593069857682,    5150564911268790270,    6322227037037620735,    8646409835803147774,
    9079115559210548094,    6723706411240357423,    7473785152115390197,    4566217254768434559,
    6340853455427420154,    8898954496021330138,    8978540997410760671,    8047681568431893743,
    3216551961047682815,    3349409966560529395,    1497235698732197756,    9064971419758655309
};

const int64_t DataSet_1_64i::outputs::MBORV[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    8646409835803147774,
    6735484207934556768,    6723706411240357423,    7473785152115390197,    1251455778753681455,
    6227092965627471738,    8898954496021330138,    8978540997410760671,    7361443306512280804,
    3216551961047682815,    2628826879219354451,    1208681222691095352,    9064971419758655309
};

const int64_t DataSet_1_64i::outputs::BORS[16] = {
    -272699405,     -274925573,     -2103305,       -133249,
    -4329629,       -272695441,     -6424585,       -4325521,
    -274733189,     -272762901,     -6363145,       -270536729,
    -274929669,     -2166925,       -272832645,     -6490261
};

const int64_t DataSet_1_64i::outputs::MBORS[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    -133249,
    6735484207934556768,    -272695441,     -6424585,       1251455778753681455,
    6227092965627471738,    -272762901,     -6363145,       7361443306512280804,
    -274929669,     2628826879219354451,    1208681222691095352,    -6490261
};

const int64_t DataSet_1_64i::outputs::BXORV[16] = {
    69931830650681874,      4639378792211827524,    5959687214285327854,    2381897754788785376,
    3789331921305481502,    5480708407490583599,    304054390722737876,     4471623648691770716,
    5034523630244478128,    7736662386311059474,    2925094697971163598,    5732540486477292559,
    900813046401814118,     3329038154910138786,    306595447533168196,     685975589559497221
};

const int64_t DataSet_1_64i::outputs::MBXORV[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    2381897754788785376,
    6735484207934556768,    5480708407490583599,    304054390722737876,     1251455778753681455,
    6227092965627471738,    7736662386311059474,    2925094697971163598,    7361443306512280804,
    900813046401814118,     2628826879219354451,    1208681222691095352,    685975589559497221
};

const int64_t DataSet_1_64i::outputs::BXORS[16] = {
    -1032743514507145487,   -5141487025444318247,   -5163390425854409770,   -6340001398147919267,
    -6735484207668286205,   -2110575625396119219,   -7172005376280636970,   -1251455778487402676,
    -6227092965902008295,   -1830515083915454549,   -6055704837794897420,   -7361443306778424441,
    -2603973831977032712,   -2628826878948758480,   -1208681222961830821,   -9063563864018456533
};

const int64_t DataSet_1_64i::outputs::MBXORS[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    -6340001398147919267,
    6735484207934556768,    -2110575625396119219,   -7172005376280636970,   1251455778753681455,
    6227092965627471738,    -1830515083915454549,   -6055704837794897420,   7361443306512280804,
    -2603973831977032712,   2628826879219354451,    1208681222691095352,    -9063563864018456533
};

const int64_t DataSet_1_64i::outputs::BNOT[16] = {
    -1032743514236676499,   -5141487025169396923,   -5163390426125132982,   -6340001398422582591,
    -6735484207934556769,   -2110575625125658159,   -7172005376542717622,   -1251455778753681456,
    -6227092965627471739,   -1830515083644858569,   -6055704838057100952,   -7361443306512280805,
    -2603973831702103196,   -2628826879219354452,   -1208681222691095353,   -9063563864280405833
};

const int64_t DataSet_1_64i::outputs::MBNOT[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    -6340001398422582591,
    6735484207934556768,    -2110575625125658159,   -7172005376542717622,   1251455778753681455,
    6227092965627471738,    -1830515083644858569,   -6055704838057100952,   7361443306512280804,
    -2603973831702103196,   2628826879219354451,    1208681222691095352,    -9063563864280405833
};

const int64_t DataSet_1_64i::outputs::HBAND[16] = {
    1032743514236676498,    454874562312560786,     432354360320614544,     432345564227575824,
    288230376151711744,     288230376151711744,     0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0
};

const int64_t DataSet_1_64i::outputs::MHBAND[16] = {
    -1,     -1,     -1,     6340001398422582590,
    6340001398422582590,    1533481206181803054,    74313791965508644,      74313791965508644,
    74313791965508644,      72057594105311232,      0,      0,
    0,      0,      0,      0
};

const int64_t DataSet_1_64i::outputs::HBANDS[16] = {
    1032743514234446082,    454874562312560642,     432354360320614400,     432345564227575808,
    288230376151711744,     288230376151711744,     0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0
};

const int64_t DataSet_1_64i::outputs::MHBANDS[16] = {
    -274929821,     -274929821,     -274929821,     6340001398147786018,
    6340001398147786018,    1533481206179701794,    74313791965504544,      74313791965504544,
    74313791965504544,      72057594105307136,      0,      0,
    0,      0,      0,      0
};

const int64_t DataSet_1_64i::outputs::HBOR[16] = {
    1032743514236676498,    5719355977093512634,    5764396406064250303,    6917317919261031871,
    6917317919261032447,    6917388288005210111,    9223231297218904063,    9223231297218904063,
    9223231297218904063,    9223231297218904063,    9223231297218904063,    9223231297218904063,
    9223231297218904063,    9223231297218904063,    9223231297218904063,    9223231297218904063
};

const int64_t DataSet_1_64i::outputs::MHBOR[16] = {
    0,      0,      0,      6340001398422582590,
    6340001398422582590,    6917095817366437694,    9222938826580131775,    9222938826580131775,
    9222938826580131775,    9223231297218379775,    9223231297218379775,    9223231297218379775,
    9223231297218379775,    9223231297218379775,    9223231297218379775,    9223231297218379775
};

const int64_t DataSet_1_64i::outputs::HBORS[16] = {
    -272699405,     -272695301,     -1,     -1,
    -1,     -1,     -1,     -1,
    -1,     -1,     -1,     -1,
    -1,     -1,     -1,     -1
};

const int64_t DataSet_1_64i::outputs::MHBORS[16] = {
    -274929821,     -274929821,     -274929821,     -133249,
    -133249,        -129,   -1,     -1,
    -1,     -1,     -1,     -1,
    -1,     -1,     -1,     -1
};

const int64_t DataSet_1_64i::outputs::HBXOR[16] = {
    1032743514236676498,    5264481414780951848,    1055880567238523293,    6438749327694777507,
    297861364632323779,     1830838646405560557,    8854190015707237976,    7763773160372262519,
    4455575714073964813,    2644221305395151301,    8122248745260353362,    1626115695337827254,
    3653039402143723309,    1641926986552468606,    436658733891086150,     8919218113479077902
};

const int64_t DataSet_1_64i::outputs::MHBXOR[16] = {
    0,      0,      0,      6340001398422582590,
    6340001398422582590,    5383614611184634640,    2971922837000322469,    2971922837000322469,
    2971922837000322469,    3483868383519378797,    7229122518800468986,    7229122518800468986,
    4643260446108110689,    4643260446108110689,    4643260446108110689,    4447326167001028649
};

const int64_t DataSet_1_64i::outputs::HBXORS[16] = {
    -1032743514507145487,   -5264481415051412917,   -1055880566963855618,   -6438749327969702976,
    -297861364366057056,    -1830838646143213682,   -8854190015981901509,   -7763773160105725676,
    -4455575713807821202,   -2644221305133078874,   -8122248745534889935,   -1626115695604364075,
    -3653039402410260402,   -1641926986290265315,   -436658733624688603,    -8919218113749543059
};

const int64_t DataSet_1_64i::outputs::MHBXORS[16] = {
    -274929821,     -274929821,     -274929821,     -6340001398147919267,
    -6340001398147919267,   -5383614610913907597,   -2971922837266588986,   -2971922837266588986,
    -2971922837266588986,   -3483868383781328370,   -7229122518525809511,   -7229122518525809511,
    -4643260445833451518,   -4643260445833451518,   -4643260445833451518,   -4447326167263232182
};

const int64_t DataSet_1_64i::outputs::LSHV[16] = {
    6351418309845450752,    -6105751184113401856,   -9168267136311013376,   -2941349070859403264,
    -230298460702638080,    -6605210055083892736,   -8970987016984461312,   -7048619350508437504,
    -6030252700973799424,   -2120541522632900608,   2935745404062030848,    -2908362503010385920,
    2534514262194559744,    4448447213305440640,    3568621957391896064,    -3164799060020297728
};

const int64_t DataSet_1_64i::outputs::MLSHV[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    -2941349070859403264,
    6735484207934556768,    -6605210055083892736,   -8970987016984461312,   1251455778753681455,
    6227092965627471738,    -2120541522632900608,   2935745404062030848,    7361443306512280804,
    2534514262194559744,    2628826879219354451,    1208681222691095352,    -3164799060020297728
};

const int64_t DataSet_1_64i::outputs::LSHS[16] = {
    662402208498712576,     -6105751184113401856,   -2103538205528686592,   8278647087378202624,
    -7232347437493387264,   -3874868154828062720,   2368939264181272576,    3730608599613833216,
    8808048269029539840,    2964823785292693504,    -5059776362739400704,   8806432172962480128,
    4204147793788403712,    -955059600044851200,    6370763995976040448,    -9039849552101769216
};

const int64_t DataSet_1_64i::outputs::MLSHS[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    8278647087378202624,
    6735484207934556768,    -3874868154828062720,   2368939264181272576,    1251455778753681455,
    6227092965627471738,    2964823785292693504,    -5059776362739400704,   7361443306512280804,
    4204147793788403712,    2628826879219354451,    1208681222691095352,    -9039849552101769216
};

const int64_t DataSet_1_64i::outputs::RSHV[16] = {
    492450482481,   38307063469,    1260593365753206,       755787062456,
    25693833190668, 515277252227943,        13358901024,    1165509017,
    6081145474245577,       218214402633,   5913774255915137,       877552426637,
    10171772780086340,      20537709993901206,      18885644104548364,      2160922018118
};

const int64_t DataSet_1_64i::outputs::MRSHV[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    755787062456,
    6735484207934556768,    515277252227943,        13358901024,    1251455778753681455,
    6227092965627471738,    218214402633,   5913774255915137,       7361443306512280804,
    10171772780086340,      2628826879219354451,    1208681222691095352,    2160922018118
};

const int64_t DataSet_1_64i::outputs::RSHS[16] = {
    7694538788,     38307063469,    38470256523,    47236691403,
    50183267950,    15725013800,    53435604099,    9324072143,
    46395458024,    13638400164,    45118516967,    54847026664,
    19401116905,    19586286538,    9005376865,     67528813066
};

const int64_t DataSet_1_64i::outputs::MRSHS[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    47236691403,
    6735484207934556768,    15725013800,    53435604099,    1251455778753681455,
    6227092965627471738,    13638400164,    45118516967,    7361443306512280804,
    19401116905,    2628826879219354451,    1208681222691095352,    67528813066
};

const int64_t DataSet_1_64i::outputs::ROLV[16] = {
    6351418309845568161,    -6105751184075992615,   -9168267136311012230,   -2941349070856520166,
    -230298460702542364,    -6605210055083892268,   -8970987016775728484,   -7048619350435593191,
    -6030252700973799079,   -2120541522632068187,   2935745404062031184,    -2908362503007038324,
    2534514262194559780,    4448447213305440658,    3568621957391896068,    -3164799060018236913
};

const int64_t DataSet_1_64i::outputs::MROLV[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    -2941349070856520166,
    6735484207934556768,    -6605210055083892268,   -8970987016775728484,   1251455778753681455,
    6227092965627471738,    -2120541522632068187,   2935745404062031184,    7361443306512280804,
    2534514262194559780,    2628826879219354451,    1208681222691095352,    -3164799060018236913
};

const int64_t DataSet_1_64i::outputs::ROLS[16] = {
    662402208506226774,     -6105751184075992615,   -2103538205491117983,   8278647087424332205,
    -7232347437444380167,   -3874868154812706262,   2368939264233455783,    3730608599622938755,
    8808048269074847904,    2964823785306012254,    -5059776362695339653,   8806432173016041677,
    4204147793807350115,    -955059600025723968,    6370763995984834761,    -9039849552035823110
};

const int64_t DataSet_1_64i::outputs::MROLS[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    8278647087424332205,
    6735484207934556768,    -3874868154812706262,   2368939264233455783,    1251455778753681455,
    6227092965627471738,    2964823785306012254,    -5059776362695339653,   7361443306512280804,
    4204147793807350115,    2628826879219354451,    1208681222691095352,    -9039849552035823110
};

const int64_t DataSet_1_64i::outputs::RORV[16] = {
    6002331225502910769,    1157028493721560749,    5428098144347200886,    -1555293626074178376,
    5951532601403301132,    -2098162149102423193,   -2294102788650522848,   -2097445164712181351,
    -2407848254796340279,   5395752376455367241,    -6497284087667081087,   -5847421861285896563,
    -7267645225050635196,   -6464645753419613034,   -2286957365109145588,   26214518128197958
};

const int64_t DataSet_1_64i::outputs::MRORV[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    -1555293626074178376,
    6735484207934556768,    -2098162149102423193,   -2294102788650522848,   1251455778753681455,
    6227092965627471738,    5395752376455367241,    -6497284087667081087,   7361443306512280804,
    -7267645225050635196,   2628826879219354451,    1208681222691095352,    26214518128197958
};

const int64_t DataSet_1_64i::outputs::RORS[16] = {
    -4229669216877193180,   1157028493721560749,    4245934201112725387,    -8167656383877564981,
    -8779402385515092370,   -1815014680657614040,   -9176411154602091389,   1667182756012100815,
    -1016424511326486040,   -7733216008719468380,   2324089312356200167,    -2671306875544062488,
    6955971684400991465,    -303594519735106614,    -6485070204710476447,   3459583717512047114
};

const int64_t DataSet_1_64i::outputs::MRORS[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    -8167656383877564981,
    6735484207934556768,    -1815014680657614040,   -9176411154602091389,   1251455778753681455,
    6227092965627471738,    -7733216008719468380,   2324089312356200167,    7361443306512280804,
    6955971684400991465,    2628826879219354451,    1208681222691095352,    3459583717512047114
};

const int64_t DataSet_1_64i::outputs::NEG[16] = {
    -1032743514236676498,   -5141487025169396922,   -5163390426125132981,   -6340001398422582590,
    -6735484207934556768,   -2110575625125658158,   -7172005376542717621,   -1251455778753681455,
    -6227092965627471738,   -1830515083644858568,   -6055704838057100951,   -7361443306512280804,
    -2603973831702103195,   -2628826879219354451,   -1208681222691095352,   -9063563864280405832
};

const int64_t DataSet_1_64i::outputs::MNEG[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    -6340001398422582590,
    6735484207934556768,    -2110575625125658158,   -7172005376542717621,   1251455778753681455,
    6227092965627471738,    -1830515083644858568,   -6055704838057100951,   7361443306512280804,
    -2603973831702103195,   2628826879219354451,    1208681222691095352,    -9063563864280405832
};

const int64_t DataSet_1_64i::outputs::ABS[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    6340001398422582590,
    6735484207934556768,    2110575625125658158,    7172005376542717621,    1251455778753681455,
    6227092965627471738,    1830515083644858568,    6055704838057100951,    7361443306512280804,
    2603973831702103195,    2628826879219354451,    1208681222691095352,    9063563864280405832
};

const int64_t DataSet_1_64i::outputs::MABS[16] = {
    1032743514236676498,    5141487025169396922,    5163390426125132981,    6340001398422582590,
    6735484207934556768,    2110575625125658158,    7172005376542717621,    1251455778753681455,
    6227092965627471738,    1830515083644858568,    6055704838057100951,    7361443306512280804,
    2603973831702103195,    2628826879219354451,    1208681222691095352,    9063563864280405832
};

const uint64_t DataSet_1_64i::outputs::ITOU[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x57fc35ae5c75353e,
    0x5d793f3374314a60,     0x1d4a455947367e2e,     0x638814e41c0536b5,     0x115e106678b9182f,
    0x566b140f478f277a,     0x19674bc524a570c8,     0x540a2f7739020697,     0x662919af46d76ce4,
    0x24232c674b04449b,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x7dc83f5051801748
};

const double DataSet_1_64i::outputs::ITOF[16] = {
    1.0327435142366765e+018,        5.1414870251693967e+018,        5.1633904261251328e+018,        6.3400013984225823e+018,
    6.7354842079345572e+018,        2.1105756251256581e+018,        7.172005376542718e+018,         1.2514557787536814e+018,
    6.2270929656274719e+018,        1.8305150836448586e+018,        6.0557048380571013e+018,        7.3614433065122806e+018,
    2.603973831702103e+018,         2.6288268792193546e+018,        1.2086812226910953e+018,        9.063563864280406e+018
};


const double DataSet_1_64f::inputs::inputA[16] = {
    -8.5218875249821816e-045, 1.7122000368240061e-213, 9.0247961030823995e-279, 1.7713552968799211e-262,
    -7.8910193199857527e+067, -4.3185304341926401e-234, -9.7934052223250672e-273, 1.658047929286562e+064,
    -1.7555649293363597e+130, 1.5403807059396589e-018, 2.7298467895665847e+103, -1.193696428548151e-301,
    -3.2327100168366796e-103, -7.3005992524189971e-037, 8.3871526846650359e+155, 1.5231189023555448e-174
};

const double DataSet_1_64f::inputs::inputB[16] = {
    3.7098891307134739e+201, 8.6009068005193272e-154, 2.925076007564774e+234, 3.3300934358761032e+098,
    1.8158756895993047e-274, -6.8334688081596772e+035, 5.4820144627028369e-114, 2.889381283878379e+097,
    -2.9945465687839791e-304, 3.8437036643523383e-123, -6.2375587223620979e+087, 6.45355304510857e-229,
    -5.8359434565227789e-068, -9.9212431741223588e+294, 1.2256023939360617e-074, 2.1537545861459669e+263
};

const double DataSet_1_64f::inputs::inputC[16] = {
    6.9781414464889818e+238, 4.7465556992920095e+227, 2.2036260169647079e+019, 4.7417426841989699e+051,
    -1.6567842488585635e-180, -4.0372248649429385e-155, -9.4629392185833011e-037, -7.071761035038374e+155,
    1.4606374469850822e+071, -5.0981779553378016e-234, 1.0205604765011866e+241, -2.5413733594285079e-239,
    1.8639765839842973e-272, 0.00011949870511260886, 5.0940362739643062e+197, -3.0317131216131958e-291
};

const uint64_t DataSet_1_64f::inputs::inputUintA[16] = {
    2792756694, 1441485571, 3332554660, 2854188413,
    3917849407, 3356278803, 363973866,  315720241,
    3083762480, 3595285540, 3751632854, 2121976320,
    2438295964, 1083364314, 36816391,   3766746593
};

const int64_t DataSet_1_64f::inputs::inputIntA[16] = {
    447678805,      -293813370,     1272230454,     1756815974,
    2107971637,     -720187790,     264742751,      1672053848,
    1481089269,     -1135701836,    -331170375,     1013950633,
    9647458,        102792793,      -1660519627,    1275239050
};

const double DataSet_1_64f::inputs::scalarA = -1255.89283461;

const bool DataSet_1_64f::inputs::maskA[16] = {
    true,   false,  false,  true,   // 4
    false,  true,   true,   false,  // 8

    false,  true,   true,   false,
    true,   false,  false,  true    // 16
};

const double DataSet_1_64f::outputs::ADDV[16] = {
    3.70988913e+201,        8.6009068e-154, 2.92507601e+234,        3.33009344e+098,
    -7.89101932e+067,       -6.83346881e+035,       5.48201446e-114,        2.88938128e+097,
    -1.75556493e+130,       1.54038071e-018,        2.72984679e+103,        6.45355305e-229,
    -5.83594346e-068,       -9.92124317e+294,       8.38715268e+155,        2.15375459e+263
};

const double DataSet_1_64f::outputs::MADDV[16] = {
    3.70988913e+201,        1.71220004e-213,        9.0247961e-279, 3.33009344e+098,
    -7.89101932e+067,       -6.83346881e+035,       5.48201446e-114,        1.65804793e+064,
    -1.75556493e+130,       1.54038071e-018,        2.72984679e+103,        -1.19369643e-301,
    -5.83594346e-068,       -7.30059925e-037,       8.38715268e+155,        2.15375459e+263
};

const double DataSet_1_64f::outputs::ADDS[16] = {
    -1255.89283,    -1255.89283,    -1255.89283,    -1255.89283,
    -7.89101932e+067,       -1255.89283,    -1255.89283,    1.65804793e+064,
    -1.75556493e+130,       -1255.89283,    2.72984679e+103,        -1255.89283,
    -1255.89283,    -1255.89283,    8.38715268e+155,        -1255.89283
};

const double DataSet_1_64f::outputs::MADDS[16] = {
    -1255.89283,    1.71220004e-213,        9.0247961e-279, -1255.89283,
    -7.89101932e+067,       -1255.89283,    -1255.89283,    1.65804793e+064,
    -1.75556493e+130,       -1255.89283,    2.72984679e+103,        -1.19369643e-301,
    -1255.89283,    -7.30059925e-037,       8.38715268e+155,        -1255.89283
};

const double DataSet_1_64f::outputs::POSTPREFINC[16] = {
    1,      1,      1,      1,
    -7.89101932e+067,       1,      1,      1.65804793e+064,
    -1.75556493e+130,       1,      2.72984679e+103,        1,
    1,      1,      8.38715268e+155,        1
};

    const double DataSet_1_64f::outputs::MPOSTPREFINC[16] = {
    1,      1.71220004e-213,        9.0247961e-279, 1,
    -7.89101932e+067,       1,      1,      1.65804793e+064,
    -1.75556493e+130,       1,      2.72984679e+103,        -1.19369643e-301,
    1,      -7.30059925e-037,       8.38715268e+155,        1
};

const double DataSet_1_64f::outputs::SUBV[16] = {
    -3.70988913e+201,       -8.6009068e-154,        -2.92507601e+234,       -3.33009344e+098,
    -7.89101932e+067,       6.83346881e+035,        -5.48201446e-114,       -2.88938128e+097,
    -1.75556493e+130,       1.54038071e-018,        2.72984679e+103,        -6.45355305e-229,
    5.83594346e-068,        9.92124317e+294,        8.38715268e+155,        -2.15375459e+263
};

const double DataSet_1_64f::outputs::MSUBV[16] = {
    -3.70988913e+201,       1.71220004e-213,        9.0247961e-279, -3.33009344e+098,
    -7.89101932e+067,       6.83346881e+035,        -5.48201446e-114,       1.65804793e+064,
    -1.75556493e+130,       1.54038071e-018,        2.72984679e+103,        -1.19369643e-301,
    5.83594346e-068,        -7.30059925e-037,       8.38715268e+155,        -2.15375459e+263
};

const double DataSet_1_64f::outputs::SUBS[16] = {
    1255.89283,     1255.89283,     1255.89283,     1255.89283,
    -7.89101932e+067,       1255.89283,     1255.89283,     1.65804793e+064,
    -1.75556493e+130,       1255.89283,     2.72984679e+103,        1255.89283,
    1255.89283,     1255.89283,     8.38715268e+155,        1255.89283
};

const double DataSet_1_64f::outputs::MSUBS[16] = {
    1255.89283,     1.71220004e-213,        9.0247961e-279, 1255.89283,
    -7.89101932e+067,       1255.89283,     1255.89283,     1.65804793e+064,
    -1.75556493e+130,       1255.89283,     2.72984679e+103,        -1.19369643e-301,
    1255.89283,     -7.30059925e-037,       8.38715268e+155,        1255.89283
};

const double DataSet_1_64f::outputs::SUBFROMV[16] = {
    3.70988913e+201,        8.6009068e-154, 2.92507601e+234,        3.33009344e+098,
    7.89101932e+067,        -6.83346881e+035,       5.48201446e-114,        2.88938128e+097,
    1.75556493e+130,        -1.54038071e-018,       -2.72984679e+103,       6.45355305e-229,
    -5.83594346e-068,       -9.92124317e+294,       -8.38715268e+155,       2.15375459e+263
};

const double DataSet_1_64f::outputs::MSUBFROMV[16] = {
    3.70988913e+201,        8.6009068e-154, 2.92507601e+234,        3.33009344e+098,
    1.81587569e-274,        -6.83346881e+035,       5.48201446e-114,        2.88938128e+097,
    -2.99454657e-304,       -1.54038071e-018,       -2.72984679e+103,       6.45355305e-229,
    -5.83594346e-068,       -9.92124317e+294,       1.22560239e-074,        2.15375459e+263
};

const double DataSet_1_64f::outputs::SUBFROMS[16] = {
    -1255.89283,    -1255.89283,    -1255.89283,    -1255.89283,
    7.89101932e+067,        -1255.89283,    -1255.89283,    -1.65804793e+064,
    1.75556493e+130,        -1255.89283,    -2.72984679e+103,       -1255.89283,
    -1255.89283,    -1255.89283,    -8.38715268e+155,       -1255.89283
};

const double DataSet_1_64f::outputs::MSUBFROMS[16] = {
    -1255.89283,    -1255.89283,    -1255.89283,    -1255.89283,
    -1255.89283,    -1255.89283,    -1255.89283,    -1255.89283,
    -1255.89283,    -1255.89283,    -2.72984679e+103,       -1255.89283,
    -1255.89283,    -1255.89283,    -1255.89283,    -1255.89283
};

const double DataSet_1_64f::outputs::POSTPREFDEC[16] = {
    -1,     -1,     -1,     -1,
    -7.89101932e+067,       -1,     -1,     1.65804793e+064,
    -1.75556493e+130,       -1,     2.72984679e+103,        -1,
    -1,     -1,     8.38715268e+155,        -1
};

const double DataSet_1_64f::outputs::MPOSTPREFDEC[16] = {
    -1,     1.71220004e-213,        9.0247961e-279, -1,
    -7.89101932e+067,       -1,     -1,     1.65804793e+064,
    -1.75556493e+130,       -1,     2.72984679e+103,        -1.19369643e-301,
    -1,     -7.30059925e-037,       8.38715268e+155,        -1
};

const double DataSet_1_64f::outputs::MULV[16] = {
    -3.16152579e+157,       0,      2.63982146e-044,        5.89877865e-164,
    -1.43291101e-206,       2.9510543e-198, -0,     4.79073265e+161,
    5.25712094e-174,        5.92076696e-141,        -1.70275797e+191,       -0,
    1.88659129e-170,        7.24310205e+258,        1.02793144e+082,        3.28042432e+089
};

const double DataSet_1_64f::outputs::MMULV[16] = {
    -3.16152579e+157,       1.71220004e-213,        9.0247961e-279, 5.89877865e-164,
    -7.89101932e+067,       2.9510543e-198, -0,     1.65804793e+064,
    -1.75556493e+130,       5.92076696e-141,        -1.70275797e+191,       -1.19369643e-301,
    1.88659129e-170,        -7.30059925e-037,       8.38715268e+155,        3.28042432e+089
};

const double DataSet_1_64f::outputs::MULS[16] = {
    1.07025775e-041,        -2.15033976e-210,       -1.13341768e-275,       -2.22463242e-259,
    9.91027462e+070,        5.42361143e-231,        1.22994674e-269,        -2.08233051e+067,
    2.20480142e+133,        -1.93455309e-015,       -3.42839502e+106,       1.49915479e-298,
    4.05993735e-100,        9.16877029e-034,        -1.0533365e+159,        -1.91287412e-171
};

const double DataSet_1_64f::outputs::MMULS[16] = {
    1.07025775e-041,        1.71220004e-213,        9.0247961e-279, -2.22463242e-259,
    -7.89101932e+067,       5.42361143e-231,        1.22994674e-269,        1.65804793e+064,
    -1.75556493e+130,       -1.93455309e-015,       -3.42839502e+106,       -1.19369643e-301,
    4.05993735e-100,        -7.30059925e-037,       8.38715268e+155,        -1.91287412e-171
};

const double DataSet_1_64f::outputs::DIVV[16] = {
    -2.29707337e-246,       1.99072037e-060,        0,      0,
    -std::numeric_limits<double>::infinity(),        6.31967534e-270,        -1.78646103e-159,       5.7384186e-034,
    std::numeric_limits<double>::infinity(), 4.00754283e+104,        -4.37646668e+015,       -1.84967323e-073,
    5.53931004e-036,        0,      6.84329006e+229,        0
};

const double DataSet_1_64f::outputs::MDIVV[16] = {
    -2.29707337e-246,       1.71220004e-213,        9.0247961e-279, 0,
    -7.89101932e+067,       6.31967534e-270,        -1.78646103e-159,       1.65804793e+064,
    -1.75556493e+130,       4.00754283e+104,        -4.37646668e+015,       -1.19369643e-301,
    5.53931004e-036,        -7.30059925e-037,       8.38715268e+155,        0
};

const double DataSet_1_64f::outputs::DIVS[16] = {
    6.78552126e-048,        -1.36333291e-216,       -7.18596034e-282,       -1.41043507e-265,
    6.28319479e+064,        3.4386138e-237, 7.7979625e-276, -1.3202145e+061,
    1.39786205e+127,        -1.22652241e-021,       -2.17363036e+100,       9.50476343e-305,
    2.57403333e-106,        5.81307501e-040,        -6.67823914e+152,       -1.21277776e-177
};

const double DataSet_1_64f::outputs::MDIVS[16] = {
    6.78552126e-048,        1.71220004e-213,        9.0247961e-279, -1.41043507e-265,
    -7.89101932e+067,       3.4386138e-237, 7.7979625e-276, 1.65804793e+064,
    -1.75556493e+130,       -1.22652241e-021,       -2.17363036e+100,       -1.19369643e-301,
    2.57403333e-106,        -7.30059925e-037,       8.38715268e+155,        -1.21277776e-177
};

const double DataSet_1_64f::outputs::RCP[16] = {
    -1.17344895e+044,       5.84043908e+212,        1.10805827e+278,        5.64539481e+261,
    -1.2672634e-068,        -2.31560253e+233,       -1.0210953e+272,        6.03118874e-065,
    -5.69617212e-131,       6.49190162e+017,        3.66320925e-104,        -8.3773393e+300,
    -3.09337984e+102,       -1.36975057e+036,       1.19229974e-156,        6.56547561e+173
};

const double DataSet_1_64f::outputs::MRCP[16] = {
    -1.17344895e+044,       1.71220004e-213,        9.0247961e-279, 5.64539481e+261,
    -7.89101932e+067,       -2.31560253e+233,       -1.0210953e+272,        1.65804793e+064,
    -1.75556493e+130,       6.49190162e+017,        3.66320925e-104,        -1.19369643e-301,
    -3.09337984e+102,       -7.30059925e-037,       8.38715268e+155,        6.56547561e+173
};

const double DataSet_1_64f::outputs::RCPS[16] = {
    1.47372613e+047,        -7.33496559e+215,       -1.39160245e+281,       -7.09001089e+264,
    1.59154703e-065,        2.90814863e+236,        1.28238627e+275,        -7.57452672e-062,
    7.15378175e-128,        -8.15313273e+020,       -4.60059824e-101,       1.05210404e+304,
    3.88495358e+105,        1.72025993e+039,        -1.49740071e-153,       -8.24553377e+176
};

const double DataSet_1_64f::outputs::MRCPS[16] = {
    1.47372613e+047,        1.71220004e-213,        9.0247961e-279, -7.09001089e+264,
    -7.89101932e+067,       2.90814863e+236,        1.28238627e+275,        1.65804793e+064,
    -1.75556493e+130,       -8.15313273e+020,       -4.60059824e-101,       -1.19369643e-301,
    3.88495358e+105,        -7.30059925e-037,       8.38715268e+155,        -8.24553377e+176
};

const bool DataSet_1_64f::outputs::CMPEQV[16] = {
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false
};

const bool DataSet_1_64f::outputs::CMPEQS[16] = {
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false
};

const bool DataSet_1_64f::outputs::CMPNEV[16] = {
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true
};

const bool DataSet_1_64f::outputs::CMPNES[16] = {
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true
};

const bool DataSet_1_64f::outputs::CMPGTV[16] = {
    false,  false,  false,  false,
    false,  true,   false,  false,
    false,  true,   true,   false,
    true,   true,   true,   false
};

const bool DataSet_1_64f::outputs::CMPGTS[16] = {
    true,   true,   true,   true,
    false,  true,   true,   true,
    false,  true,   true,   true,
    true,   true,   true,   true
};

const bool DataSet_1_64f::outputs::CMPLTV[16] = {
    true,   true,   true,   true,
    true,   false,  true,   true,
    true,   false,  false,  true,
    false,  false,  false,  true
};

const bool DataSet_1_64f::outputs::CMPLTS[16] = {
    false,  false,  false,  false,
    true,   false,  false,  false,
    true,   false,  false,  false,
    false,  false,  false,  false
};

const bool DataSet_1_64f::outputs::CMPGEV[16] = {
    false,  false,  false,  false,
    false,  true,   false,  false,
    false,  true,   true,   false,
    true,   true,   true,   false
};

const bool DataSet_1_64f::outputs::CMPGES[16] = {
    true,   true,   true,   true,
    false,  true,   true,   true,
    false,  true,   true,   true,
    true,   true,   true,   true
};

const bool DataSet_1_64f::outputs::CMPLEV[16] = {
    true,   true,   true,   true,
    true,   false,  true,   true,
    true,   false,  false,  true,
    false,  false,  false,  true
};

const bool DataSet_1_64f::outputs::CMPLES[16] = {
    false,  false,  false,  false,
    true,   false,  false,  false,
    true,   false,  false,  false,
    false,  false,  false,  false
};

const bool  DataSet_1_64f::outputs::CMPEV = false;
const bool  DataSet_1_64f::outputs::CMPES = false;

const double DataSet_1_64f::outputs::HADD[16] = {
    -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045,
    -7.89101932e+067,       -7.89101932e+067,       -7.89101932e+067,       -7.88936127e+067,
    -1.75556493e+130,       -1.75556493e+130,       -1.75556493e+130,       -1.75556493e+130,
    -1.75556493e+130,       -1.75556493e+130,       8.38715268e+155,        8.38715268e+155
};

const double DataSet_1_64f::outputs::MHADD[16] = {
    -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045,
    -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045,
    -8.52188752e-045,       1.54038071e-018,        2.72984679e+103,        2.72984679e+103,
    2.72984679e+103,        2.72984679e+103,        2.72984679e+103,        2.72984679e+103
};

const double DataSet_1_64f::outputs::HMUL[16] = {
    -8.52188752e-045,       -1.45911761e-257,       -0,     -0,
    0,      -0,     0,      0,
    -0,     -0,     -0,     0,
    -0,     0,      0,      0
};

const double DataSet_1_64f::outputs::MHMUL[16] = {
    -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045,       -1.50952906e-306,
    -1.50952906e-306,       0,      -0,     -0,
    -0,     -0,     -0,     -0,
    0,      0,      0,      0
};

const double DataSet_1_64f::outputs::FMULADDV[16] = {
    6.97814145e+238,        4.7465557e+227, 2.20362602e+019,        4.74174268e+051,
    -1.65678425e-180,       -4.03722486e-155,       -9.46293922e-037,       4.79072558e+161,
    1.46063745e+071,        5.92076696e-141,        1.02056048e+241,        -2.54137336e-239,
    1.88659129e-170,        7.24310205e+258,        5.09403627e+197,        3.28042432e+089
};

const double DataSet_1_64f::outputs::MFMULADDV[16] = {
    6.97814145e+238,        1.71220004e-213,        9.0247961e-279, 4.74174268e+051,
    -7.89101932e+067,       -4.03722486e-155,       -9.46293922e-037,       1.65804793e+064,
    -1.75556493e+130,       5.92076696e-141,        1.02056048e+241,        -1.19369643e-301,
    1.88659129e-170,        -7.30059925e-037,       8.38715268e+155,        3.28042432e+089
};

const double DataSet_1_64f::outputs::FMULSUBV[16] = {
    -6.97814145e+238,       -4.7465557e+227,        -2.20362602e+019,       -4.74174268e+051,
    1.65678425e-180,        4.03722486e-155,        9.46293922e-037,        4.79073973e+161,
    -1.46063745e+071,       5.92076696e-141,        -1.02056048e+241,       2.54137336e-239,
    1.88659129e-170,        7.24310205e+258,        -5.09403627e+197,       3.28042432e+089
};

const double DataSet_1_64f::outputs::MFMULSUBV[16] = {
    -6.97814145e+238,       1.71220004e-213,        9.0247961e-279, -4.74174268e+051,
    -7.89101932e+067,       4.03722486e-155,        9.46293922e-037,        1.65804793e+064,
    -1.75556493e+130,       5.92076696e-141,        -1.02056048e+241,       -1.19369643e-301,
    1.88659129e-170,        -7.30059925e-037,       8.38715268e+155,        3.28042432e+089
};

const double DataSet_1_64f::outputs::FADDMULV[16] = {
    std::numeric_limits<double>::infinity(), 4.08246832e+074,        6.44577359e+253,        1.57904462e+150,
    1.30737165e-112,        2.75882502e-119,        -5.18759697e-150,       -2.0433014e+253,
    -2.56424388e+201,       -7.85313496e-252,       std::numeric_limits<double>::infinity(), -0,
    -0,     -1.18557571e+291,       std::numeric_limits<double>::infinity(), -6.52956604e-028
};

const double DataSet_1_64f::outputs::MFADDMULV[16] = {
    std::numeric_limits<double>::infinity(), 1.71220004e-213,        9.0247961e-279, 1.57904462e+150,
    -7.89101932e+067,       2.75882502e-119,        -5.18759697e-150,       1.65804793e+064,
    -1.75556493e+130,       -7.85313496e-252,       std::numeric_limits<double>::infinity(), -1.19369643e-301,
    -0,     -7.30059925e-037,       8.38715268e+155,        -6.52956604e-028
};

const double DataSet_1_64f::outputs::FSUBMULV[16] = {
    -std::numeric_limits<double>::infinity(),        -4.08246832e+074,       -6.44577359e+253,       -1.57904462e+150,
    1.30737165e-112,        -2.75882502e-119,       5.18759697e-150,        2.0433014e+253,
    -2.56424388e+201,       -7.85313496e-252,       std::numeric_limits<double>::infinity(), 0,
    0,      1.18557571e+291,        std::numeric_limits<double>::infinity(), 6.52956604e-028
};

const double DataSet_1_64f::outputs::MFSUBMULV[16] = {
    -std::numeric_limits<double>::infinity(),        1.71220004e-213,        9.0247961e-279, -1.57904462e+150,
    -7.89101932e+067,       -2.75882502e-119,       5.18759697e-150,        1.65804793e+064,
    -1.75556493e+130,       -7.85313496e-252,       std::numeric_limits<double>::infinity(), -1.19369643e-301,
    0,      -7.30059925e-037,       8.38715268e+155,        6.52956604e-028
};

const double DataSet_1_64f::outputs::MAXV[16] = {
    3.70988913e+201,        8.6009068e-154, 2.92507601e+234,        3.33009344e+098,
    1.81587569e-274,        -4.31853043e-234,       5.48201446e-114,        2.88938128e+097,
    -2.99454657e-304,       1.54038071e-018,        2.72984679e+103,        6.45355305e-229,
    -3.23271002e-103,       -7.30059925e-037,       8.38715268e+155,        2.15375459e+263
};

const double DataSet_1_64f::outputs::MMAXV[16] = {
    3.70988913e+201,        1.71220004e-213,        9.0247961e-279, 3.33009344e+098,
    -7.89101932e+067,       -4.31853043e-234,       5.48201446e-114,        1.65804793e+064,
    -1.75556493e+130,       1.54038071e-018,        2.72984679e+103,        -1.19369643e-301,
    -3.23271002e-103,       -7.30059925e-037,       8.38715268e+155,        2.15375459e+263
};

const double DataSet_1_64f::outputs::MAXS[16] = {
    -8.52188752e-045,       1.71220004e-213,        9.0247961e-279, 1.7713553e-262,
    -1255.89283,    -4.31853043e-234,       -9.79340522e-273,       1.65804793e+064,
    -1255.89283,    1.54038071e-018,        2.72984679e+103,        -1.19369643e-301,
    -3.23271002e-103,       -7.30059925e-037,       8.38715268e+155,        1.5231189e-174
};

const double DataSet_1_64f::outputs::MMAXS[16] = {
    -8.52188752e-045,       1.71220004e-213,        9.0247961e-279, 1.7713553e-262,
    -7.89101932e+067,       -4.31853043e-234,       -9.79340522e-273,       1.65804793e+064,
    -1.75556493e+130,       1.54038071e-018,        2.72984679e+103,        -1.19369643e-301,
    -3.23271002e-103,       -7.30059925e-037,       8.38715268e+155,        1.5231189e-174
};

const double DataSet_1_64f::outputs::MINV[16] = {
    -8.52188752e-045,       1.71220004e-213,        9.0247961e-279, 1.7713553e-262,
    -7.89101932e+067,       -6.83346881e+035,       -9.79340522e-273,       1.65804793e+064,
    -1.75556493e+130,       3.84370366e-123,        -6.23755872e+087,       -1.19369643e-301,
    -5.83594346e-068,       -9.92124317e+294,       1.22560239e-074,        1.5231189e-174
};

const double DataSet_1_64f::outputs::MMINV[16] = {
    -8.52188752e-045,       1.71220004e-213,        9.0247961e-279, 1.7713553e-262,
    -7.89101932e+067,       -6.83346881e+035,       -9.79340522e-273,       1.65804793e+064,
    -1.75556493e+130,       3.84370366e-123,        -6.23755872e+087,       -1.19369643e-301,
    -5.83594346e-068,       -7.30059925e-037,       8.38715268e+155,        1.5231189e-174
};

const double DataSet_1_64f::outputs::MINS[16] = {
    -1255.89283,    -1255.89283,    -1255.89283,    -1255.89283,
    -7.89101932e+067,       -1255.89283,    -1255.89283,    -1255.89283,
    -1.75556493e+130,       -1255.89283,    -1255.89283,    -1255.89283,
    -1255.89283,    -1255.89283,    -1255.89283,    -1255.89283
};

const double DataSet_1_64f::outputs::MMINS[16] = {
    -1255.89283,    1.71220004e-213,        9.0247961e-279, -1255.89283,
    -7.89101932e+067,       -1255.89283,    -1255.89283,    1.65804793e+064,
    -1.75556493e+130,       -1255.89283,    -1255.89283,    -1.19369643e-301,
    -1255.89283,    -7.30059925e-037,       8.38715268e+155,        -1255.89283
};

const double DataSet_1_64f::outputs::HMAX[16] = {
    -8.52188752e-045,       1.71220004e-213,        1.71220004e-213,        1.71220004e-213,
    1.71220004e-213,        1.71220004e-213,        1.71220004e-213,        1.65804793e+064,
    1.65804793e+064,        1.65804793e+064,        2.72984679e+103,        2.72984679e+103,
    2.72984679e+103,        2.72984679e+103,        8.38715268e+155,        8.38715268e+155
};

const double DataSet_1_64f::outputs::MHMAX[16] = {
    -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045,       1.7713553e-262,
    1.7713553e-262, 1.7713553e-262, 1.7713553e-262, 1.7713553e-262,
    1.7713553e-262, 1.54038071e-018,        2.72984679e+103,        2.72984679e+103,
    2.72984679e+103,        2.72984679e+103,        2.72984679e+103,        2.72984679e+103
};

const double DataSet_1_64f::outputs::HMIN[16] = {
    -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045,
    -7.89101932e+067,       -7.89101932e+067,       -7.89101932e+067,       -7.89101932e+067,
    -1.75556493e+130,       -1.75556493e+130,       -1.75556493e+130,       -1.75556493e+130,
    -1.75556493e+130,       -1.75556493e+130,       -1.75556493e+130,       -1.75556493e+130
};

const double DataSet_1_64f::outputs::MHMIN[16] = {
    -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045,
    -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045,
    -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045,
    -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045,       -8.52188752e-045
};

const double DataSet_1_64f::outputs::NEG[16] = {
    8.52188752e-045,        -1.71220004e-213,       -9.0247961e-279,        -1.7713553e-262,
    7.89101932e+067,        4.31853043e-234,        9.79340522e-273,        -1.65804793e+064,
    1.75556493e+130,        -1.54038071e-018,       -2.72984679e+103,       1.19369643e-301,
    3.23271002e-103,        7.30059925e-037,        -8.38715268e+155,       -1.5231189e-174
};

const double DataSet_1_64f::outputs::MNEG[16] = {
    8.52188752e-045,        1.71220004e-213,        9.0247961e-279, -1.7713553e-262,
    -7.89101932e+067,       4.31853043e-234,        9.79340522e-273,        1.65804793e+064,
    -1.75556493e+130,       -1.54038071e-018,       -2.72984679e+103,       -1.19369643e-301,
    3.23271002e-103,        -7.30059925e-037,       8.38715268e+155,        -1.5231189e-174
};

const double DataSet_1_64f::outputs::ABS[16] = {
    8.52188752e-045,        1.71220004e-213,        9.0247961e-279, 1.7713553e-262,
    7.89101932e+067,        4.31853043e-234,        9.79340522e-273,        1.65804793e+064,
    1.75556493e+130,        1.54038071e-018,        2.72984679e+103,        1.19369643e-301,
    3.23271002e-103,        7.30059925e-037,        8.38715268e+155,        1.5231189e-174
};

const double DataSet_1_64f::outputs::MABS[16] = {
    8.52188752e-045,        1.71220004e-213,        9.0247961e-279, 1.7713553e-262,
    -7.89101932e+067,       4.31853043e-234,        9.79340522e-273,        1.65804793e+064,
    -1.75556493e+130,       1.54038071e-018,        2.72984679e+103,        -1.19369643e-301,
    3.23271002e-103,        -7.30059925e-037,       8.38715268e+155,        1.5231189e-174
};

const double DataSet_1_64f::outputs::SQR[16] = {
    0.000000000000000,      0.000000000000000,      0.000000000000000,      0.000000000000000,
    6226818590838841800000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,       0.000000000000000,      0.000000000000000,      274912293581145570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,
    308200822111577820000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,  0.000000000000000,      745206349450698970000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,        0.000000000000000,
    0.000000000000000,      0.000000000000000,      std::numeric_limits<double>::infinity(),        0.000000000000000
};

const double DataSet_1_64f::outputs::MSQR[16] = {
    0.000000000000000,      0.000000000000000,      0.000000000000000,      0.000000000000000,
    -78910193199857537000000000000000000000000000000000000000000000000000.000000000000000,  0.000000000000000,      0.000000000000000,      16580479292865620000000000000000000000000000000000000000000000000.000000000000000,
    -17555649293363600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   0.000000000000000,      745206349450698970000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,        -0.000000000000000,
    0.000000000000000,      -0.000000000000000,     838715268466503690000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   0.000000000000000
};

const double DataSet_1_64f::outputs::SQRT[16] = {
    0.000000000000000,      0.000000000000000,      0.000000000000000,      0.000000000000000,
    8883140953506114800000000000000000.000000000000000,     0.000000000000000,      0.000000000000000,      128765209947662580000000000000000.000000000000000,
    132497733163113400000000000000000000000000000000000000000000000000.000000000000000,     0.000000001241121,      5224793574454961200000000000000000000000000000000000.000000000000000,   0.000000000000000,
    0.000000000000000,      0.000000000000000,      915813992285826200000000000000000000000000000000000000000000000000000000000000.000000000000000, 0.000000000000000
};

const double DataSet_1_64f::outputs::MSQRT[16] = {
    0.000000000000000,      0.000000000000000,      0.000000000000000,      0.000000000000000,
    -78910193199857537000000000000000000000000000000000000000000000000000.000000000000000,  0.000000000000000,      0.000000000000000,      16580479292865620000000000000000000000000000000000000000000000000.000000000000000,
    -17555649293363600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   0.000000001241121,      5224793574454961200000000000000000000000000000000000.000000000000000,   -0.000000000000000,
    0.000000000000000,      -0.000000000000000,     838715268466503690000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   0.000000000000000
};

const double DataSet_1_64f::outputs::ROUND[16] = {
    0.000000000000000,      0.000000000000000,      0.000000000000000,      0.000000000000000,
    -78910193199857537000000000000000000000000000000000000000000000000000.000000000000000,  0.000000000000000,      0.000000000000000,      16580479292865620000000000000000000000000000000000000000000000000.000000000000000,
    -17555649293363600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   0.000000000000000,      27298467895665845000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,       0.000000000000000,
    0.000000000000000,      0.000000000000000,      838715268466503690000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   0.000000000000000
};

const double DataSet_1_64f::outputs::MROUND[16] = {
    0.000000000000000,      0.000000000000000,      0.000000000000000,      0.000000000000000,
    -78910193199857537000000000000000000000000000000000000000000000000000.000000000000000,  0.000000000000000,      0.000000000000000,      16580479292865620000000000000000000000000000000000000000000000000.000000000000000,
    -17555649293363600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   0.000000000000000,      27298467895665845000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,       -0.000000000000000,
    0.000000000000000,      -0.000000000000000,     838715268466503690000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   0.000000000000000
};

const int64_t DataSet_1_64f::outputs::TRUNC[16] = {
    /*
    0ll,    0ll,    0ll,    0ll,
    -9223372036854775808ll, 0ll,    0ll,    -9223372036854775808ll,
    -9223372036854775808ll, 0ll,    -9223372036854775808ll, 0ll,
    0ll,    0ll,    -9223372036854775808ll, 0ll*/
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

const int64_t DataSet_1_64f::outputs::MTRUNC[16] = {
    /*0ll,    0ll,    0ll,    0ll,
    0ll,    0ll,    0ll,    0ll,
    0ll,    0ll,    -9223372036854775808ll,  0ll,
    0ll,    0ll,    0ll,    0ll*/
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

const double DataSet_1_64f::outputs::FLOOR[16] = {
    -1.000000000000000,     0.000000000000000,      0.000000000000000,      0.000000000000000,
    -78910193199857537000000000000000000000000000000000000000000000000000.000000000000000,  -1.000000000000000,     -1.000000000000000,     16580479292865620000000000000000000000000000000000000000000000000.000000000000000,
    -17555649293363600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   0.000000000000000,      27298467895665845000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,       -1.000000000000000,
    -1.000000000000000,     -1.000000000000000,     838715268466503690000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   0.000000000000000
};

const double DataSet_1_64f::outputs::MFLOOR[16] = {
    -1.000000000000000,     0.000000000000000,      0.000000000000000,      0.000000000000000,
    -78910193199857537000000000000000000000000000000000000000000000000000.000000000000000,  -1.000000000000000,     -1.000000000000000,     16580479292865620000000000000000000000000000000000000000000000000.000000000000000,
    -17555649293363600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   0.000000000000000,      27298467895665845000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,       -0.000000000000000,
    -1.000000000000000,     -0.000000000000000,     838715268466503690000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   0.000000000000000
};

const double DataSet_1_64f::outputs::CEIL[16] = {
    -0.000000000000000,     1.000000000000000,      1.000000000000000,      1.000000000000000,
    -78910193199857537000000000000000000000000000000000000000000000000000.000000000000000,  -0.000000000000000,     -0.000000000000000,     16580479292865620000000000000000000000000000000000000000000000000.000000000000000,
    -17555649293363600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   1.000000000000000,      27298467895665845000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,       -0.000000000000000,
    -0.000000000000000,     -0.000000000000000,     838715268466503690000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   1.000000000000000
};

const double DataSet_1_64f::outputs::MCEIL[16] = {
    -0.000000000000000,     0.000000000000000,      0.000000000000000,      1.000000000000000,
    -78910193199857537000000000000000000000000000000000000000000000000000.000000000000000,  -0.000000000000000,     -0.000000000000000,     16580479292865620000000000000000000000000000000000000000000000000.000000000000000,
    -17555649293363600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   1.000000000000000,      27298467895665845000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,       -0.000000000000000,
    -0.000000000000000,     -0.000000000000000,     838715268466503690000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   1.000000000000000
};

const bool DataSet_1_64f::outputs::ISFIN[16] = {
    true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true
};

const bool DataSet_1_64f::outputs::ISINF[16] = {
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false
};


const bool DataSet_1_64f::outputs::ISAN[16] = {
    true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true
};

const bool DataSet_1_64f::outputs::ISNAN[16] = {
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false
};

const bool DataSet_1_64f::outputs::ISNORM[16] = {
    true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true
};

const bool DataSet_1_64f::outputs::ISSUB[16] = {
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false
};

const bool DataSet_1_64f::outputs::ISZERO[16] = {
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false
};

const bool DataSet_1_64f::outputs::ISZEROSUB[16] = {
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false
};

const double DataSet_1_64f::outputs::SIN[16] = {
    -0.000000000000000,     0.000000000000000,      0.000000000000000,      0.000000000000000,
    0.536427719813807,      -0.000000000000000,     -0.000000000000000,     -0.817606036021875,
    0.162170622988973,      0.000000000000000,      0.923880169211420,      -0.000000000000000,
    -0.000000000000000,     -0.000000000000000,     -0.539255531982249,     0.000000000000000
};

const double DataSet_1_64f::outputs::MSIN[16] = {
    -0.000000000000000,     0.000000000000000,      0.000000000000000,      0.000000000000000,
    -78910193199857537000000000000000000000000000000000000000000000000000.000000000000000,  -0.000000000000000,     -0.000000000000000,     16580479292865620000000000000000000000000000000000000000000000000.000000000000000,
    -17555649293363600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   0.000000000000000,      0.923880169211420,      -0.000000000000000,
    -0.000000000000000,     -0.000000000000000,     838715268466503690000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   0.000000000000000
};

const double DataSet_1_64f::outputs::COS[16] = {
    1.000000000000000,      1.000000000000000,      1.000000000000000,      1.000000000000000,
    0.843946266900541,      1.000000000000000,      1.000000000000000,      -0.575778056077684,
    0.986762731886125,      1.000000000000000,      -0.382681895231377,     1.000000000000000,
    1.000000000000000,      1.000000000000000,      -0.842142191809995,     1.000000000000000
};

const double DataSet_1_64f::outputs::MCOS[16] = {
    1.000000000000000,      0.000000000000000,      0.000000000000000,      1.000000000000000,
    -78910193199857537000000000000000000000000000000000000000000000000000.000000000000000,  1.000000000000000,      1.000000000000000,      16580479292865620000000000000000000000000000000000000000000000000.000000000000000,
    -17555649293363600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   1.000000000000000,      -0.382681895231377,     -0.000000000000000,
    1.000000000000000,      -0.000000000000000,     838715268466503690000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   1.000000000000000
};

const double DataSet_1_64f::outputs::TAN[16] = {
    -0.000000000000000,     0.000000000000000,      0.000000000000000,      0.000000000000000,
    0.635618333598275,      -0.000000000000000,     -0.000000000000000,     1.420002077869332,
    0.164346116597853,      0.000000000000000,      -2.414224923425823,     -0.000000000000000,
    -0.000000000000000,     -0.000000000000000,     0.640337863637067,      0.000000000000000
};

const double DataSet_1_64f::outputs::MTAN[16] = {
    -0.000000000000000,     0.000000000000000,      0.000000000000000,      0.000000000000000,
    -78910193199857537000000000000000000000000000000000000000000000000000.000000000000000,  -0.000000000000000,     -0.000000000000000,     16580479292865620000000000000000000000000000000000000000000000000.000000000000000,
    -17555649293363600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   0.000000000000000,      std::numeric_limits<double>::quiet_NaN(),     -0.000000000000000,
    -0.000000000000000,     -0.000000000000000,     838715268466503690000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   0.000000000000000
};

const double DataSet_1_64f::outputs::CTAN[16] = {
    -117344895373057720000000000000000000000000000.000000000000000, 584043907541854480000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,  110805827475531760000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,        5645394810185215100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,
    1.573271170985484,      -231560253016244510000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,    -102109529555705290000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,     0.704224321629492,
    6.084719375797326,      649190161980107780.000000000000000,     -0.414211613133785,     -8377339297364429000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,
    -3093379841655376900000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,       -1369750571733762500000000000000000000.000000000000000, 1.561675572829758,      656547560701579300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000
};

const double DataSet_1_64f::outputs::MCTAN[16] = {
    -117344895373057720000000000000000000000000000.000000000000000, 0.000000000000000,      0.000000000000000,      5645394810185215100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,
    -78910193199857537000000000000000000000000000000000000000000000000000.000000000000000,  -231560253016244510000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,    -102109529555705290000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,     16580479292865620000000000000000000000000000000000000000000000000.000000000000000,
    -17555649293363600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   649190161980107780.000000000000000,     -0.414211613133785,     -0.000000000000000,
    -3093379841655376900000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,       -0.000000000000000,     838715268466503690000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000,   656547560701579300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.000000000000000
};

const uint64_t DataSet_1_64f::outputs::FTOU[16] = {
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x8000000000000000,     0x0000000000000000,     0x0000000000000000,     0x8000000000000000,
    0x8000000000000000,     0x0000000000000000,     0x8000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x8000000000000000,     0x0000000000000000
};

const int64_t DataSet_1_64f::outputs::FTOI[16] = {
    /*0ll,      0ll,      0ll,      0ll,
    -9223372036854775808ll,   0ll,      0,      -9223372036854775808ll,
    -9223372036854775808ll,   0ll,      -9223372036854775808ll,   0ll,
    0ll,      0ll,      -9223372036854775808ll,   0ll*/
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};
