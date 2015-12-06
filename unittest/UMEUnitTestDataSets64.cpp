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
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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


const int64_t DataSet_1_64i::outputs::ADDV[16] = {
    0x1d0283212871bd12,     0x4e9294909944b4b8,     0x5cc5100aa414ce10,     0xceee3c7b5d7e8f1c,
    0xc76895b0a978c5de,     0x6e8f6b74be05e02f,     0xcb3838237f6d3716,     0x40ae86c8adf465a2,
    0x6a2040ad5d164344,     0x8ba0ba7b2c3c99a2,     0xd09c58cdb8f32df0,     0x8fd0244bc2a8b5cf,
    0x4cc6a53d77d94f98,     0x2ec3deac64d89644,     0x254d40ef7bf768b4,     0xf2156bcaaf487c95
};

const int64_t DataSet_1_64i::outputs::MADDV[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0xceee3c7b5d7e8f1c,
    0x5d793f3374314a60,     0x6e8f6b74be05e02f,     0xcb3838237f6d3716,     0x115e106678b9182f,
    0x566b140f478f277a,     0x8ba0ba7b2c3c99a2,     0xd09c58cdb8f32df0,     0x662919af46d76ce4,
    0x4cc6a53d77d94f98,     0x247b781e57de4b53,     0x10c6192b0d300338,     0xf2156bcaaf487c95
};

const int64_t DataSet_1_64i::outputs::ADDS[16] = {
    0x0e550ac115c750f5,     0x475a3b35581d5c1d,     0x47a80c3c49744c18,     0x57fc35ae4c121ca1,
    0x5d793f3363ce31c3,     0x1d4a455936d36591,     0x638814e40ba21e18,     0x115e10666855ff92,
    0x566b140f372c0edd,     0x19674bc51442582b,     0x540a2f77289eedfa,     0x662919af36745447,
    0x24232c673aa12bfe,     0x247b781e477b32b6,     0x10c6192afcccea9b,     0x7dc83f50411cfeab
};

const int64_t DataSet_1_64i::outputs::MADDS[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x57fc35ae4c121ca1,
    0x5d793f3374314a60,     0x1d4a455936d36591,     0x638814e40ba21e18,     0x115e106678b9182f,
    0x566b140f478f277a,     0x19674bc51442582b,     0x540a2f77289eedfa,     0x662919af46d76ce4,
    0x24232c673aa12bfe,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x7dc83f50411cfeab
};

const int64_t DataSet_1_64i::outputs::POSTPREFINC[16] = {
    0x0e550ac1262a6993,     0x475a3b35688074bb,     0x47a80c3c59d764b6,     0x57fc35ae5c75353f,
    0x5d793f3374314a61,     0x1d4a455947367e2f,     0x638814e41c0536b6,     0x115e106678b91830,
    0x566b140f478f277b,     0x19674bc524a570c9,     0x540a2f7739020698,     0x662919af46d76ce5,
    0x24232c674b04449c,     0x247b781e57de4b54,     0x10c6192b0d300339,     0x7dc83f5051801749
};

const int64_t DataSet_1_64i::outputs::MPOSTPREFINC[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x57fc35ae5c75353f,
    0x5d793f3374314a60,     0x1d4a455947367e2f,     0x638814e41c0536b6,     0x115e106678b9182f,
    0x566b140f478f277a,     0x19674bc524a570c9,     0x540a2f7739020698,     0x662919af46d76ce4,
    0x24232c674b04449c,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x7dc83f5051801749
};

const int64_t DataSet_1_64i::outputs::SUBV[16] = {
    0xffa7926123e31612,     0x4021e1da37bc34bc,     0x328b086e0f99fb5a,     0xe10a2ee15b6bdb60,
    0xf389e8b63ee9cee2,     0xcc051f3dd0671c2d,     0xfbd7f1a4b89d3654,     0xe20d9a04437dcabc,
    0x42b5e77132080bb0,     0xa72ddd0f1d0e47ee,     0xd7780620b910df3e,     0x3c820f12cb0623f9,
    0xfb7fb3911e2f399e,     0x1a3311904ae40062,     0xfc3ef1669e689dbc,     0x097b12d5f3b7b1fb
};

const int64_t DataSet_1_64i::outputs::MSUBV[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0xe10a2ee15b6bdb60,
    0x5d793f3374314a60,     0xcc051f3dd0671c2d,     0xfbd7f1a4b89d3654,     0x115e106678b9182f,
    0x566b140f478f277a,     0xa72ddd0f1d0e47ee,     0xd7780620b910df3e,     0x662919af46d76ce4,
    0xfb7fb3911e2f399e,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x097b12d5f3b7b1fb
};

const int64_t DataSet_1_64i::outputs::SUBS[16] = {
    0x0e550ac1368d822f,     0x475a3b3578e38d57,     0x47a80c3c6a3a7d52,     0x57fc35ae6cd84ddb,
    0x5d793f33849462fd,     0x1d4a4559579996cb,     0x638814e42c684f52,     0x115e1066891c30cc,
    0x566b140f57f24017,     0x19674bc535088965,     0x540a2f7749651f34,     0x662919af573a8581,
    0x24232c675b675d38,     0x247b781e684163f0,     0x10c6192b1d931bd5,     0x7dc83f5061e32fe5
};

const int64_t DataSet_1_64i::outputs::MSUBS[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x57fc35ae6cd84ddb,
    0x5d793f3374314a60,     0x1d4a4559579996cb,     0x638814e42c684f52,     0x115e106678b9182f,
    0x566b140f478f277a,     0x19674bc535088965,     0x540a2f7749651f34,     0x662919af46d76ce4,
    0x24232c675b675d38,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x7dc83f5061e32fe5
};

const int64_t DataSet_1_64i::outputs::SUBFROMV[16] = {
    0x00586d9edc1ce9ee,     0xbfde1e25c843cb44,     0xcd74f791f06604a6,     0x1ef5d11ea49424a0,
    0x0c761749c116311e,     0x33fae0c22f98e3d3,     0x04280e5b4762c9ac,     0x1df265fbbc823544,
    0xbd4a188ecdf7f450,     0x58d222f0e2f1b812,     0x2887f9df46ef20c2,     0xc37df0ed34f9dc07,
    0x04804c6ee1d0c662,     0xe5ccee6fb51bff9e,     0x03c10e9961976244,     0xf684ed2a0c484e05
};

const int64_t DataSet_1_64i::outputs::MSUBFROMV[16] = {
    0x0ead786002475380,     0x0738595b30c43ffe,     0x151d03ce4a3d695b,     0x1ef5d11ea49424a0,
    0x69ef567d35477b7e,     0x33fae0c22f98e3d3,     0x04280e5b4762c9ac,     0x2f507662353b4d73,
    0x13b52c9e15871bca,     0x58d222f0e2f1b812,     0x2887f9df46ef20c2,     0x29a70a9c7bd148eb,
    0x04804c6ee1d0c662,     0x0a48668e0cfa4af1,     0x148727c46ec7657c,     0xf684ed2a0c484e05
};

const int64_t DataSet_1_64i::outputs::SUBFROMS[16] = {
    0xf1aaf53ec9727dd1,     0xb8a5c4ca871c72a9,     0xb857f3c395c582ae,     0xa803ca519327b225,
    0xa286c0cc7b6b9d03,     0xe2b5baa6a8666935,     0x9c77eb1bd397b0ae,     0xeea1ef9976e3cf34,
    0xa994ebf0a80dbfe9,     0xe698b43acaf7769b,     0xabf5d088b69ae0cc,     0x99d6e650a8c57a7f,
    0xdbdcd398a498a2c8,     0xdb8487e197be9c10,     0xef39e6d4e26ce42b,     0x8237c0af9e1cd01b
};

const int64_t DataSet_1_64i::outputs::MSUBFROMS[16] = {
    0xffffffffef9ce763,     0xffffffffef9ce763,     0xffffffffef9ce763,     0xa803ca519327b225,
    0xffffffffef9ce763,     0xe2b5baa6a8666935,     0x9c77eb1bd397b0ae,     0xffffffffef9ce763,
    0xffffffffef9ce763,     0xe698b43acaf7769b,     0xabf5d088b69ae0cc,     0xffffffffef9ce763,
    0xdbdcd398a498a2c8,     0xffffffffef9ce763,     0xffffffffef9ce763,     0x8237c0af9e1cd01b
};

const int64_t DataSet_1_64i::outputs::POSTPREFDEC[16] = {
    0x0e550ac1262a6991,     0x475a3b35688074b9,     0x47a80c3c59d764b4,     0x57fc35ae5c75353d,
    0x5d793f3374314a5f,     0x1d4a455947367e2d,     0x638814e41c0536b4,     0x115e106678b9182e,
    0x566b140f478f2779,     0x19674bc524a570c7,     0x540a2f7739020696,     0x662919af46d76ce3,
    0x24232c674b04449a,     0x247b781e57de4b52,     0x10c6192b0d300337,     0x7dc83f5051801747
};

const int64_t DataSet_1_64i::outputs::MPOSTPREFDEC[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x57fc35ae5c75353d,
    0x5d793f3374314a60,     0x1d4a455947367e2d,     0x638814e41c0536b4,     0x115e106678b9182f,
    0x566b140f478f277a,     0x19674bc524a570c7,     0x540a2f7739020696,     0x662919af46d76ce4,
    0x24232c674b04449a,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x7dc83f5051801747
};

const int64_t DataSet_1_64i::outputs::MULV[16] = {
    0xbfedf14940ed1f00,     0xc583240f8a95968c,     0xd51cee6299000957,     0x3e94f7530a54b9c4,
    0x8c0bd3185f9ebb40,     0xefba48af5eb61a2e,     0x7544863fd681ba95,     0xce7d41003541001d,
    0x0cee40a163750444,     0x2bb295760c794a50,     0xf924bc093ddc4b7f,     0xa85979f41085154c,
    0x65c012ed06dcdb2f,     0x7f5dee9e6d18e723,     0xd3cc0da2e50ea720,     0xf644682764f668a8
};

const int64_t DataSet_1_64i::outputs::MMULV[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x3e94f7530a54b9c4,
    0x5d793f3374314a60,     0xefba48af5eb61a2e,     0x7544863fd681ba95,     0x115e106678b9182f,
    0x566b140f478f277a,     0x2bb295760c794a50,     0xf924bc093ddc4b7f,     0x662919af46d76ce4,
    0x65c012ed06dcdb2f,     0x247b781e57de4b53,     0x10c6192b0d300338,     0xf644682764f668a8
};

const int64_t DataSet_1_64i::outputs::MULS[16] = {
    0x01c1e2f7aaa19176,     0x8bed3d7a1a58f9ee,     0xcbd6a1f9737744ff,     0xbd15a6bed82688fa,
    0xa68e66d75bac6320,     0xacf9279c8bf64dca,     0xdd9ea40adbad7aff,     0xc89a14bf520ac32d,
    0x4011a517cd535a2e,     0x1d0489a1e69f1558,     0xb2a28a57dcbecd65,     0x9b2cf4bfff80d82c,
    0xe9a1ceae600264f1,     0xb3ca9d7af4830619,     0xb7ad6d97aa98c6a8,     0x951a1e1c816af8d8
};

const int64_t DataSet_1_64i::outputs::MMULS[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0xbd15a6bed82688fa,
    0x5d793f3374314a60,     0xacf9279c8bf64dca,     0xdd9ea40adbad7aff,     0x115e106678b9182f,
    0x566b140f478f277a,     0x1d0489a1e69f1558,     0xb2a28a57dcbecd65,     0x662919af46d76ce4,
    0xe9a1ceae600264f1,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x951a1e1c816af8d8
};

const int64_t DataSet_1_64i::outputs::DIVV[16] = {
    0x0000000000000000,     0x0000000000000009,     0x0000000000000003,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000004,     0x0000000000000000,     0x0000000000000000,     0x0000000000000002,
    0x0000000000000000,     0x0000000000000003,     0x0000000000000000,     0x0000000000000001
};

const int64_t DataSet_1_64i::outputs::MDIVV[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x0000000000000000,
    0x5d793f3374314a60,     0x0000000000000000,     0x0000000000000000,     0x115e106678b9182f,
    0x566b140f478f277a,     0x0000000000000000,     0x0000000000000000,     0x662919af46d76ce4,
    0x0000000000000000,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x0000000000000001
};

const int64_t DataSet_1_64i::outputs::DIVS[16] = {
    0xffffffff201a0b37,     0xfffffffba5540777,     0xfffffffba0946031,     0xfffffffaa17dadea,
    0xfffffffa4bc01e66,     0xfffffffe366da3eb,     0xfffffff9ed1ce6d8,     0xfffffffef0af5de3,
    0xfffffffab9f82c45,     0xfffffffe73253013,     0xfffffffadf205426,     0xfffffff9c40af848,
    0xfffffffdcb75b5ab,     0xfffffffdc612594f,     0xfffffffef9f5621c,     0xfffffff8530623c6
};

const int64_t DataSet_1_64i::outputs::MDIVS[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0xfffffffaa17dadea,
    0x5d793f3374314a60,     0xfffffffe366da3eb,     0xfffffff9ed1ce6d8,     0x115e106678b9182f,
    0x566b140f478f277a,     0xfffffffe73253013,     0xfffffffadf205426,     0x662919af46d76ce4,
    0xfffffffdcb75b5ab,     0x247b781e57de4b53,     0x10c6192b0d300338,     0xfffffff8530623c6
};

const int64_t DataSet_1_64i::outputs::RCP[16] = {
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000
};

const int64_t DataSet_1_64i::outputs::MRCP[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x0000000000000000,
    0x5d793f3374314a60,     0x0000000000000000,     0x0000000000000000,     0x115e106678b9182f,
    0x566b140f478f277a,     0x0000000000000000,     0x0000000000000000,     0x662919af46d76ce4,
    0x0000000000000000,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x0000000000000000
};

const int64_t DataSet_1_64i::outputs::RCPS[16] = {
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000
};

const int64_t DataSet_1_64i::outputs::MRCPS[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x0000000000000000,
    0x5d793f3374314a60,     0x0000000000000000,     0x0000000000000000,     0x115e106678b9182f,
    0x566b140f478f277a,     0x0000000000000000,     0x0000000000000000,     0x662919af46d76ce4,
    0x0000000000000000,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x0000000000000000
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
    0x0e550ac1262a6992,     0x55af45f68eaade4c,     0x9d575232e8824301,     0xf55387e144f7783f,
    0x52ccc714b928c29f,     0x70170c6e005f40cd,     0xd39f21521c647782,     0xe4fd31b8951d8fb1,
    0x3b6845c7dcacb72b,     0x54cf918d015227f3,     0xa8d9c1043a542e8a,     0x0f02dab3812b9b6e,
    0x3326071acc2fe009,     0x57a17f39240e2b5c,     0x68679864313e2e94,     0xe62fd7b482be45dc
};

const int64_t DataSet_1_64i::outputs::MHADD[16] = {
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x57fc35ae5c75353e,
    0x57fc35ae5c75353e,     0x75467b07a3abb36c,     0xd8ce8febbfb0ea21,     0xd8ce8febbfb0ea21,
    0xd8ce8febbfb0ea21,     0xf235dbb0e4565ae9,     0x46400b281d586180,     0x46400b281d586180,
    0x6a63378f685ca61b,     0x6a63378f685ca61b,     0x6a63378f685ca61b,     0xe82b76dfb9dcbd63
};

const int64_t DataSet_1_64i::outputs::HMUL[16] = {
    0x0e550ac1262a6992,     0x132e26f30ba6dc14,     0x5290caba33bd6a24,     0xa3e2f5fd314d28b8,
    0xc6613a12f5ec7500,     0x0effe5c640130600,     0x7eaff3b66eb73e00,     0xb8c5745b4f746200,
    0x2a57c3cf5664b400,     0x3f3f0bff916ca000,     0x92b6c3a092d26000,     0x427e6de223dd8000,
    0xcfe58d688d1c8000,     0x35d0887419bd8000,     0x869f28bad9f40000,     0xce0f6c2238a00000
};

const int64_t DataSet_1_64i::outputs::MHMUL[16] = {
    0x0000000000000001,     0x0000000000000001,     0x0000000000000001,     0x57fc35ae5c75353e,
    0x57fc35ae5c75353e,     0xc62eb1e1ba581524,     0x218d4bffae708a74,     0x218d4bffae708a74,
    0x218d4bffae708a74,     0x35cd05571142eaa0,     0xf102a9e635382460,     0xf102a9e635382460,
    0x43e56eadd4278620,     0x43e56eadd4278620,     0x43e56eadd4278620,     0x7bdf1e26682a9900
};

const int64_t DataSet_1_64i::outputs::FMULADDV[16] = {
    0xcee3055a42688671,     0x43234edec6d3dafa,     0xf53161d7d687767b,     0xa4160c137e81e5a0,
    0xf6a6d8ffb23a1c65,     0x1604a12e5fdb5700,     0xc3e6dfdbe11dc2d7,     0xd8928f3a6c606601,
    0x4dd58f7c6caa2e3d,     0x4e959be71d927aec,     0x2e7214864831a07f,     0x2570ac6d88178346,
    0x7f171f61378c4eee,     0xe1574b9a84291a11,     0x327d11e34947d7f1,     0x2078da9dab918d88
};

const int64_t DataSet_1_64i::outputs::MFMULADDV[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0xa4160c137e81e5a0,
    0x5d793f3374314a60,     0x1604a12e5fdb5700,     0xc3e6dfdbe11dc2d7,     0x115e106678b9182f,
    0x566b140f478f277a,     0x4e959be71d927aec,     0x2e7214864831a07f,     0x662919af46d76ce4,
    0x7f171f61378c4eee,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x2078da9dab918d88
};

const int64_t DataSet_1_64i::outputs::FMULSUBV[16] = {
    0xb0f8dd383f71b78f,     0x47e2f9404e57521e,     0xb5087aed5b789c33,     0xd913e29296278de8,
    0x2170cd310d035a1b,     0xc96ff0305d90dd5c,     0x26a22ca3cbe5b253,     0xc467f2c5fe219a39,
    0xcc06f1c65a3fda4b,     0x08cf8f04fb6019b4,     0xc3d7638c3386f67f,     0x2b42477a98f2a752,
    0x4c690678d62d6770,     0x1d6491a25608b435,     0x751b096280d5764f,     0xcc0ff5b11e5b43c8
};

const int64_t DataSet_1_64i::outputs::MFMULSUBV[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0xd913e29296278de8,
    0x5d793f3374314a60,     0xc96ff0305d90dd5c,     0x26a22ca3cbe5b253,     0x115e106678b9182f,
    0x566b140f478f277a,     0x08cf8f04fb6019b4,     0xc3d7638c3386f67f,     0x662919af46d76ce4,
    0x4c690678d62d6770,     0x247b781e57de4b53,     0x10c6192b0d300338,     0xcc0ff5b11e5b43c8
};

const int64_t DataSet_1_64i::outputs::FADDMULV[16] = {
    0x73820f7d86ecb2f2,     0x9097c89f00168710,     0x6193a2be6919ca40,     0x8c26ffd072b8b010,
    0xac56bf922ad7b716,     0x5d035e57d727ea8e,     0xed7bb73eb148e3ac,     0x70ea2c7e76616e48,
    0xfccafdad6c815124,     0x4ddfaffb148dfeb8,     0xd6bc25225ef0b000,     0x5942816fd738af26,
    0x62c8c30cf2cbaa68,     0x2ac4cdf18ef4fb38,     0x49ef26bf38b23af4,     0x58b48ba1bd28f660
};

const int64_t DataSet_1_64i::outputs::MFADDMULV[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x8c26ffd072b8b010,
    0x5d793f3374314a60,     0x5d035e57d727ea8e,     0xed7bb73eb148e3ac,     0x115e106678b9182f,
    0x566b140f478f277a,     0x4ddfaffb148dfeb8,     0xd6bc25225ef0b000,     0x662919af46d76ce4,
    0x62c8c30cf2cbaa68,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x58b48ba1bd28f660
};

const int64_t DataSet_1_64i::outputs::FSUBMULV[16] = {
    0x35b4b2b4e1c3fbf2,     0x19cc1a86c66898c8,     0x2a374558c022aaa8,     0xdf121d62be69a680,
    0xe774c3f0560488aa,     0xf91e15e35eb0a8ea,     0x7da86558e56aa1a8,     0xda9643ceadc8bb70,
    0xb67281c08f228e30,     0x9dd4a7e2366e7508,     0x4fc14a0b25b59600,     0xc5d1ffb62452262a,
    0xde992a82bf1ff6e2,     0x8e8d4586fc2b7f1c,     0xe5fad4637fd8067c,     0xfb6f2c107abc07a0
};

const int64_t DataSet_1_64i::outputs::MFSUBMULV[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0xdf121d62be69a680,
    0x5d793f3374314a60,     0xf91e15e35eb0a8ea,     0x7da86558e56aa1a8,     0x115e106678b9182f,
    0x566b140f478f277a,     0x9dd4a7e2366e7508,     0x4fc14a0b25b59600,     0x662919af46d76ce4,
    0xde992a82bf1ff6e2,     0x247b781e57de4b53,     0x10c6192b0d300338,     0xfb6f2c107abc07a0
};

const int64_t DataSet_1_64i::outputs::MAXV[16] = {
    0x0ead786002475380,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x76f206cd010959de,
    0x69ef567d35477b7e,     0x5145261b76cf6201,     0x67b0233f63680061,     0x2f507662353b4d73,
    0x566b140f478f277a,     0x72396eb6079728da,     0x7c9229567ff12759,     0x662919af46d76ce4,
    0x28a378d62cd50afd,     0x247b781e57de4b53,     0x148727c46ec7657c,     0x7dc83f5051801748
};

const int64_t DataSet_1_64i::outputs::MMAXV[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x76f206cd010959de,
    0x5d793f3374314a60,     0x5145261b76cf6201,     0x67b0233f63680061,     0x115e106678b9182f,
    0x566b140f478f277a,     0x72396eb6079728da,     0x7c9229567ff12759,     0x662919af46d76ce4,
    0x28a378d62cd50afd,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x7dc83f5051801748
};

const int64_t DataSet_1_64i::outputs::MAXS[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x57fc35ae5c75353e,
    0x5d793f3374314a60,     0x1d4a455947367e2e,     0x638814e41c0536b5,     0x115e106678b9182f,
    0x566b140f478f277a,     0x19674bc524a570c8,     0x540a2f7739020697,     0x662919af46d76ce4,
    0x24232c674b04449b,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x7dc83f5051801748
};

const int64_t DataSet_1_64i::outputs::MMAXS[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x57fc35ae5c75353e,
    0x5d793f3374314a60,     0x1d4a455947367e2e,     0x638814e41c0536b5,     0x115e106678b9182f,
    0x566b140f478f277a,     0x19674bc524a570c8,     0x540a2f7739020697,     0x662919af46d76ce4,
    0x24232c674b04449b,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x7dc83f5051801748
};

const int64_t DataSet_1_64i::outputs::MINV[16] = {
    0x0e550ac1262a6992,     0x0738595b30c43ffe,     0x151d03ce4a3d695b,     0x57fc35ae5c75353e,
    0x5d793f3374314a60,     0x1d4a455947367e2e,     0x638814e41c0536b5,     0x115e106678b9182f,
    0x13b52c9e15871bca,     0x19674bc524a570c8,     0x540a2f7739020697,     0x29a70a9c7bd148eb,
    0x24232c674b04449b,     0x0a48668e0cfa4af1,     0x10c6192b0d300338,     0x744d2c7a5dc8654d
};

const int64_t DataSet_1_64i::outputs::MMINV[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x57fc35ae5c75353e,
    0x5d793f3374314a60,     0x1d4a455947367e2e,     0x638814e41c0536b5,     0x115e106678b9182f,
    0x566b140f478f277a,     0x19674bc524a570c8,     0x540a2f7739020697,     0x662919af46d76ce4,
    0x24232c674b04449b,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x744d2c7a5dc8654d
};

const int64_t DataSet_1_64i::outputs::MINS[16] = {
    0xffffffffef9ce763,     0xffffffffef9ce763,     0xffffffffef9ce763,     0xffffffffef9ce763,
    0xffffffffef9ce763,     0xffffffffef9ce763,     0xffffffffef9ce763,     0xffffffffef9ce763,
    0xffffffffef9ce763,     0xffffffffef9ce763,     0xffffffffef9ce763,     0xffffffffef9ce763,
    0xffffffffef9ce763,     0xffffffffef9ce763,     0xffffffffef9ce763,     0xffffffffef9ce763
};

const int64_t DataSet_1_64i::outputs::MMINS[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0xffffffffef9ce763,
    0x5d793f3374314a60,     0xffffffffef9ce763,     0xffffffffef9ce763,     0x115e106678b9182f,
    0x566b140f478f277a,     0xffffffffef9ce763,     0xffffffffef9ce763,     0x662919af46d76ce4,
    0xffffffffef9ce763,     0x247b781e57de4b53,     0x10c6192b0d300338,     0xffffffffef9ce763
};

const int64_t DataSet_1_64i::outputs::HMAX[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x57fc35ae5c75353e,
    0x5d793f3374314a60,     0x5d793f3374314a60,     0x638814e41c0536b5,     0x638814e41c0536b5,
    0x638814e41c0536b5,     0x638814e41c0536b5,     0x638814e41c0536b5,     0x662919af46d76ce4,
    0x662919af46d76ce4,     0x662919af46d76ce4,     0x662919af46d76ce4,     0x7dc83f5051801748
};

const int64_t DataSet_1_64i::outputs::MHMAX[16] = {
    0x8000000000000000,     0x8000000000000000,     0x8000000000000000,     0x57fc35ae5c75353e,
    0x57fc35ae5c75353e,     0x57fc35ae5c75353e,     0x638814e41c0536b5,     0x638814e41c0536b5,
    0x638814e41c0536b5,     0x638814e41c0536b5,     0x638814e41c0536b5,     0x638814e41c0536b5,
    0x638814e41c0536b5,     0x638814e41c0536b5,     0x638814e41c0536b5,     0x7dc83f5051801748
};

const int64_t DataSet_1_64i::outputs::HMIN[16] = {
    0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,
    0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,
    0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,
    0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff
};

const int64_t DataSet_1_64i::outputs::MHMIN[16] = {
    0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,
    0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,
    0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,
    0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff
};

const int64_t DataSet_1_64i::outputs::BANDV[16] = {
    0x0e05084002024180,     0x07181911208034ba,     0x0508000c48156011,     0x56f0048c0001111e,
    0x4969163134014a60,     0x1140041946066200,     0x6380002400000021,     0x0150106230390823,
    0x1221040e0587034a,     0x10214a84048520c8,     0x5402295639000611,     0x2021088c42d148e0,
    0x2023284608040099,     0x0048600e04da4a51,     0x108601000c000138,     0x74482c5051800548
};

const int64_t DataSet_1_64i::outputs::MBANDV[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x56f0048c0001111e,
    0x5d793f3374314a60,     0x1140041946066200,     0x6380002400000021,     0x115e106678b9182f,
    0x566b140f478f277a,     0x10214a84048520c8,     0x5402295639000611,     0x662919af46d76ce4,
    0x2023284608040099,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x74482c5051800548
};

const int64_t DataSet_1_64i::outputs::BANDS[16] = {
    0x0e550ac126086102,     0x475a3b3568806422,     0x47a80c3c49946421,     0x57fc35ae4c142522,
    0x5d793f3364104260,     0x1d4a455947146622,     0x638814e40c042621,     0x115e106668980023,
    0x566b140f478c2762,     0x19674bc524846040,     0x540a2f7729000603,     0x662919af46946460,
    0x24232c674b044403,     0x247b781e479c4343,     0x10c6192b0d100320,     0x7dc83f5041800740
};

const int64_t DataSet_1_64i::outputs::MBANDS[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x57fc35ae4c142522,
    0x5d793f3374314a60,     0x1d4a455947146622,     0x638814e40c042621,     0x115e106678b9182f,
    0x566b140f478f277a,     0x19674bc524846040,     0x540a2f7729000603,     0x662919af46d76ce4,
    0x24232c674b044403,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x7dc83f5041800740
};

const int64_t DataSet_1_64i::outputs::BORV[16] = {
    0x0efd7ae1266f7b92,     0x477a7b7f78c47ffe,     0x57bd0ffe5bff6dff,     0x77fe37ef5d7d7dfe,
    0x7dff7f7f75777b7e,     0x5d4f675b77ff7e2f,     0x67b837ff7f6d36f5,     0x3f5e76667dbb5d7f,
    0x57ff3c9f578f3ffa,     0x7b7f6ff727b778da,     0x7c9a2f777ff327df,     0x6faf1bbf7fd76cef,
    0x2ca37cf76fd54eff,     0x2e7b7e9e5ffe4bf3,     0x14c73fef6ff7677c,     0x7dcd3f7a5dc8774d
};

const int64_t DataSet_1_64i::outputs::MBORV[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x77fe37ef5d7d7dfe,
    0x5d793f3374314a60,     0x5d4f675b77ff7e2f,     0x67b837ff7f6d36f5,     0x115e106678b9182f,
    0x566b140f478f277a,     0x7b7f6ff727b778da,     0x7c9a2f777ff327df,     0x662919af46d76ce4,
    0x2ca37cf76fd54eff,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x7dcd3f7a5dc8774d
};

const int64_t DataSet_1_64i::outputs::BORS[16] = {
    0xffffffffefbeeff3,     0xffffffffef9cf7fb,     0xffffffffffdfe7f7,     0xfffffffffffdf77f,
    0xffffffffffbdef63,     0xffffffffefbeff6f,     0xffffffffff9df7f7,     0xffffffffffbdff6f,
    0xffffffffef9fe77b,     0xffffffffefbdf7eb,     0xffffffffff9ee7f7,     0xffffffffefdfefe7,
    0xffffffffef9ce7fb,     0xffffffffffdeef73,     0xffffffffefbce77b,     0xffffffffff9cf76b
};

const int64_t DataSet_1_64i::outputs::MBORS[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0xfffffffffffdf77f,
    0x5d793f3374314a60,     0xffffffffefbeff6f,     0xffffffffff9df7f7,     0x115e106678b9182f,
    0x566b140f478f277a,     0xffffffffefbdf7eb,     0xffffffffff9ee7f7,     0x662919af46d76ce4,
    0xffffffffef9ce7fb,     0x247b781e57de4b53,     0x10c6192b0d300338,     0xffffffffff9cf76b
};

const int64_t DataSet_1_64i::outputs::BXORV[16] = {
    0x00f872a1246d3a12,     0x4062626e58444b44,     0x52b50ff213ea0dee,     0x210e33635d7c6ce0,
    0x3496694e4176311e,     0x4c0f634231f91c2f,     0x043837db7f6d36d4,     0x3e0e66044d82555c,
    0x45de389152083cb0,     0x6b5e257323325812,     0x2898062146f321ce,     0x4f8e13333d06240f,
    0x0c8054b167d14e66,     0x2e331e905b2401a2,     0x04413eef63f76644,     0x0985132a0c487205
};

const int64_t DataSet_1_64i::outputs::MBXORV[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x210e33635d7c6ce0,
    0x5d793f3374314a60,     0x4c0f634231f91c2f,     0x043837db7f6d36d4,     0x115e106678b9182f,
    0x566b140f478f277a,     0x6b5e257323325812,     0x2898062146f321ce,     0x662919af46d76ce4,
    0x0c8054b167d14e66,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x0985132a0c487205
};

const int64_t DataSet_1_64i::outputs::BXORS[16] = {
    0xf1aaf53ec9b68ef1,     0xb8a5c4ca871c93d9,     0xb857f3c3b64b83d6,     0xa803ca51b3e9d25d,
    0xa286c0cc9badad03,     0xe2b5baa6a8aa994d,     0x9c77eb1bf399d1d6,     0xeea1ef999725ff4c,
    0xa994ebf0a813c019,     0xe698b43acb3997ab,     0xabf5d088d69ee1f4,     0x99d6e650a94b8b87,
    0xdbdcd398a498a3f8,     0xdb8487e1b842ac30,     0xef39e6d4e2ace45b,     0x8237c0afbe1cf02b
};

const int64_t DataSet_1_64i::outputs::MBXORS[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0xa803ca51b3e9d25d,
    0x5d793f3374314a60,     0xe2b5baa6a8aa994d,     0x9c77eb1bf399d1d6,     0x115e106678b9182f,
    0x566b140f478f277a,     0xe698b43acb3997ab,     0xabf5d088d69ee1f4,     0x662919af46d76ce4,
    0xdbdcd398a498a3f8,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x8237c0afbe1cf02b
};

const int64_t DataSet_1_64i::outputs::BNOT[16] = {
    0xf1aaf53ed9d5966d,     0xb8a5c4ca977f8b45,     0xb857f3c3a6289b4a,     0xa803ca51a38acac1,
    0xa286c0cc8bceb59f,     0xe2b5baa6b8c981d1,     0x9c77eb1be3fac94a,     0xeea1ef998746e7d0,
    0xa994ebf0b870d885,     0xe698b43adb5a8f37,     0xabf5d088c6fdf968,     0x99d6e650b928931b,
    0xdbdcd398b4fbbb64,     0xdb8487e1a821b4ac,     0xef39e6d4f2cffcc7,     0x8237c0afae7fe8b7
};

const int64_t DataSet_1_64i::outputs::MBNOT[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0xa803ca51a38acac1,
    0x5d793f3374314a60,     0xe2b5baa6b8c981d1,     0x9c77eb1be3fac94a,     0x115e106678b9182f,
    0x566b140f478f277a,     0xe698b43adb5a8f37,     0xabf5d088c6fdf968,     0x662919af46d76ce4,
    0xdbdcd398b4fbbb64,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x8237c0afae7fe8b7
};

const int64_t DataSet_1_64i::outputs::HBAND[16] = {
    0x0e550ac1262a6992,     0x06500a0120006092,     0x0600080000006090,     0x0600000000002010,
    0x0400000000000000,     0x0400000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000
};

const int64_t DataSet_1_64i::outputs::MHBAND[16] = {
    0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,     0x57fc35ae5c75353e,
    0x57fc35ae5c75353e,     0x154805084434342e,     0x0108040004043424,     0x0108040004043424,
    0x0108040004043424,     0x0100000004043000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000
};

const int64_t DataSet_1_64i::outputs::HBANDS[16] = {
    0x0e550ac126086102,     0x06500a0120006002,     0x0600080000006000,     0x0600000000002000,
    0x0400000000000000,     0x0400000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000
};

const int64_t DataSet_1_64i::outputs::MHBANDS[16] = {
    0xffffffffef9ce763,     0xffffffffef9ce763,     0xffffffffef9ce763,     0x57fc35ae4c142522,
    0x57fc35ae4c142522,     0x1548050844142422,     0x0108040004042420,     0x0108040004042420,
    0x0108040004042420,     0x0100000004042000,     0x0000000000000000,     0x0000000000000000,
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x0000000000000000
};

const int64_t DataSet_1_64i::outputs::HBOR[16] = {
    0x0e550ac1262a6992,     0x4f5f3bf56eaa7dba,     0x4fff3ffd7fff7dbf,     0x5fff3fff7fff7dbf,
    0x5fff3fff7fff7fff,     0x5fff7fff7fff7fff,     0x7fff7fff7fff7fff,     0x7fff7fff7fff7fff,
    0x7fff7fff7fff7fff,     0x7fff7fff7fff7fff,     0x7fff7fff7fff7fff,     0x7fff7fff7fff7fff,
    0x7fff7fff7fff7fff,     0x7fff7fff7fff7fff,     0x7fff7fff7fff7fff,     0x7fff7fff7fff7fff
};

const int64_t DataSet_1_64i::outputs::MHBOR[16] = {
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x57fc35ae5c75353e,
    0x57fc35ae5c75353e,     0x5ffe75ff5f777f3e,     0x7ffe75ff5f777fbf,     0x7ffe75ff5f777fbf,
    0x7ffe75ff5f777fbf,     0x7fff7fff7ff77fff,     0x7fff7fff7ff77fff,     0x7fff7fff7ff77fff,
    0x7fff7fff7ff77fff,     0x7fff7fff7ff77fff,     0x7fff7fff7ff77fff,     0x7fff7fff7ff77fff
};

const int64_t DataSet_1_64i::outputs::HBORS[16] = {
    0xffffffffefbeeff3,     0xffffffffefbefffb,     0xffffffffffffffff,     0xffffffffffffffff,
    0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,
    0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,
    0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff
};

const int64_t DataSet_1_64i::outputs::MHBORS[16] = {
    0xffffffffef9ce763,     0xffffffffef9ce763,     0xffffffffef9ce763,     0xfffffffffffdf77f,
    0xfffffffffffdf77f,     0xffffffffffffff7f,     0xffffffffffffffff,     0xffffffffffffffff,
    0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,
    0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff,     0xffffffffffffffff
};

const int64_t DataSet_1_64i::outputs::HBXOR[16] = {
    0x0e550ac1262a6992,     0x490f31f44eaa1d28,     0x0ea73dc8177d799d,     0x595b08664b084ca3,
    0x042237553f3906c3,     0x1968720c780f78ed,     0x7ae066e8640a4e58,     0x6bbe768e1cb35677,
    0x3dd562815b3c710d,     0x24b229447f9901c5,     0x70b80633469b0752,     0x16911f9c004c6bb6,
    0x32b233fb4b482f2d,     0x16c94be51c96647e,     0x060f52ce11a66746,     0x7bc76d9e4026700e
};

const int64_t DataSet_1_64i::outputs::MHBXOR[16] = {
    0x0000000000000000,     0x0000000000000000,     0x0000000000000000,     0x57fc35ae5c75353e,
    0x57fc35ae5c75353e,     0x4ab670f71b434b10,     0x293e641307467da5,     0x293e641307467da5,
    0x293e641307467da5,     0x30592fd623e30d6d,     0x645300a11ae10bfa,     0x645300a11ae10bfa,
    0x40702cc651e54f61,     0x40702cc651e54f61,     0x40702cc651e54f61,     0x3db8139600655829
};

const int64_t DataSet_1_64i::outputs::HBXORS[16] = {
    0xf1aaf53ec9b68ef1,     0xb6f0ce0ba136fa4b,     0xf158c237f8e19efe,     0xa6a4f799a494abc0,
    0xfbddc8aad0a5e1a0,     0xe6978df397939f8e,     0x851f99178b96a93b,     0x94418971f32fb114,
    0xc22a9d7eb4a0966e,     0xdb4dd6bb9005e6a6,     0x8f47f9cca907e031,     0xe96ee063efd08cd5,
    0xcd4dcc04a4d4c84e,     0xe936b41af30a831d,     0xf9f0ad31fe3a8025,     0x84389261afba976d
};

const int64_t DataSet_1_64i::outputs::MHBXORS[16] = {
    0xffffffffef9ce763,     0xffffffffef9ce763,     0xffffffffef9ce763,     0xa803ca51b3e9d25d,
    0xa803ca51b3e9d25d,     0xb5498f08f4dfac73,     0xd6c19bece8da9ac6,     0xd6c19bece8da9ac6,
    0xd6c19bece8da9ac6,     0xcfa6d029cc7fea0e,     0x9bacff5ef57dec99,     0x9bacff5ef57dec99,
    0xbf8fd339be79a802,     0xbf8fd339be79a802,     0xbf8fd339be79a802,     0xc247ec69eff9bf4a
};

const uint64_t DataSet_1_64i::outputs::LSHV[16] = {
    0x5824c54d32400000,     0xab4403a5d0000000,     0x80c3c59d764b5000,     0xd72e3a9a9f000000,
    0xfccdd0c529800000,     0xa455947367e2e000,     0x8380a6d6a0000000,     0x9e2e460bc0000000,
    0xac503d1e3c9de800,     0xe29252b864000000,     0x28bddce4081a5c00,     0xd7a36bb672000000,
    0x232c674b04449b00,     0x3dbc0f2bef25a980,     0x31864ac34c00ce00,     0xd4146005d2000000
};

const uint64_t DataSet_1_64i::outputs::MLSHV[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0xd72e3a9a9f000000,
    0x5d793f3374314a60,     0xa455947367e2e000,     0x8380a6d6a0000000,     0x115e106678b9182f,
    0x566b140f478f277a,     0xe29252b864000000,     0x28bddce4081a5c00,     0x662919af46d76ce4,
    0x232c674b04449b00,     0x247b781e57de4b53,     0x10c6192b0d300338,     0xd4146005d2000000
};

const uint64_t DataSet_1_64i::outputs::LSHS[16] = {
    0x0931534c90000000,     0xab4403a5d0000000,     0xe2cebb25a8000000,     0x72e3a9a9f0000000,
    0x9ba18a5300000000,     0xca39b3f170000000,     0x20e029b5a8000000,     0x33c5c8c178000000,
    0x7a3c793bd0000000,     0x29252b8640000000,     0xb9c81034b8000000,     0x7a36bb6720000000,
    0x3a582224d8000000,     0xf2bef25a98000000,     0x58698019c0000000,     0x828c00ba40000000
};

const uint64_t DataSet_1_64i::outputs::MLSHS[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x72e3a9a9f0000000,
    0x5d793f3374314a60,     0xca39b3f170000000,     0x20e029b5a8000000,     0x115e106678b9182f,
    0x566b140f478f277a,     0x29252b8640000000,     0xb9c81034b8000000,     0x662919af46d76ce4,
    0x3a582224d8000000,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x828c00ba40000000
};

const uint64_t DataSet_1_64i::outputs::RSHV[16] = {
    0x00000072a8560931,     0x00000008eb4766ad,     0x00047a80c3c59d76,     0x000000aff86b5cb8,
    0x0000175e4fccdd0c,     0x0001d4a455947367,     0x000000031c40a720,     0x0000000045784199,
    0x00159ac503d1e3c9,     0x00000032ce978a49,     0x0015028bddce4081,     0x000000cc52335e8d,
    0x0024232c674b0444,     0x0048f6f03cafbc96,     0x00431864ac34c00c,     0x000001f720fd4146
};

const uint64_t DataSet_1_64i::outputs::MRSHV[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x000000aff86b5cb8,
    0x5d793f3374314a60,     0x0001d4a455947367,     0x000000031c40a720,     0x115e106678b9182f,
    0x566b140f478f277a,     0x00000032ce978a49,     0x0015028bddce4081,     0x662919af46d76ce4,
    0x0024232c674b0444,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x000001f720fd4146
};

const uint64_t DataSet_1_64i::outputs::RSHS[16] = {
    0x00000001caa15824,     0x00000008eb4766ad,     0x00000008f501878b,     0x0000000aff86b5cb,
    0x0000000baf27e66e,     0x00000003a948ab28,     0x0000000c71029c83,     0x000000022bc20ccf,
    0x0000000acd6281e8,     0x000000032ce978a4,     0x0000000a8145eee7,     0x0000000cc52335e8,
    0x0000000484658ce9,     0x000000048f6f03ca,     0x0000000218c32561,     0x0000000fb907ea0a
};

const uint64_t DataSet_1_64i::outputs::MRSHS[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x0000000aff86b5cb,
    0x5d793f3374314a60,     0x00000003a948ab28,     0x0000000c71029c83,     0x115e106678b9182f,
    0x566b140f478f277a,     0x000000032ce978a4,     0x0000000a8145eee7,     0x662919af46d76ce4,
    0x0000000484658ce9,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x0000000fb907ea0a
};

const uint64_t DataSet_1_64i::outputs::ROLV[16] = {
    0x5824c54d3241caa1,     0xab4403a5d23ad1d9,     0x80c3c59d764b547a,     0xd72e3a9a9f2bfe1a,
    0xfccdd0c5298175e4,     0xa455947367e2e1d4,     0x8380a6d6ac71029c,     0x9e2e460bc4578419,
    0xac503d1e3c9de959,     0xe29252b8640cb3a5,     0x28bddce4081a5d50,     0xd7a36bb67233148c,
    0x232c674b04449b24,     0x3dbc0f2bef25a992,     0x31864ac34c00ce04,     0xd4146005d21f720f
};

const uint64_t DataSet_1_64i::outputs::MROLV[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0xd72e3a9a9f2bfe1a,
    0x5d793f3374314a60,     0xa455947367e2e1d4,     0x8380a6d6ac71029c,     0x115e106678b9182f,
    0x566b140f478f277a,     0xe29252b8640cb3a5,     0x28bddce4081a5d50,     0x662919af46d76ce4,
    0x232c674b04449b24,     0x247b781e57de4b53,     0x10c6192b0d300338,     0xd4146005d21f720f
};

const uint64_t DataSet_1_64i::outputs::ROLS[16] = {
    0x0931534c9072a856,     0xab4403a5d23ad1d9,     0xe2cebb25aa3d4061,     0x72e3a9a9f2bfe1ad,
    0x9ba18a5302ebc9f9,     0xca39b3f170ea522a,     0x20e029b5ab1c40a7,     0x33c5c8c1788af083,
    0x7a3c793bd2b358a0,     0x29252b8640cb3a5e,     0xb9c81034baa0517b,     0x7a36bb67233148cd,
    0x3a582224d9211963,     0xf2bef25a9923dbc0,     0x58698019c08630c9,     0x828c00ba43ee41fa
};

const uint64_t DataSet_1_64i::outputs::MROLS[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x72e3a9a9f2bfe1ad,
    0x5d793f3374314a60,     0xca39b3f170ea522a,     0x20e029b5ab1c40a7,     0x115e106678b9182f,
    0x566b140f478f277a,     0x29252b8640cb3a5e,     0xb9c81034baa0517b,     0x662919af46d76ce4,
    0x3a582224d9211963,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x828c00ba43ee41fa
};

const uint64_t DataSet_1_64i::outputs::RORV[16] = {
    0x534c9072a8560931,     0x100e9748eb4766ad,     0x4b547a80c3c59d76,     0xea6a7caff86b5cb8,
    0x5298175e4fccdd0c,     0xe2e1d4a455947367,     0xe029b5ab1c40a720,     0xe2e460bc45784199,
    0xde959ac503d1e3c9,     0x4ae19032ce978a49,     0xa5d5028bddce4081,     0xaed9c8cc52335e8d,
    0x9b24232c674b0444,     0xa648f6f03cafbc96,     0xe0431864ac34c00c,     0x005d21f720fd4146
};

const uint64_t DataSet_1_64i::outputs::MRORV[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0xea6a7caff86b5cb8,
    0x5d793f3374314a60,     0xe2e1d4a455947367,     0xe029b5ab1c40a720,     0x115e106678b9182f,
    0x566b140f478f277a,     0x4ae19032ce978a49,     0xa5d5028bddce4081,     0x662919af46d76ce4,
    0x9b24232c674b0444,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x005d21f720fd4146
};

const uint64_t DataSet_1_64i::outputs::RORS[16] = {
    0xc54d3241caa15824,     0x100e9748eb4766ad,     0x3aec96a8f501878b,     0x8ea6a7caff86b5cb,
    0x86294c0baf27e66e,     0xe6cfc5c3a948ab28,     0x80a6d6ac71029c83,     0x172305e22bc20ccf,
    0xf1e4ef4acd6281e8,     0x94ae19032ce978a4,     0x2040d2ea8145eee7,     0xdaed9c8cc52335e8,
    0x6088936484658ce9,     0xfbc96a648f6f03ca,     0xa600670218c32561,     0x3002e90fb907ea0a
};

const uint64_t DataSet_1_64i::outputs::MRORS[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x8ea6a7caff86b5cb,
    0x5d793f3374314a60,     0xe6cfc5c3a948ab28,     0x80a6d6ac71029c83,     0x115e106678b9182f,
    0x566b140f478f277a,     0x94ae19032ce978a4,     0x2040d2ea8145eee7,     0x662919af46d76ce4,
    0x6088936484658ce9,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x3002e90fb907ea0a
};

const int64_t DataSet_1_64i::outputs::NEG[16] = {
    0xf1aaf53ed9d5966e,     0xb8a5c4ca977f8b46,     0xb857f3c3a6289b4b,     0xa803ca51a38acac2,
    0xa286c0cc8bceb5a0,     0xe2b5baa6b8c981d2,     0x9c77eb1be3fac94b,     0xeea1ef998746e7d1,
    0xa994ebf0b870d886,     0xe698b43adb5a8f38,     0xabf5d088c6fdf969,     0x99d6e650b928931c,
    0xdbdcd398b4fbbb65,     0xdb8487e1a821b4ad,     0xef39e6d4f2cffcc8,     0x8237c0afae7fe8b8
};

const int64_t DataSet_1_64i::outputs::MNEG[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0xa803ca51a38acac2,
    0x5d793f3374314a60,     0xe2b5baa6b8c981d2,     0x9c77eb1be3fac94b,     0x115e106678b9182f,
    0x566b140f478f277a,     0xe698b43adb5a8f38,     0xabf5d088c6fdf969,     0x662919af46d76ce4,
    0xdbdcd398b4fbbb65,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x8237c0afae7fe8b8
};

const int64_t DataSet_1_64i::outputs::ABS[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x57fc35ae5c75353e,
    0x5d793f3374314a60,     0x1d4a455947367e2e,     0x638814e41c0536b5,     0x115e106678b9182f,
    0x566b140f478f277a,     0x19674bc524a570c8,     0x540a2f7739020697,     0x662919af46d76ce4,
    0x24232c674b04449b,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x7dc83f5051801748
};

const int64_t DataSet_1_64i::outputs::MABS[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x57fc35ae5c75353e,
    0x5d793f3374314a60,     0x1d4a455947367e2e,     0x638814e41c0536b5,     0x115e106678b9182f,
    0x566b140f478f277a,     0x19674bc524a570c8,     0x540a2f7739020697,     0x662919af46d76ce4,
    0x24232c674b04449b,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x7dc83f5051801748
};

const uint64_t DataSet_1_64i::outputs::ITOU[16] = {
    0x0e550ac1262a6992,     0x475a3b35688074ba,     0x47a80c3c59d764b5,     0x57fc35ae5c75353e,
    0x5d793f3374314a60,     0x1d4a455947367e2e,     0x638814e41c0536b5,     0x115e106678b9182f,
    0x566b140f478f277a,     0x19674bc524a570c8,     0x540a2f7739020697,     0x662919af46d76ce4,
    0x24232c674b04449b,     0x247b781e57de4b53,     0x10c6192b0d300338,     0x7dc83f5051801748
};


const double DataSet_1_64i::outputs::ITOF[16] = {
    -1489706112.000000000f, -1949446144.000000000f, -368683072.000000000f,  1400938496.000000000f,
    -1302030976.000000000f, 1882764288.000000000f,  2345500.000000000f,     1205584896.000000000f,
    447369856.000000000f,   2002189952.000000000f,  1814442624.000000000f,  -1252331392.000000000f,
    -991493696.000000000f,  1779725184.000000000f,  620782208.000000000f,   841973248.000000000
};




const double DataSet_1_64f::inputs::inputA[16] = {
    -1742.759521484375f,    674.611816406250f,      3355.357421875000f,     1645.405273437500f,
    4394.818359375000f,     -1560.869140625000f,    -1027.100341796875f,    -4268.776367187500f,
    -3708.151367187500f,    -636.768554687500f,     -893.734375000000f,     -1646.626220703125f,
    4145.176757812500f,     4291.054687500000f,     1929.837890625000f,     -2136.753417968750f
};

const double DataSet_1_64f::inputs::inputB[16] = {
    1177.861816406250f,     -3868.068359375000f,    -1390.881103515625f,    3157.292480468750f,
    -3422.803466796875f,    1116.214355468750f,     128.024902343750f,      -873.287109375000f,
    -2733.695556640625f,    2728.202148437500f,     -4162.877441406250f,    -4380.474121093750f,
    1163.518066406250f,     -1805.627685546875f,    -3889.431396484375f,    347.758300781250f
};

const double DataSet_1_64f::inputs::inputC[16] = {
    -1323.129882812500f,    310.525878906250f,      1684.774414062500f,     -4822.687500000000f,
    -2467.879394531250f,    -65.157226562500f,      -4566.942382812500f,    2491.988769531250f,
    631.275390625000f,      3341.013671875000f,     3608.355468750000f,     3277.840820312500f,
    876.338867187500f,      -807.977539062500f,     3113.345703125000f,     3137.760742187500f
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


const double DataSet_1_64f::inputs::scalarA = 255.897461f;

const bool DataSet_1_64f::inputs::maskA[16] = {
    true,   false,  false,  true,   // 4
    false,  true,   true,   false,  // 8

    false,  true,   true,   false,
    true,   false,  false,  true   // 16
};

const double DataSet_1_64f::outputs::ADDV[16] = {
    -564.897705078f,        -3193.456542969f,       1964.476318359f,        4802.697753906f,
    972.014892578f,         -444.654785156f,        -899.075439453f,        -5142.063476563f,
    -6441.846679688f,       2091.433593750f,        -5056.611816406f,       -6027.100585938f,
    5308.694824219f,        2485.427001953f,        -1959.593505859f,       -1788.995117188f
};

const double DataSet_1_64f::outputs::MADDV[16] = {
    -1742.759521484f,       -3193.456542969f,       1964.476318359f,        1645.405273438f,
    972.014892578f,         -1560.869140625f,       -1027.100341797f,       -5142.063476563f,
    -6441.846679688f,       -636.768554688f,        -893.734375000f,        -6027.100585938f,
    4145.176757813f,        2485.427001953f,        -1959.593505859f,       -2136.753417969f
};

const double DataSet_1_64f::outputs::ADDS[16] = {
    -1486.862060547f,       930.509277344f,         3611.254882813f,        1901.302734375f,
    4650.715820313f,        -1304.971679688f,       -771.202880859f,        -4012.878906250f,
    -3452.253906250f,       -380.871093750f,        -637.836914063f,        -1390.728759766f,
    4401.074218750f,        4546.952148438f,        2185.735351563f,        -1880.855957031f
};

const double DataSet_1_64f::outputs::MADDS[16] = {
    -1742.759521484f,       930.509277344f,         3611.254882813f,        1645.405273438f,
    4650.715820313f,        -1560.869140625f,       -1027.100341797f,       -4012.878906250f,
    -3452.253906250f,       -636.768554688f,        -893.734375000f,        -1390.728759766f,
    4145.176757813f,        4546.952148438f,        2185.735351563f,        -2136.753417969f
};

const double DataSet_1_64f::outputs::POSTPREFINC[16] = {
    -1741.759521484f,       675.611816406f,         3356.357421875f,        1646.405273438f,
    4395.818359375f,        -1559.869140625f,       -1026.100341797f,       -4267.776367188f,
    -3707.151367188f,       -635.768554688f,        -892.734375000f,        -1645.626220703f,
    4146.176757813f,        4292.054687500f,        1930.837890625f,        -2135.753417969f
};

const double DataSet_1_64f::outputs::MPOSTPREFINC[16] = {
    -1742.759521484f,       675.611816406f,         3356.357421875f,        1645.405273438f,
    4395.818359375f,        -1560.869140625f,       -1027.100341797f,       -4267.776367188f,
    -3707.151367188f,       -636.768554688f,        -893.734375000f,        -1645.626220703f,
    4145.176757813f,        4292.054687500f,        1930.837890625f,        -2136.753417969f
};

const double DataSet_1_64f::outputs::SUBV[16] = {
    -2920.621337891f,       4542.680175781f,        4746.238281250f,        -1511.887207031f,
    7817.622070313f,        -2677.083496094f,       -1155.125244141f,       -3395.489257813f,
    -974.455810547f,        -3364.970703125f,       3269.143066406f,        2733.847900391f,
    2981.658691406f,        6096.682617188f,        5819.269531250f,        -2484.511718750f
};

const double DataSet_1_64f::outputs::MSUBV[16] = {
    -1742.759521484f,       4542.680175781f,        4746.238281250f,        1645.405273438f,
    7817.622070313f,        -1560.869140625f,       -1027.100341797f,       -3395.489257813f,
    -974.455810547f,        -636.768554688f,        -893.734375000f,        2733.847900391f,
    4145.176757813f,        6096.682617188f,        5819.269531250f,        -2136.753417969f
};

const double DataSet_1_64f::outputs::SUBS[16] = {
    -1998.656982422f,       418.714355469f,         3099.459960938f,        1389.507812500f,
    4138.920898438f,        -1816.766601563f,       -1282.997802734f,       -4524.673828125f,
    -3964.048828125f,       -892.666015625f,        -1149.631835938f,       -1902.523681641f,
    3889.279296875f,        4035.157226563f,        1673.940429688f,        -2392.650878906f
};

const double DataSet_1_64f::outputs::MSUBS[16] = {
    -1742.759521484f,       418.714355469f,         3099.459960938f,        1645.405273438f,
    4138.920898438f,        -1560.869140625f,       -1027.100341797f,       -4524.673828125f,
    -3964.048828125f,       -636.768554688f,        -893.734375000f,        -1902.523681641f,
    4145.176757813f,        4035.157226563f,        1673.940429688f,        -2136.753417969f
};

const double DataSet_1_64f::outputs::SUBFROMV[16] = {
    2920.621337891f,        -4542.680175781f,       -4746.238281250f,       1511.887207031f,
    -7817.622070313f,       2677.083496094f,        1155.125244141f,        3395.489257813f,
    974.455810547f,         3364.970703125f,        -3269.143066406f,       -2733.847900391f,
    -2981.658691406f,       -6096.682617188f,       -5819.269531250f,       2484.511718750f
};

const double DataSet_1_64f::outputs::MSUBFROMV[16] = {
    1177.861816406f,        -4542.680175781f,       -4746.238281250f,       3157.292480469f,
    -7817.622070313f,       1116.214355469f,        128.024902344f,         3395.489257813f,
    974.455810547f,         2728.202148438f,        -4162.877441406f,       -2733.847900391f,
    1163.518066406f,        -6096.682617188f,       -5819.269531250f,       347.758300781f
};

const double DataSet_1_64f::outputs::SUBFROMS[16] = {
    1998.656982422f,        -418.714355469f,        -3099.459960938f,       -1389.507812500f,
    -4138.920898438f,       1816.766601563f,        1282.997802734f,        4524.673828125f,
    3964.048828125f,        892.666015625f,         1149.631835938f,        1902.523681641f,
    -3889.279296875f,       -4035.157226563f,       -1673.940429688f,       2392.650878906f
};

const double DataSet_1_64f::outputs::MSUBFROMS[16] = {
    255.897460938f,         -418.714355469f,        -3099.459960938f,       255.897460938f,
    -4138.920898438f,       255.897460938f,         255.897460938f,         4524.673828125f,
    3964.048828125f,        255.897460938f,         255.897460938f,         1902.523681641f,
    255.897460938f,         -4035.157226563f,       -1673.940429688f,       255.897460938f
};

const double DataSet_1_64f::outputs::POSTPREFDEC[16] = {
    -1743.759521484f,       673.611816406f,         3354.357421875f,        1644.405273438f,
    4393.818359375f,        -1561.869140625f,       -1028.100341797f,       -4269.776367188f,
    -3709.151367188f,       -637.768554688f,        -894.734375000f,        -1647.626220703f,
    4144.176757813f,        4290.054687500f,        1928.837890625f,        -2137.753417969f
};

const double DataSet_1_64f::outputs::MPOSTPREFDEC[16] = {
    -1742.759521484f,       673.611816406f,         3354.357421875f,        1645.405273438f,
    4393.818359375f,        -1560.869140625f,       -1027.100341797f,       -4269.776367188f,
    -3709.151367188f,       -636.768554688f,        -893.734375000f,        -1647.626220703f,
    4145.176757813f,        4290.054687500f,        1928.837890625f,        -2136.753417969f
};

const double DataSet_1_64f::outputs::MULV[16] = {
    -2052729.875000000f,    -2609444.500000000f,    -4666903.000000000f,    5195025.500000000f,
    -15042600.000000000f,   -1742264.500000000f,    -131494.421875000f,     3727867.250000000f,
    10136957.000000000f,    -1737233.375000000f,    3720506.750000000f,     7213003.500000000f,
    4822988.000000000f,     -7748047.000000000f,    -7505972.000000000f,    -743073.750000000f
};

const double DataSet_1_64f::outputs::MMULV[16] = {
    -1742.759521484f,       -2609444.500000000f,    -4666903.000000000f,    1645.405273438f,
    -15042600.000000000f,   -1560.869140625f,       -1027.100341797f,       3727867.250000000f,
    10136957.000000000f,    -636.768554688f,        -893.734375000f,        7213003.500000000f,
    4145.176757813f,        -7748047.000000000f,    -7505972.000000000f,    -2136.753417969f
};

const double DataSet_1_64f::outputs::MULS[16] = {
    -445967.750000000f,     172631.453125000f,      858627.437500000f,      421055.031250000f,
    1124622.875000000f,     -399422.437500000f,     -262832.375000000f,     -1092369.000000000f,
    -948906.500000000f,     -162947.453125000f,     -228704.359375000f,     -421367.468750000f,
    1060740.250000000f,     1098070.000000000f,     493840.625000000f,      -546789.750000000f
};

const double DataSet_1_64f::outputs::MMULS[16] = {
    -1742.759521484f,       172631.453125000f,      858627.437500000f,      1645.405273438f,
    1124622.875000000f,     -1560.869140625f,       -1027.100341797f,       -1092369.000000000f,
    -948906.500000000f,     -636.768554688f,        -893.734375000f,        -421367.468750000f,
    4145.176757813f,        1098070.000000000f,     493840.625000000f,      -2136.753417969f
};

const double DataSet_1_64f::outputs::DIVV[16] = {
    -1.479595900f,  -0.174405351f,  -2.412396908f,  0.521144390f,
    -1.283982038f,  -1.398359656f,  -8.022660255f,  4.888170719f,
    1.356460929f,   -0.233402267f,  0.214691490f,   0.375901371f,
    3.562623501f,   -2.376489162f,  -0.496174812f,  -6.144363403f
};

const double DataSet_1_64f::outputs::MDIVV[16] = {
    -1742.759521484f,       -0.174405351f,          -2.412396908f,          1645.405273438f,
    -1.283982038f,          -1560.869140625f,       -1027.100341797f,       4.888170719f,
    1.356460929f,           -636.768554688f,        -893.734375000f,        0.375901371f,
    4145.176757813f,        -2.376489162f,          -0.496174812f,          -2136.753417969f
};

const double DataSet_1_64f::outputs::DIVS[16] = {
    -6.810382366f,  2.636258364f,   13.112116814f,  6.429939747f,
    17.174139023f,  -6.099588394f,  -4.013718605f,  -16.681589127f,
    -14.490770340f, -2.488373756f,  -3.492548704f,  -6.434710979f,
    16.198585510f,  16.768648148f,  7.541450024f,   -8.350037575f
};

const double DataSet_1_64f::outputs::MDIVS[16] = {
    -1742.759521484f,       2.636258364f,           13.112116814f,          1645.405273438f,
    17.174139023f,          -1560.869140625f,       -1027.100341797f,       -16.681589127f,
    -14.490770340f,         -636.768554688f,        -893.734375000f,        -6.434710979f,
    4145.176757813f,        16.768648148f,          7.541450024f,           -2136.753417969f
};

const double DataSet_1_64f::outputs::RCP[16] = {
    -0.000573803f,  0.001482334f,   0.000298031f,   0.000607753f,
    0.000227541f,   -0.000640669f,  -0.000973615f,  -0.000234259f,
    -0.000269676f,  -0.001570429f,  -0.001118901f,  -0.000607302f,
    0.000241244f,   0.000233043f,   0.000518178f,   -0.000468000f
};

const double DataSet_1_64f::outputs::MRCP[16] = {
    -1742.759521484f,       0.001482334f,           0.000298031f,           1645.405273438f,
    0.000227541f,           -1560.869140625f,       -1027.100341797f,       -0.000234259f,
    -0.000269676f,          -636.768554688f,        -893.734375000f,        -0.000607302f,
    4145.176757813f,        0.000233043f,           0.000518178f,           -2136.753417969f
};

const double DataSet_1_64f::outputs::RCPS[16] = {
    -0.146834642f,  0.379325509f,   0.076265335f,   0.155522451f,
    0.058227085f,   -0.163945496f,  -0.249145538f,  -0.059946325f,
    -0.069009446f,  -0.401868880f,  -0.286323845f,  -0.155407131f,
    0.061733786f,   0.059635095f,   0.132600501f,   -0.119759940f
};

const double DataSet_1_64f::outputs::MRCPS[16] = {
    -1742.759521484f,       0.379325509f,           0.076265335f,           1645.405273438f,
    0.058227085f,           -1560.869140625f,       -1027.100341797f,       -0.059946325f,
    -0.069009446f,          -636.768554688f,        -893.734375000f,        -0.155407131f,
    4145.176757813f,        0.059635095f,           0.132600501f,           -2136.753417969f
};

const bool DataSet_1_64f::outputs::CMPEQV[16] = {
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false
};

const bool DataSet_1_64f::outputs::CMPEQS[16] = {
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false
};

const bool DataSet_1_64f::outputs::CMPNEV[16] = {
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true
};

const bool DataSet_1_64f::outputs::CMPNES[16] = {
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true
};

const bool DataSet_1_64f::outputs::CMPGTV[16] = {
    false,  true,   true,   false,  true,   false,  false,  false,
    false,  false,  true,   true,   true,   true,   true,   false
};

const bool DataSet_1_64f::outputs::CMPGTS[16] = {
    false,  true,   true,   true,   true,   false,  false,  false,
    false,  false,  false,  false,  true,   true,   true,   false
};

const bool DataSet_1_64f::outputs::CMPLTV[16] = {
    true,   false,  false,  true,   false,  true,   true,   true,
    true,   true,   false,  false,  false,  false,  false,  true
};

const bool DataSet_1_64f::outputs::CMPLTS[16] = {
    true,   false,  false,  false,  false,  true,   true,   true,
    true,   true,   true,   true,   false,  false,  false,  true
};

const bool DataSet_1_64f::outputs::CMPGEV[16] = {
    false,  true,   true,   false,  true,   false,  false,  false,
    false,  false,  true,   true,   true,   true,   true,   false
};

const bool DataSet_1_64f::outputs::CMPGES[16] = {
    false,  true,   true,   true,   true,   false,  false,  false,
    false,  false,  false,  false,  true,   true,   true,   false
};

const bool DataSet_1_64f::outputs::CMPLEV[16] = {
    true,   false,  false,  true,   false,  true,   true,   true,
    true,   true,   false,  false,  false,  false,  false,  true
};

const bool DataSet_1_64f::outputs::CMPLES[16] = {
    true,   false,  false,  false,  false,  true,   true,   true,
    true,   true,   true,   true,   false,  false,  false,  true
};

const bool DataSet_1_64f::outputs::CMPEV = false;
const bool DataSet_1_64f::outputs::CMPES = false;

const double DataSet_1_64f::outputs::HADD[16] = {
    8967.709960938f,        9642.322265625f,        12997.679687500f,       14643.084960938f,
    19037.902343750f,       17477.033203125f,       16449.933593750f,       12181.157226563f,
    8473.005859375f,        7836.237304688f,        6942.502929688f,        5295.876953125f,
    9441.053710938f,        13732.108398438f,       15661.946289063f,       13525.193359375f
};

const double DataSet_1_64f::outputs::MHADD[16] = {
    8570.207031250f,        9244.818359375f,        12600.175781250f,       12600.175781250f,
    16994.994140625f,       16994.994140625f,       16994.994140625f,       12726.217773438f,
    9018.066406250f,        9018.066406250f,        9018.066406250f,        7371.440429688f,
    7371.440429688f,        11662.495117188f,       13592.333007813f,       13592.333007813f
};

const double DataSet_1_64f::outputs::HMUL[16] = {
    -std::numeric_limits<double>::infinity(),        -std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),        -std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),        std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),        std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),        std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),        std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity(),         std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity(),         -std::numeric_limits<double>::infinity()
};

const double DataSet_1_64f::outputs::MHMUL[16] = {
    3940816271106599700000000.000000000f,           2658521144070606200000000000.000000000f,
    8920288855932268100000000000000.000000000f,     8920288855932268100000000000000.000000000f,
    39203048025888244000000000000000000.000000000f, 39203048025888244000000000000000000.000000000f,
    39203048025888244000000000000000000.000000000f, -167349046107201740000000000000000000000.000000000f,
    std::numeric_limits<double>::infinity(),         std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity(),         -std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),        -std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),        -std::numeric_limits<double>::infinity()
};

const double DataSet_1_64f::outputs::FMULADDV[16] = {
    -2054053.000000000f,    -2609134.000000000f,    -4665218.000000000f,    5190203.000000000f,
    -15045068.000000000f,   -1742329.625000000f,    -136061.359375000f,     3730359.250000000f,
    10137588.000000000f,    -1733892.375000000f,    3724115.000000000f,     7216281.500000000f,
    4823864.500000000f,     -7748855.000000000f,    -7502858.500000000f,    -739936.000000000f
};

const double DataSet_1_64f::outputs::MFMULADDV[16] = {
    -1742.759521484f,       -2609134.000000000f,    -4665218.000000000f,    1645.405273438f,
    -15045068.000000000f,   -1560.869140625f,       -1027.100341797f,       3730359.250000000f,
    10137588.000000000f,    -636.768554688f,        -893.734375000f,        7216281.500000000f,
    4145.176757813f,        -7748855.000000000f,    -7502858.500000000f,    -2136.753417969f
};

const double DataSet_1_64f::outputs::FMULSUBV[16] = {
    -2051406.750000000f,    -2609755.000000000f,    -4668588.000000000f,    5199848.000000000f,
    -15040132.000000000f,   -1742199.375000000f,    -126927.476562500f,     3725375.250000000f,
    10136326.000000000f,    -1740574.375000000f,    3716898.500000000f,     7209725.500000000f,
    4822111.500000000f,     -7747239.000000000f,    -7509085.500000000f,    -746211.500000000f
};

const double DataSet_1_64f::outputs::MFMULSUBV[16] = {
    -1742.759521484f,       -2609755.000000000f,    -4668588.000000000f,    1645.405273438f,
    -15040132.000000000f,   -1560.869140625f,       -1027.100341797f,       3725375.250000000f,
    10136326.000000000f,    -636.768554688f,        -893.734375000f,        7209725.500000000f,
    4145.176757813f,        -7747239.000000000f,    -7509085.500000000f,    -2136.753417969f
};

const double DataSet_1_64f::outputs::FADDMULV[16] = {
    747433.062500000f,      -991650.875000000f,     3309699.500000000f,     -23161910.000000000f,
    -2398815.500000000f,    28972.472656250f,       4106025.750000000f,     -12813964.000000000f,
    -4066579.250000000f,    6987508.000000000f,     -18246052.000000000f,   -19755876.000000000f,
    4652215.500000000f,     -2008169.250000000f,    -6100892.000000000f,    -5613438.500000000f
};

const double DataSet_1_64f::outputs::MFADDMULV[16] = {
    -1742.759521484f,       -991650.875000000f,     3309699.500000000f,     1645.405273438f,
    -2398815.500000000f,    -1560.869140625f,       -1027.100341797f,       -12813964.000000000f,
    -4066579.250000000f,    -636.768554688f,        -893.734375000f,        -19755876.000000000f,
    4145.176757813f,        -2008169.250000000f,    -6100892.000000000f,    -2136.753417969f
};

const double DataSet_1_64f::outputs::FSUBMULV[16] = {
    3864361.250000000f,     1410619.750000000f,     7996341.000000000f,     7291359.500000000f,
    -19292948.000000000f,   174431.328125000f,      5275390.500000000f,     -8461521.000000000f,
    -615150.000000000f,     -11242413.000000000f,   11796230.000000000f,    8961118.000000000f,
    2612943.500000000f,     -4925982.500000000f,    18117398.000000000f,    -7795803.500000000f
};

const double DataSet_1_64f::outputs::MFSUBMULV[16] = {
    -1742.759521484f,       1410619.750000000f,     7996341.000000000f,     1645.405273438f,
    -19292948.000000000f,   -1560.869140625f,       -1027.100341797f,       -8461521.000000000f,
    -615150.000000000f,     -636.768554688f,        -893.734375000f,        8961118.000000000f,
    4145.176757813f,        -4925982.500000000f,    18117398.000000000f,    -2136.753417969f
};

const double DataSet_1_64f::outputs::MAXV[16] = {
    1177.861816406f,        674.611816406f,         3355.357421875f,        3157.292480469f,
    4394.818359375f,        1116.214355469f,        128.024902344f,         -873.287109375f,
    -2733.695556641f,       2728.202148438f,        -893.734375000f,        -1646.626220703f,
    4145.176757813f,        4291.054687500f,        1929.837890625f,        347.758300781f
};

const double DataSet_1_64f::outputs::MMAXV[16] = {
    -1742.759521484f,       674.611816406f,         3355.357421875f,        1645.405273438f,
    4394.818359375f,        -1560.869140625f,       -1027.100341797f,       -873.287109375f,
    -2733.695556641f,       -636.768554688f,        -893.734375000f,        -1646.626220703f,
    4145.176757813f,        4291.054687500f,        1929.837890625f,        -2136.753417969f
};

const double DataSet_1_64f::outputs::MAXS[16] = {
    255.897460938f,         674.611816406f,         3355.357421875f,        1645.405273438f,
    4394.818359375f,        255.897460938f,         255.897460938f,         255.897460938f,
    255.897460938f,         255.897460938f,         255.897460938f,         255.897460938f,
    4145.176757813f,        4291.054687500f,        1929.837890625f,        255.897460938f
};

const double DataSet_1_64f::outputs::MMAXS[16] = {
    -1742.759521484f,       674.611816406f,         3355.357421875f,        1645.405273438f,
    4394.818359375f,        -1560.869140625f,       -1027.100341797f,       255.897460938f,
    255.897460938f,         -636.768554688f,        -893.734375000f,        255.897460938f,
    4145.176757813f,        4291.054687500f,        1929.837890625f,        -2136.753417969f
};

const double DataSet_1_64f::outputs::MINV[16] = {
    -1742.759521484f,       -3868.068359375f,       -1390.881103516f,       1645.405273438f,
    -3422.803466797f,       -1560.869140625f,       -1027.100341797f,       -4268.776367188f,
    -3708.151367188f,       -636.768554688f,        -4162.877441406f,       -4380.474121094f,
    1163.518066406f,        -1805.627685547f,       -3889.431396484f,       -2136.753417969f
};

const double DataSet_1_64f::outputs::MMINV[16] = {
    -1742.759521484f,       -3868.068359375f,       -1390.881103516f,       1645.405273438f,
    -3422.803466797f,       -1560.869140625f,       -1027.100341797f,       -4268.776367188f,
    -3708.151367188f,       -636.768554688f,        -893.734375000f,        -4380.474121094f,
    4145.176757813f,        -1805.627685547f,       -3889.431396484f,       -2136.753417969f
};

const double DataSet_1_64f::outputs::MINS[16] = {
    -1742.759521484f,       255.897460938f,         255.897460938f,         255.897460938f,
    255.897460938f,         -1560.869140625f,       -1027.100341797f,       -4268.776367188f,
    -3708.151367188f,       -636.768554688f,        -893.734375000f,        -1646.626220703f,
    255.897460938f,         255.897460938f,         255.897460938f,         -2136.753417969f
};

const double DataSet_1_64f::outputs::MMINS[16] = {
    -1742.759521484f,       255.897460938f,         255.897460938f,         1645.405273438f,
    255.897460938f,         -1560.869140625f,       -1027.100341797f,       -4268.776367188f,
    -3708.151367188f,       -636.768554688f,        -893.734375000f,        -1646.626220703f,
    4145.176757813f,        255.897460938f,         255.897460938f,         -2136.753417969f
};

const double DataSet_1_64f::outputs::HMAX[16] = {
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f
};

const double DataSet_1_64f::outputs::MHMAX[16] = {
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f
};

const double DataSet_1_64f::outputs::HMIN[16] = {
    -3039.185791016f,       -3039.185791016f,       -3039.185791016f,       -3039.185791016f,
    -3039.185791016f,       -3039.185791016f,       -3039.185791016f,       -4268.776367188f,
    -4268.776367188f,       -4268.776367188f,       -4268.776367188f,       -4268.776367188f,
    -4268.776367188f,       -4268.776367188f,       -4268.776367188f,       -4268.776367188f
};

const double DataSet_1_64f::outputs::MHMIN[16] = {
    -2183.446777344f,       -2183.446777344f,       -2183.446777344f,       -2183.446777344f,
    -2183.446777344f,       -2183.446777344f,       -2183.446777344f,       -4268.776367188f,
    -4268.776367188f,       -4268.776367188f,       -4268.776367188f,       -4268.776367188f,
    -4268.776367188f,       -4268.776367188f,       -4268.776367188f,       -4268.776367188f
};

const double DataSet_1_64f::outputs::NEG[16] = {
    1742.759521484f,        -674.611816406f,        -3355.357421875f,       -1645.405273438f,
    -4394.818359375f,       1560.869140625f,        1027.100341797f,        4268.776367188f,
    3708.151367188f,        636.768554688f,         893.734375000f,         1646.626220703f,
    -4145.176757813f,       -4291.054687500f,       -1929.837890625f,       2136.753417969f
};

const double DataSet_1_64f::outputs::MNEG[16] = {
    -1742.759521484f,       -674.611816406f,        -3355.357421875f,       1645.405273438f,
    -4394.818359375f,       -1560.869140625f,       -1027.100341797f,       4268.776367188f,
    3708.151367188f,        -636.768554688f,        -893.734375000f,        1646.626220703f,
    4145.176757813f,        -4291.054687500f,       -1929.837890625f,       -2136.753417969f
};

const double DataSet_1_64f::outputs::ABS[16] = {
    1742.759521484f,        674.611816406f,         3355.357421875f,        1645.405273438f,
    4394.818359375f,        1560.869140625f,        1027.100341797f,        4268.776367188f,
    3708.151367188f,        636.768554688f,         893.734375000f,         1646.626220703f,
    4145.176757813f,        4291.054687500f,        1929.837890625f,        2136.753417969f
};

const double DataSet_1_64f::outputs::MABS[16] = {
    -1742.759521484f,       674.611816406f,         3355.357421875f,        1645.405273438f,
    4394.818359375f,        -1560.869140625f,       -1027.100341797f,       4268.776367188f,
    3708.151367188f,        -636.768554688f,        -893.734375000f,        1646.626220703f,
    4145.176757813f,        4291.054687500f,        1929.837890625f,        -2136.753417969f
};

const double DataSet_1_64f::outputs::SQR[16] = {
    3037210.750000000f,     455101.093750000f,      11258423.000000000f,    2707358.500000000f,
    19314428.000000000f,    2436312.500000000f,     1054935.125000000f,     18222452.000000000f,
    13750387.000000000f,    405474.187500000f,      798761.125000000f,      2711378.000000000f,
    17182490.000000000f,    18413150.000000000f,    3724274.250000000f,     4565715.000000000f
};

const double DataSet_1_64f::outputs::MSQR[16] = {
    -1742.759521484f,       455101.093750000f,      11258423.000000000f,    1645.405273438f,
    19314428.000000000f,    -1560.869140625f,       -1027.100341797f,       18222452.000000000f,
    13750387.000000000f,    -636.768554688f,        -893.734375000f,        2711378.000000000f,
    4145.176757813f,        18413150.000000000f,    3724274.250000000f,     -2136.753417969f
};

// SQRT(ABS(g_Init1_l));
const double DataSet_1_64f::outputs::SQRT[16] = {
    41.746372223f,  25.973289490f,  57.925445557f,  40.563594818f,
    66.293426514f,  39.507835388f,  32.048404694f,  65.335876465f,
    60.894592285f,  25.234273911f,  29.895391464f,  40.578643799f,
    64.383049011f,  65.506141663f,  43.929920197f,  46.225028992f
};

const double DataSet_1_64f::outputs::MSQRT[16] = {
    -1742.759521484f,       25.973289490f,          57.925445557f,          1645.405273438f,
    66.293426514f,          -1560.869140625f,       -1027.100341797f,       65.335876465f,
    60.894592285f,          -636.768554688f,        -893.734375000f,        40.578643799f,
    4145.176757813f,        65.506141663f,          43.929920197f,          -2136.753417969f
};

// POWV
// MPOWV
// POWS
// MPOWS

const double DataSet_1_64f::outputs::ROUND[16] = {
    -1742.000000000f,       675.000000000f,         3355.000000000f,        1645.000000000f,
    4395.000000000f,        -1560.000000000f,       -1026.000000000f,       -4268.000000000f,
    -3707.000000000f,       -636.000000000f,        -893.000000000f,        -1646.000000000f,
    4145.000000000f,        4291.000000000f,        1930.000000000f,        -2136.000000000f
};

const double DataSet_1_64f::outputs::MROUND[16] = {
    -1742.759521484f,       675.000000000f,         3355.000000000f,        1645.405273438f,
    4395.000000000f,        -1560.869140625f,       -1027.100341797f,       -4268.000000000f,
    -3707.000000000f,       -636.768554688f,        -893.734375000f,        -1646.000000000f,
    4145.176757813f,        4291.000000000f,        1930.000000000f,        -2136.753417969f
};

const int64_t DataSet_1_64f::outputs::TRUNC[16] = {
    -1742,  674,    3355,   1645,
    4394,   -1560,  -1027,  -4268,
    -3708,  -636,   -893,   -1646,
    4145,   4291,   1929,   -2136
};

const int64_t DataSet_1_64f::outputs::MTRUNC[16] = {
    0,      674,    3355,   0,
    4394,   0,      0,      -4268,
    -3708,  0,      0,      -1646,
    0,      4291,   1929,   0
};

const double DataSet_1_64f::outputs::FLOOR[16] = {
    -1743.000000000f,       674.000000000f,         3355.000000000f,        1645.000000000f,
    4394.000000000f,        -1561.000000000f,       -1028.000000000f,       -4269.000000000f,
    -3709.000000000f,       -637.000000000f,        -894.000000000f,        -1647.000000000f,
    4145.000000000f,        4291.000000000f,        1929.000000000f,        -2137.000000000f
};

const double DataSet_1_64f::outputs::MFLOOR[16] = {
    -1742.759521484f,       674.000000000f,         3355.000000000f,        1645.405273438f,
    4394.000000000f,        -1560.869140625f,       -1027.100341797f,       -4269.000000000f,
    -3709.000000000f,       -636.768554688f,        -893.734375000f,        -1647.000000000f,
    4145.176757813f,        4291.000000000f,        1929.000000000f,        -2136.753417969f
};

const double DataSet_1_64f::outputs::CEIL[16] = {
    -1742.000000000f,       675.000000000f,         3356.000000000f,        1646.000000000f,
    4395.000000000f,        -1560.000000000f,       -1027.000000000f,       -4268.000000000f,
    -3708.000000000f,       -636.000000000f,        -893.000000000f,        -1646.000000000f,
    4146.000000000f,        4292.000000000f,        1930.000000000f,        -2136.000000000f
};

const double DataSet_1_64f::outputs::MCEIL[16] = {
    -1742.759521484f,       675.000000000f,         3356.000000000f,        1645.405273438f,
    4395.000000000f,        -1560.869140625f,       -1027.100341797f,       -4268.000000000f,
    -3708.000000000f,       -636.768554688f,        -893.734375000f,        -1646.000000000f,
    4145.176757813f,        4292.000000000f,        1930.000000000f,        -2136.753417969f
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
    -0.734141350f,  0.738338947f,   0.136044651f,   -0.709844232f,
    0.266503006f,   -0.481537551f,  -0.199116156f,  -0.603632152f,
    -0.878176212f,  -0.827563822f,  -0.998812675f,  -0.418388397f,
    -0.988050282f,  -0.353095174f,  0.783327758f,   -0.453254938f
};

const double DataSet_1_64f::outputs::MSIN[16] = {
    -1742.759521484f,       0.738338947f,           0.136044651f,           1645.405273438f,
    0.266503006f,           -1560.869140625f,       -1027.100341797f,       -0.603632152f,
    -0.878176212f,          -636.768554688f,        -893.734375000f,        -0.418388397f,
    4145.176757813f,        -0.353095174f,          0.783327758f,           -2136.753417969f
};

const double DataSet_1_64f::outputs::COS[16] = {
    -0.678996623f,  -0.674429834f,  0.990702689f,   0.704358697f,
    -0.963834107f,  -0.876425445f,  -0.979975879f,  -0.797262967f,
    0.478337288f,   -0.561371684f,  0.048715658f,   0.908268213f,
    -0.154131711f,  0.935587406f,   0.621608913f,   0.891380906f
};

const double DataSet_1_64f::outputs::MCOS[16] = {
    -1742.759521484f,       -0.674429834f,          0.990702689f,           1645.405273438f,
    -0.963834107f,          -1560.869140625f,       -1027.100341797f,       -0.797262967f,
    0.478337288f,           -636.768554688f,        -893.734375000f,        0.908268213f,
    4145.176757813f,        0.935587406f,           0.621608913f,           -2136.753417969f
};


const double DataSet_1_64f::outputs::TAN[16] = {
    1.081215024f,   -1.094760180f,  0.137321368f,   -1.007788062f,
    -0.276502997f,  0.549433529f,   0.203184739f,   0.757130504f,
    -1.835893154f,  1.474181652f,   -20.502908707f, -0.460644096f,
    6.410428524f,   -0.377404779f,  1.260161638f,   -0.508486211f
};

const double DataSet_1_64f::outputs::MTAN[16] = {
    -1742.759521484f,       -1.094760180f,          0.137321368f,           1645.405273438f,
    -0.276502997f,          -1560.869140625f,       -1027.100341797f,       0.757130504f,
    -1.835893154f,          -636.768554688f,        -893.734375000f,        -0.460644096f,
    4145.176757813f,        -0.377404779f,          1.260161638f,           -2136.753417969f
};

const double DataSet_1_64f::outputs::CTAN[16] = {
    0.924885392f,   -0.913442075f,  7.282187939f,   -0.992272139f,
    -3.616597414f,  1.820056438f,   4.921629429f,   1.320776224f,
    -0.544694006f,  0.678342462f,   -0.048773568f,  -2.170873404f,
    0.155995816f,   -2.649674892f,  0.793549001f,   -1.966621637f
};

const double DataSet_1_64f::outputs::MCTAN[16] = {
    -1742.759521484f,       -0.913442075f,          7.282187939f,           1645.405273438f,
    -3.616597414f,          -1560.869140625f,       -1027.100341797f,       1.320776224f,
    -0.544694006f,          -636.768554688f,        -893.734375000f,        -2.170873404f,
    4145.176757813f,        -2.649674892f,          0.793549001f,           -2136.753417969f
};

const uint64_t DataSet_1_64f::outputs::FTOU[16] = {
    0xfffff932,     0x000002a2,     0x00000d1b,     0x0000066d,
    0x0000112a,     0xfffff9e8,     0xfffffbfd,     0xffffef54,
    0xfffff184,     0xfffffd84,     0xfffffc83,     0xfffff992,
    0x00001031,     0x000010c3,     0x00000789,     0xfffff7a8
};

const int64_t DataSet_1_64f::outputs::FTOI[16] = {
    0xfffff932,     0x000002a2,     0x00000d1b,     0x0000066d,
    0x0000112a,     0xfffff9e8,     0xfffffbfd,     0xffffef54,
    0xfffff184,     0xfffffd84,     0xfffffc83,     0xfffff992,
    0x00001031,     0x000010c3,     0x00000789,     0xfffff7a8
};
