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
// This piece of code was developed as part of ICE-DIP project at CERN.
//  "ICE-DIP is a European Industrial Doctorate project funded by the European Community's 
//  7th Framework programme Marie Curie Actions under grant PITN-GA-2012-316596".
//

#include "UMEUnitTestDataSets32.h"

#include <limits>

// Below are pre-computed values for unit-tests.
// inputA, inputB, inputC, scalar and mask are used as inputs.
// Other arrays are used as model values used in comparison

const bool DataSet_1_mask::inputs::maskA[128] = {
    false,  true,   false,  false,  true,   false,  false,  false,
    false,  true,   true,   false,  true,   true,   false,  false,
    true,   true,   true,   false,  true,   false,  false,  false,
    true,   false,  true,   false,  true,   true,   true,   true,
    false,  false,  false,  false,  false,  false,  true,   false,
    true,   false,  false,  false,  true,   true,   true,   true,
    false,  false,  false,  true,   true,   false,  false,  true,
    true,   true,   true,   true,   false,  false,  false,  false,
    false,  false,  true,   false,  false,  false,  false,  false,
    true,   false,  true,   true,   true,   false,  false,  false,
    true,   false,  false,  true,   true,   false,  false,  false,
    false,  false,  true,   false,  true,   true,   false,  true,
    true,   true,   true,   true,   false,  false,  true,   true,
    false,  false,  true,   false,  true,   false,  true,   false,
    false,  true,   true,   true,   false,  true,   true,   false,
    false,  true,   false,  false,  false,  true,   true,   true
};

const bool DataSet_1_mask::inputs::maskB[128] = {
    true,   true,   true,   false,  true,   true,   false,  false,
    false,  false,  true,   false,  true,   false,  true,   true,
    true,   true,   false,  false,  true,   false,  true,   true,
    false,  true,   true,   true,   true,   true,   true,   true,
    true,   true,   false,  true,   false,  true,   true,   true,
    false,  false,  false,  true,   true,   false,  false,  true,
    true,   true,   true,   false,  true,   false,  true,   true,
    false,  false,  true,   false,  false,  false,  true,   false,
    false,  false,  true,   false,  false,  false,  false,  true,
    false,  true,   true,   false,  true,   true,   false,  true,
    false,  false,  true,   false,  false,  true,   false,  false,
    false,  true,   true,   false,  false,  true,   true,   true,
    false,  true,   false,  true,   false,  false,  true,   true,
    false,  false,  false,  false,  false,  true,   true,   true,
    false,  true,   false,  true,   true,   true,   true,   false,
    true,   false,  false,  false,  false,  false,  true,   false
};

const bool DataSet_1_mask::inputs::scalarA = true;
const bool DataSet_1_mask::inputs::scalarB = false;

const bool DataSet_1_mask::outputs::LANDV[128] = {
    false,  true,   false,  false,  true,   false,  false,  false,
    false,  false,  true,   false,  true,   false,  false,  false,
    true,   true,   false,  false,  true,   false,  false,  false,
    false,  false,  true,   false,  true,   true,   true,   true,
    false,  false,  false,  false,  false,  false,  true,   false,
    false,  false,  false,  false,  true,   false,  false,  true,
    false,  false,  false,  false,  true,   false,  false,  true,
    false,  false,  true,   false,  false,  false,  false,  false,
    false,  false,  true,   false,  false,  false,  false,  false,
    false,  false,  true,   false,  true,   false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  true,   false,  false,  true,   false,  true,
    false,  true,   false,  true,   false,  false,  true,   true,
    false,  false,  false,  false,  false,  false,  true,   false,
    false,  true,   false,  true,   false,  true,   true,   false,
    false,  false,  false,  false,  false,  false,  true,   false
};

const bool DataSet_1_mask::outputs::LANDS_A[128] = {
    false,  true,   false,  false,  true,   false,  false,  false,
    false,  true,   true,   false,  true,   true,   false,  false,
    true,   true,   true,   false,  true,   false,  false,  false,
    true,   false,  true,   false,  true,   true,   true,   true,
    false,  false,  false,  false,  false,  false,  true,   false,
    true,   false,  false,  false,  true,   true,   true,   true,
    false,  false,  false,  true,   true,   false,  false,  true,
    true,   true,   true,   true,   false,  false,  false,  false,
    false,  false,  true,   false,  false,  false,  false,  false,
    true,   false,  true,   true,   true,   false,  false,  false,
    true,   false,  false,  true,   true,   false,  false,  false,
    false,  false,  true,   false,  true,   true,   false,  true,
    true,   true,   true,   true,   false,  false,  true,   true,
    false,  false,  true,   false,  true,   false,  true,   false,
    false,  true,   true,   true,   false,  true,   true,   false,
    false,  true,   false,  false,  false,  true,   true,   true
};

const bool DataSet_1_mask::outputs::LANDS_B[128] = {
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false
};

const bool DataSet_1_mask::outputs::LORV[128] = {
    true,   true,   true,   false,  true,   true,   false,  false,
    false,  true,   true,   false,  true,   true,   true,   true,
    true,   true,   true,   false,  true,   false,  true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   false,  true,   false,  true,   true,   true,
    true,   false,  false,  true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   false,  true,   true,
    true,   true,   true,   true,   false,  false,  true,   false,
    false,  false,  true,   false,  false,  false,  false,  true,
    true,   true,   true,   true,   true,   true,   false,  true,
    true,   false,  true,   true,   true,   true,   false,  false,
    false,  true,   true,   false,  true,   true,   true,   true,
    true,   true,   true,   true,   false,  false,  true,   true,
    false,  false,  true,   false,  true,   true,   true,   true,
    false,  true,   true,   true,   true,   true,   true,   false,
    true,   true,   false,  false,  false,  true,   true,   true
};

const bool DataSet_1_mask::outputs::LORS_A[128] = {
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true
};

const bool DataSet_1_mask::outputs::LORS_B[128] = {
    false,  true,   false,  false,  true,   false,  false,  false,
    false,  true,   true,   false,  true,   true,   false,  false,
    true,   true,   true,   false,  true,   false,  false,  false,
    true,   false,  true,   false,  true,   true,   true,   true,
    false,  false,  false,  false,  false,  false,  true,   false,
    true,   false,  false,  false,  true,   true,   true,   true,
    false,  false,  false,  true,   true,   false,  false,  true,
    true,   true,   true,   true,   false,  false,  false,  false,
    false,  false,  true,   false,  false,  false,  false,  false,
    true,   false,  true,   true,   true,   false,  false,  false,
    true,   false,  false,  true,   true,   false,  false,  false,
    false,  false,  true,   false,  true,   true,   false,  true,
    true,   true,   true,   true,   false,  false,  true,   true,
    false,  false,  true,   false,  true,   false,  true,   false,
    false,  true,   true,   true,   false,  true,   true,   false,
    false,  true,   false,  false,  false,  true,   true,   true
};

const bool DataSet_1_mask::outputs::LXORV[128] = {
    true,   false,  true,   false,  false,  true,   false,  false,
    false,  true,   false,  false,  false,  true,   true,   true,
    false,  false,  true,   false,  false,  false,  true,   true,
    true,   true,   false,  true,   false,  false,  false,  false,
    true,   true,   false,  true,   false,  true,   false,  true,
    true,   false,  false,  true,   false,  true,   true,   false,
    true,   true,   true,   true,   false,  false,  true,   false,
    true,   true,   false,  true,   false,  false,  true,   false,
    false,  false,  false,  false,  false,  false,  false,  true,
    true,   true,   false,  true,   false,  true,   false,  true,
    true,   false,  true,   true,   true,   true,   false,  false,
    false,  true,   false,  false,  true,   false,  true,   false,
    true,   false,  true,   false,  false,  false,  false,  false,
    false,  false,  true,   false,  true,   true,   false,  true,
    false,  false,  true,   false,  true,   false,  false,  false,
    true,   true,   false,  false,  false,  true,   false,  true
};

const bool DataSet_1_mask::outputs::LXORS_A[128] = {
    true,   false,  true,   true,   false,  true,   true,   true,
    true,   false,  false,  true,   false,  false,  true,   true,
    false,  false,  false,  true,   false,  true,   true,   true,
    false,  true,   false,  true,   false,  false,  false,  false,
    true,   true,   true,   true,   true,   true,   false,  true,
    false,  true,   true,   true,   false,  false,  false,  false,
    true,   true,   true,   false,  false,  true,   true,   false,
    false,  false,  false,  false,  true,   true,   true,   true,
    true,   true,   false,  true,   true,   true,   true,   true,
    false,  true,   false,  false,  false,  true,   true,   true,
    false,  true,   true,   false,  false,  true,   true,   true,
    true,   true,   false,  true,   false,  false,  true,   false,
    false,  false,  false,  false,  true,   true,   false,  false,
    true,   true,   false,  true,   false,  true,   false,  true,
    true,   false,  false,  false,  true,   false,  false,  true,
    true,   false,  true,   true,   true,   false,  false,  false
};

const bool DataSet_1_mask::outputs::LXORS_B[128] = {
    false,  true,   false,  false,  true,   false,  false,  false,
    false,  true,   true,   false,  true,   true,   false,  false,
    true,   true,   true,   false,  true,   false,  false,  false,
    true,   false,  true,   false,  true,   true,   true,   true,
    false,  false,  false,  false,  false,  false,  true,   false,
    true,   false,  false,  false,  true,   true,   true,   true,
    false,  false,  false,  true,   true,   false,  false,  true,
    true,   true,   true,   true,   false,  false,  false,  false,
    false,  false,  true,   false,  false,  false,  false,  false,
    true,   false,  true,   true,   true,   false,  false,  false,
    true,   false,  false,  true,   true,   false,  false,  false,
    false,  false,  true,   false,  true,   true,   false,  true,
    true,   true,   true,   true,   false,  false,  true,   true,
    false,  false,  true,   false,  true,   false,  true,   false,
    false,  true,   true,   true,   false,  true,   true,   false,
    false,  true,   false,  false,  false,  true,   true,   true
};

const bool DataSet_1_mask::outputs::LNOT[128] = {
    true,   false,  true,   true,   false,  true,   true,   true,
    true,   false,  false,  true,   false,  false,  true,   true,
    false,  false,  false,  true,   false,  true,   true,   true,
    false,  true,   false,  true,   false,  false,  false,  false,
    true,   true,   true,   true,   true,   true,   false,  true,
    false,  true,   true,   true,   false,  false,  false,  false,
    true,   true,   true,   false,  false,  true,   true,   false,
    false,  false,  false,  false,  true,   true,   true,   true,
    true,   true,   false,  true,   true,   true,   true,   true,
    false,  true,   false,  false,  false,  true,   true,   true,
    false,  true,   true,   false,  false,  true,   true,   true,
    true,   true,   false,  true,   false,  false,  true,   false,
    false,  false,  false,  false,  true,   true,   false,  false,
    true,   true,   false,  true,   false,  true,   false,  true,
    true,   false,  false,  false,  true,   false,  false,  true,
    true,   false,  true,   true,   true,   false,  false,  false
};
const bool DataSet_1_mask::outputs::HLAND[128] = {
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false
};
const bool DataSet_1_mask::outputs::HLOR[128] = {
    false,  true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true
};
const bool DataSet_1_mask::outputs::HLXOR[128] = {
    false,  true,   true,   true,   false,  false,  false,  false,
    false,  true,   false,  false,  true,   false,  false,  false,
    true,   false,  true,   true,   false,  false,  false,  false,
    true,   true,   false,  false,  true,   false,  true,   false,
    false,  false,  false,  false,  false,  false,  true,   true,
    false,  false,  false,  false,  true,   false,  true,   false,
    false,  false,  false,  true,   false,  false,  false,  true,
    false,  true,   false,  true,   true,   true,   true,   true,
    true,   true,   false,  false,  false,  false,  false,  false,
    true,   true,   false,  true,   false,  false,  false,  false,
    true,   true,   true,   false,  true,   true,   true,   true,
    true,   true,   false,  false,  true,   false,  false,  true,
    false,  true,   false,  true,   true,   true,   false,  true,
    true,   true,   false,  false,  true,   true,   false,  false,
    false,  true,   false,  true,   true,   false,  true,   true,
    true,   false,  false,  false,  false,  true,   false,  true
};

const uint32_t DataSet_1_32u::inputs::inputA[32] = {
    2890127753, 3623131505, 3730078463, 2142934923,
    3878122330, 4195248000, 1453286364, 1634157816,
    587950636,  3299964697, 1823632563, 3120724469,
    1568573771, 3435400464, 703518081,  817188400,
    3165700845, 27010288,   3087364633, 431601683,
    4030944788, 1853857795, 2691787100, 868104033,
    3265031366, 2189542274, 180288151,  2452030208,
    2748106285, 4225759456, 4035989014, 446256628
};

const uint32_t DataSet_1_32u::inputs::inputB[32] = {
    15667904,   794793862,  147627889,  2249388176, 
    1586257963, 3765622899, 4226445805, 469642333,
    2084138484, 157623475,  4146090199, 2263217432,
    2261116474, 289745607,  2743838183, 2028729679,
    2086680194, 444702103,  3207991232, 460691727,
    3667110195, 2667605829, 2649569023, 189568803,
    2603418874, 1317779160, 2953755803, 3129831066,
    1022843651, 904356825,  2189632558, 1398010737
};

const uint32_t DataSet_1_32u::inputs::inputC[32] = {
    1689902809, 2348736796, 3292230136, 1569671280,
    777549897,  2869579483, 1789754979, 1767019178,
    3643171494, 2291920162, 796845760,  1526756488,
    3027058465, 3350141993, 2620505403, 1857313343,
    709637021,  3115471696, 4281729486, 1711164905,
    1594329641, 2546652425, 1922139181, 587383476,
    2317880783, 2983673643, 2020558581, 2365802563,
    3332725190, 2891378317, 3720270723, 3845563885
};

const uint32_t DataSet_1_32u::inputs::inputShiftA[32] = {
    52,     4,      29,     44,
    7,      13,     11,     8,
    46,     53,     13,     62,
    39,     26,     19,     40,
    45,     13,     47,     1008,
    694,    529,    790,    492,
    58,     702,    206,    761,
    201,    8,      851,    691
};

const uint32_t DataSet_1_32u::inputs::scalarA = 636364;
const uint32_t DataSet_1_32u::inputs::inputShiftScalarA = 27;

const bool DataSet_1_32u::inputs::maskA[32] = {
    false,   false,  false,  true,   // 4
    false,  true,   true,   false,  // 8

    false,  true,   true,   false,  
    true,   false,  false,  true,   // 16
    
    false,  true,   true,   false,
    true,   false,  false,  true,
    true,   false,  false,  true,
    false,  true,   true,   false,  // 32
};

const uint32_t DataSet_1_32u::outputs::ADDV[32] = {
    2905795657, 122958071,  3877706352, 97355803,
    1169412997, 3665903603, 1384764873, 2103800149,
    2672089120, 3457588172, 1674755466, 1088974605,
    3829690245, 3725146071, 3447356264, 2845918079,
    957413743,  471712391,  2000388569, 892293410,
    3403087687, 226496328,  1046388827, 1057672836,
    1573482944, 3507321434, 3134043954, 1286893978,
    3770949936, 835148985,  1930654276, 1844267365
};

const uint32_t DataSet_1_32u::outputs::MADDV[32] = {
    2890127753, 3623131505, 3730078463, 97355803,
    3878122330, 3665903603, 1384764873, 1634157816,
    587950636,  3457588172, 1674755466, 3120724469,
    3829690245, 3435400464, 703518081,  2845918079,
    3165700845, 471712391,  2000388569, 431601683,
    3403087687, 1853857795, 2691787100, 1057672836,
    1573482944, 2189542274, 180288151,  1286893978,
    2748106285, 835148985,  1930654276, 446256628
};

const uint32_t DataSet_1_32u::outputs::ADDS[32] = {
    2890764117, 3623767869, 3730714827, 2143571287,
    3878758694, 4195884364, 1453922728, 1634794180,
    588587000,  3300601061, 1824268927, 3121360833,
    1569210135, 3436036828, 704154445,  817824764,
    3166337209, 27646652,   3088000997, 432238047,
    4031581152, 1854494159, 2692423464, 868740397,
    3265667730, 2190178638, 180924515,  2452666572,
    2748742649, 4226395820, 4036625378, 446892992
};

const uint32_t DataSet_1_32u::outputs::MADDS[32] = {
    2890127753, 3623131505, 3730078463, 2143571287,
    3878122330, 4195884364, 1453922728, 1634157816,
    587950636,  3300601061, 1824268927, 3120724469,
    1569210135, 3435400464, 703518081,  817824764,
    3165700845, 27646652,   3088000997, 431601683,
    4031581152, 1853857795, 2691787100, 868740397,
    3265667730, 2189542274, 180288151,  2452666572,
    2748106285, 4226395820, 4036625378, 446256628
};

const uint32_t DataSet_1_32u::outputs::POSTPREFINC[32] = {
    2890127754, 3623131506, 3730078464, 2142934924,
    3878122331, 4195248001, 1453286365, 1634157817,
    587950637,  3299964698, 1823632564, 3120724470,
    1568573772, 3435400465, 703518082,  817188401,
    3165700846, 27010289,   3087364634, 431601684,
    4030944789, 1853857796, 2691787101, 868104034,
    3265031367, 2189542275, 180288152,  2452030209,
    2748106286, 4225759457, 4035989015, 446256629
};

const uint32_t DataSet_1_32u::outputs::MPOSTPREFINC[32] = {
    2890127753, 3623131505, 3730078463, 2142934924,
    3878122330, 4195248001, 1453286365, 1634157816,
    587950636,  3299964698, 1823632564, 3120724469,
    1568573772, 3435400464, 703518081,  817188401,
    3165700845, 27010289,   3087364634, 431601683,
    4030944789, 1853857795, 2691787100, 868104034,
    3265031367, 2189542274, 180288151,  2452030209,
    2748106285, 4225759457, 4035989015, 446256628
};

const uint32_t DataSet_1_32u::outputs::SUBV[32] = {
    2874459849, 2828337643, 3582450574, 4188514043,
    2291864367, 429625101,  1521807855, 1164515483,
    2798779448, 3142341222, 1972509660, 857507037,
    3602424593, 3145654857, 2254647194, 3083426017,
    1079020651, 3877275481, 4174340697, 4265877252,
    363834593,  3481219262, 42218077,   678535230,
    661612492,  871763114,  1521499644, 3617166438,
    1725262634, 3321402631, 1846356456, 3343213187
};

const uint32_t DataSet_1_32u::outputs::MSUBV[32] = {
    2890127753, 3623131505, 3730078463, 4188514043,
    3878122330, 429625101,  1521807855, 1634157816,
    587950636,  3142341222, 1972509660, 3120724469,
    3602424593, 3435400464, 703518081,  3083426017,
    3165700845, 3877275481, 4174340697, 431601683,
    363834593,  1853857795, 2691787100, 678535230,
    661612492,  2189542274, 180288151,  3617166438,
    2748106285, 3321402631, 1846356456, 446256628
};

const uint32_t DataSet_1_32u::outputs::SUBS[32] = {
    2889491389, 3622495141, 3729442099, 2142298559,
    3877485966, 4194611636, 1452650000, 1633521452,
    587314272,  3299328333, 1822996199, 3120088105,
    1567937407, 3434764100, 702881717,  816552036,
    3165064481, 26373924,   3086728269, 430965319,
    4030308424, 1853221431, 2691150736, 867467669,
    3264395002, 2188905910, 179651787,  2451393844,
    2747469921, 4225123092, 4035352650, 445620264
};

const uint32_t DataSet_1_32u::outputs::MSUBS[32] = {
    2890127753, 3623131505, 3730078463, 2142298559,
    3878122330, 4194611636, 1452650000, 1634157816,
    587950636,  3299328333, 1822996199, 3120724469,
    1567937407, 3435400464, 703518081,  816552036,
    3165700845, 26373924,   3086728269, 431601683,
    4030308424, 1853857795, 2691787100, 867467669,
    3264395002, 2189542274, 180288151,  2451393844,
    2748106285, 4225123092, 4035352650, 446256628
};

const uint32_t DataSet_1_32u::outputs::SUBFROMV[32] = {
    1420507447, 1466629653, 712516722,  106453253,
    2003102929, 3865342195, 2773159441, 3130451813,
    1496187848, 1152626074, 2322457636, 3437460259,
    692542703,  1149312439, 2040320102, 1211541279,
    3215946645, 417691815,  120626599,  29090044,
    3931132703, 813748034,  4252749219, 3616432066,
    3633354804, 3423204182, 2773467652, 677800858,
    2569704662, 973564665,  2448610840, 951754109
};

const uint32_t DataSet_1_32u::outputs::MSUBFROMV[32] = {
    15667904,   794793862,  147627889,  106453253,
    1586257963, 3865342195, 2773159441, 469642333,
    2084138484, 1152626074, 2322457636, 2263217432,
    692542703,  289745607,  2743838183, 1211541279,
    2086680194, 417691815,  120626599,  460691727,
    3931132703, 2667605829, 2649569023, 3616432066,
    3633354804, 1317779160, 2953755803, 677800858,
    1022843651, 973564665,  2448610840, 1398010737
};

const uint32_t DataSet_1_32u::outputs::SUBFROMS[32] = {
    1405475907, 672472155,  565525197,  2152668737,
    417481330,  100355660,  2842317296, 2661445844,
    3707653024, 995638963,  2471971097, 1174879191,
    2727029889, 860203196,  3592085579, 3478415260,
    1129902815, 4268593372, 1208239027, 3864001977,
    264658872,  2441745865, 1603816560, 3427499627,
    1030572294, 2106061386, 4115315509, 1843573452,
    1547497375, 69844204,   259614646,  3849347032
};

const uint32_t DataSet_1_32u::outputs::MSUBFROMS[32] = {
    636364,     636364,     636364,     2152668737,
    636364,     100355660,  2842317296, 636364,
    636364,     995638963,  2471971097, 636364,
    2727029889, 636364,     636364,     3478415260,
    636364,     4268593372, 1208239027, 636364,
    264658872,  636364,     636364,     3427499627,
    1030572294, 636364,     636364,     1843573452,
    636364,     69844204,   259614646,  636364
};

const uint32_t DataSet_1_32u::outputs::POSTPREFDEC[32] = {
    2890127752, 3623131504, 3730078462, 2142934922,
    3878122329, 4195247999, 1453286363, 1634157815,
    587950635,  3299964696, 1823632562, 3120724468,
    1568573770, 3435400463, 703518080,  817188399,
    3165700844, 27010287,   3087364632, 431601682,
    4030944787, 1853857794, 2691787099, 868104032,
    3265031365, 2189542273, 180288150,  2452030207,
    2748106284, 4225759455, 4035989013, 446256627
};

const uint32_t DataSet_1_32u::outputs::MPOSTPREFDEC[32] = {
    2890127753, 3623131505, 3730078463, 2142934922,
    3878122330, 4195247999, 1453286363, 1634157816,
    587950636,  3299964696, 1823632562, 3120724469,
    1568573770, 3435400464, 703518081,  817188399,
    3165700845, 27010287,   3087364632, 431601683,
    4030944787, 1853857795, 2691787100, 868104032,
    3265031365, 2189542274, 180288151,  2452030207,
    2748106285, 4225759455, 4035989013, 446256628
};

const uint32_t DataSet_1_32u::outputs::MULV[32] = { 
    253085888,  7927846,    2531437455, 818777648,  
    3222937630, 2462875264, 2277215916, 3094261272,  
    920776176,  1521666683, 1462843989, 622585848,
    3334175998, 3956983664, 3593063527, 3520532688,  
    4290837082, 2997942672, 235604416,  2727617565, 
    3334035964, 4224150991, 3009239716, 3782723907,  
    2135118172, 571564464,  1173003629, 1438764544,  
    2000900487, 272143840,  2787844084, 1147243700
};

const uint32_t DataSet_1_32u::outputs::MMULV[32] = {
    2890127753, 3623131505, 3730078463, 818777648,
    3878122330, 2462875264, 2277215916, 1634157816,
    587950636,  1521666683, 1462843989, 3120724469,
    3334175998, 3435400464, 703518081,  3520532688,
    3165700845, 2997942672, 235604416,  431601683,
    3334035964, 1853857795, 2691787100, 3782723907,
    2135118172, 2189542274, 180288151,  1438764544,
    2748106285, 272143840,  2787844084, 446256628
};

const uint32_t DataSet_1_32u::outputs::MULS[32] = {
    3836753452, 1818241804, 960450100,  163121604,
    640191928,  371718656,  995762000,  247877024,
    3132471056, 1719702764, 2538876324, 1137731644,
    3414847172, 1057405120, 3369031628, 3228712512,
    822207964,  4210761536, 2162399468, 1204756004,
    406371312,  3924861284, 3189375312, 2871309900,
    2156178376, 2654254488, 1722512212, 657810432,
    1189133532, 3511726720, 2029635464, 2710176368
};

const uint32_t DataSet_1_32u::outputs::MMULS[32] = {
    2890127753, 3623131505, 3730078463, 163121604,
    3878122330, 371718656,  995762000,  1634157816,
    587950636,  1719702764, 2538876324, 3120724469,
    3414847172, 3435400464, 703518081,  3228712512,
    3165700845, 4210761536, 2162399468, 431601683,
    406371312,  1853857795, 2691787100, 2871309900,
    2156178376, 2189542274, 180288151,  657810432,
    2748106285, 3511726720, 2029635464, 446256628
};

const uint32_t DataSet_1_32u::outputs::DIVV[32] = {
    184,    4,  25, 0,
    2,      1,  0,  3,
    0,      20, 0,  1,
    0,      11, 0,  0,
    1,      0,  0,  0,
    1,      0,  1,  4,
    1,      1,  0,  0,
    2,      4,  1,  0
};

const uint32_t DataSet_1_32u::outputs::MDIVV[32] = {
    2890127753, 3623131505, 3730078463, 0,
    3878122330, 1,          0,          1634157816,
    587950636,  20,         0,          3120724469,
    0,          3435400464, 703518081,  0,
    3165700845, 0,          0,          431601683,
    1,          1853857795, 2691787100, 4,
    1,          2189542274, 180288151,  0,
    2748106285, 4,          1,          446256628
};

const uint32_t DataSet_1_32u::outputs::DIVS[32] = {
    4541,   5693,   5861,   3367,
    6094,   6592,   2283,   2567,
    923,    5185,   2865,   4903,
    2464,   5398,   1105,   1284,
    4974,   42,     4851,   678,
    6334,   2913,   4229,   1364,
    5130,   3440,   283,    3853,
    4318,   6640,   6342,   701
};

const uint32_t DataSet_1_32u::outputs::MDIVS[32] = {
    2890127753, 3623131505, 3730078463, 3367,
    3878122330, 6592,       2283,       1634157816,
    587950636,  5185,       2865,       3120724469,
    2464,       3435400464, 703518081,  1284,
    3165700845, 42,         4851,       431601683,
    6334,       1853857795, 2691787100, 1364,
    5130,       2189542274, 180288151,  3853,
    2748106285, 6640,       6342,       446256628
};

const uint32_t DataSet_1_32u::outputs::RCP[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

const uint32_t DataSet_1_32u::outputs::MRCP[32] = {
    2890127753, 3623131505, 3730078463, 0,
    3878122330, 0,          0,          1634157816,
    587950636,  0,          0,          3120724469,
    0,          3435400464, 703518081,  0,
    3165700845, 0,          0,          431601683,
    0,          1853857795, 2691787100, 0,
    0,          2189542274, 180288151,  0,
    2748106285, 0,          0,          446256628
};

const uint32_t DataSet_1_32u::outputs::RCPS[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

const uint32_t DataSet_1_32u::outputs::MRCPS[32] = {
    2890127753, 3623131505, 3730078463, 0,
    3878122330, 0,          0,          1634157816,
    587950636,  0,          0,          3120724469,
    0,          3435400464, 703518081,  0,
    3165700845, 0,          0,          431601683,
    0,          1853857795, 2691787100, 0,
    0,          2189542274, 180288151,  0,
    2748106285, 0,          0,          446256628
};

const bool DataSet_1_32u::outputs::CMPEQV[32] = {
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false
};

const bool DataSet_1_32u::outputs::CMPEQS[32] = {
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false
};

const bool DataSet_1_32u::outputs::CMPNEV[32] = {
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true
};

const bool DataSet_1_32u::outputs::CMPNES[32] = {
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true
};

const bool DataSet_1_32u::outputs::CMPGTV[32] = {
    true,   true,   true,   false,  true,   true,   false,  true,
    false,  true,   false,  true,   false,  true,   false,  false,
    true,   false,  false,  false,  true,   false,  true,   true,
    true,   true,   false,  false,  true,   true,   true,   false
};

const bool DataSet_1_32u::outputs::CMPGTS[32] = {
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true
};

const bool DataSet_1_32u::outputs::CMPLTV[32] = {
    false,  false,  false,  true,   false,  false,  true,   false,
    true,   false,  true,   false,  true,   false,  true,   true,
    false,  true,   true,   true,   false,  true,   false,  false,
    false,  false,  true,   true,   false,  false,  false,  true
};

const bool DataSet_1_32u::outputs::CMPLTS[32] = {
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false
};

const bool DataSet_1_32u::outputs::CMPGEV[32] = {
    true,   true,   true,   false,  true,   true,   false,  true,
    false,  true,   false,  true,   false,  true,   false,  false,
    true,   false,  false,  false,  true,   false,  true,   true,
    true,   true,   false,  false,  true,   true,   true,   false
};

const bool DataSet_1_32u::outputs::CMPGES[32] = {
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true
};

const bool DataSet_1_32u::outputs::CMPLEV[32] = {
    false,  false,  false,  true,   false,  false,  true,   false,
    true,   false,  true,   false,  true,   false,  true,   true,
    false,  true,   true,   true,   false,  true,   false,  false,
    false,  false,  true,   true,   false,  false,  false,  true
};

const bool DataSet_1_32u::outputs::CMPLES[32] = {
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false
};

const bool DataSet_1_32u::outputs::CMPEV = false;
const bool DataSet_1_32u::outputs::CMPES = false;

const uint32_t DataSet_1_32u::outputs::HADD[32] = {
    2890127753, 2218291962, 1653403129, 3796338052,
    3379493086, 3279773790, 438092858,  2072250674,
    2660201310, 1665198711, 3488831274, 2314588447,
    3883162218, 3023595386, 3727113467, 249334571,
    3415035416, 3442045704, 2234443041, 2666044724,
    2402022216, 4255880011, 2652699815, 3520803848,
    2490867918, 385442896,  565731047,  3017761255,
    1470900244, 1401692404, 1142714122, 1588970750
};

const uint32_t DataSet_1_32u::outputs::MHADD[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

const uint32_t DataSet_1_32u::outputs::HMUL[32] = {
    2890127753, 4236896256, 3538304135, 1137475149,
    1801190930, 708069120,  2506228736, 1543757824,
    191528960,  518422528,  3643506688, 1713471488,
    3645931520, 2877816832, 2542272512, 3649044480,
    1535115264, 3355443200, 134217728,  2550136832,
    3758096384, 2684354560, 0,          0,
    0,          0,          0,          0,
    0,          0,          0,          0
};

const uint32_t DataSet_1_32u::outputs::MHMUL[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

const uint32_t DataSet_1_32u::outputs::BANDV[32] = {
    4395136,    122982656,  138680945,  101879936,
    1174691850, 3758243840, 1384800716, 23463000,
    537487396,  2171921,    1679835283, 2181067024,
    71598090,   4458496,    562726273,  815875072,
    1008730752, 8389776,    3087270400, 422680579,
    3489857552, 234889217,  2153848924, 51124001,
    2181825730, 42041984,   964755,     2449801728,
    549732865,  835150016,  2155873286, 303124848
};

const uint32_t DataSet_1_32u::outputs::MBANDV[32] = {
    2890127753, 3623131505, 3730078463, 101879936,
    3878122330, 3758243840, 1384800716, 1634157816,
    587950636,  2171921,    1679835283, 3120724469,
    71598090,   3435400464, 703518081,  815875072,
    3165700845, 8389776,    3087270400, 431601683,
    3489857552, 1853857795, 2691787100, 51124001,
    2181825730, 2189542274, 180288151,  2449801728,
    2748106285, 835150016,  2155873286, 446256628
};

const uint32_t DataSet_1_32u::outputs::BANDS[32] = {
    102792, 37184,  12492,  562568,
    78152,  533888, 598476, 66760,
    598028, 79112,  9344,   12740,
    557384, 1280,   562560, 66560,
    32972,  533696, 77832,  634880,
    70660,  630784, 74060,  537920,
    533700, 98688,  569476, 65792,
    558092, 635072, 1028,   595396
};

const uint32_t DataSet_1_32u::outputs::MBANDS[32] = {
    2890127753, 3623131505, 3730078463, 562568,
    3878122330, 533888,     598476,     1634157816,
    587950636,  79112,      9344,       3120724469,
    557384,     3435400464, 703518081,  66560,
    3165700845, 533696,     77832,      431601683,
    70660,      1853857795, 2691787100, 537920,
    533700,     2189542274, 180288151,  65792,
    2748106285, 635072,     1028,       446256628
};

const uint32_t DataSet_1_32u::outputs::BORV[32] = {
    2901400521, 4294942711, 3739025407, 4290443163,
    4289688443, 4202627059, 4294931453, 2080337149,
    2134601724, 3455416251, 4289887479, 3202874877,
    3758092155, 3720687575, 2884629991, 2030043007,
    4243650287, 463322615,  3208085465, 469612831,
    4208197431, 4286574407, 3187507199, 1006548835,
    3686624510, 3465279450, 3133079199, 3132059546,
    3221217071, 4294966265, 4069748286, 1541142517
};

const uint32_t DataSet_1_32u::outputs::MBORV[32] = {
    2890127753, 3623131505, 3730078463, 4290443163,
    3878122330, 4202627059, 4294931453, 1634157816,
    587950636,  3455416251, 4289887479, 3120724469,
    3758092155, 3435400464, 703518081,  2030043007,
    3165700845, 463322615,  3208085465, 431601683,
    4208197431, 1853857795, 2691787100, 1006548835,
    3686624510, 2189542274, 180288151,  3132059546,
    2748106285, 4294966265, 4069748286, 446256628
};

const uint32_t DataSet_1_32u::outputs::BORS[32] = {
    2890661325, 3623730685, 3730702335, 2143008719,
    3878680542, 4195350476, 1453324252, 1634727420,
    587988972,  3300521949, 1824259583, 3121348093,
    1568652751, 3436035548, 703591885,  817758204,
    3166304237, 27112956,   3087923165, 431603167,
    4031510492, 1853863375, 2692349404, 868202477,
    3265134030, 2190079950, 180355039,  2452600780,
    2748184557, 4225760748, 4036624350, 446297596
};

const uint32_t DataSet_1_32u::outputs::MBORS[32] = {
    2890127753, 3623131505, 3730078463, 2143008719,
    3878122330, 4195350476, 1453324252, 1634157816,
    587950636,  3300521949, 1824259583, 3120724469,
    1568652751, 3435400464, 703518081,  817758204,
    3165700845, 27112956,   3087923165, 431601683,
    4031510492, 1853857795, 2691787100, 868202477,
    3265134030, 2189542274, 180288151,  2452600780,
    2748106285, 4225760748, 4036624350, 446256628
};

const uint32_t DataSet_1_32u::outputs::BXORV[32] = {
    2897005385, 4171960055, 3600344462, 4188563227,
    3114996593, 444383219,  2910130737, 2056874149,
    1597114328, 3453244330, 2610052196, 1021807853,
    3686494065, 3716229079, 2321903718, 1214167935,
    3234919535, 454932839,  120815065,  46932252,
    718339879,  4051685190, 1033658275, 955424834,
    1504798780, 3423237466, 3132114444, 682257818,
    2671484206, 3459816249, 1913875000, 1238017669
};

const uint32_t DataSet_1_32u::outputs::MBXORV[32] = {
    2890127753, 3623131505, 3730078463, 4188563227,
    3878122330, 444383219,  2910130737, 1634157816,
    587950636,  3453244330, 2610052196, 3120724469,
    3686494065, 3435400464, 703518081,  1214167935,
    3165700845, 454932839,  120815065,  431601683,
    718339879,  1853857795, 2691787100, 955424834,
    1504798780, 2189542274, 180288151,  682257818,
    2748106285, 3459816249, 1913875000, 446256628
};

const uint32_t DataSet_1_32u::outputs::BXORS[32] = {
    2890558533, 3623693501, 3730689843, 2142446151,
    3878602390, 4194816588, 1452725776, 1634660660,
    587390944,  3300442837, 1824250239, 3121335353,
    1568095367, 3436034268, 703029325,  817691644,
    3166271265, 26579260,   3087845333, 430968287,
    4031439832, 1853232591, 2692275344, 867664557,
    3264600330, 2189981262, 179785563,  2452534988,
    2747626465, 4225125676, 4036623322, 445702200
};

const uint32_t DataSet_1_32u::outputs::MBXORS[32] = {
    2890127753, 3623131505, 3730078463, 2142446151,
    3878122330, 4194816588, 1452725776, 1634157816,
    587950636,  3300442837, 1824250239, 3120724469,
    1568095367, 3435400464, 703518081,  817691644,
    3165700845, 26579260,   3087845333, 431601683,
    4031439832, 1853857795, 2691787100, 867664557,
    3264600330, 2189542274, 180288151,  2452534988,
    2748106285, 4225125676, 4036623322, 446256628
};

const uint32_t DataSet_1_32u::outputs::BNOT[32] = {
    1404839542, 671835790,  564888832,  2152032372,
    416844965,  99719295,   2841680931, 2660809479,
    3707016659, 995002598,  2471334732, 1174242826,
    2726393524, 859566831,  3591449214, 3477778895,
    1129266450, 4267957007, 1207602662, 3863365612,
    264022507,  2441109500, 1603180195, 3426863262,
    1029935929, 2105425021, 4114679144, 1842937087,
    1546861010, 69207839,   258978281,  3848710667
};

const uint32_t DataSet_1_32u::outputs::MBNOT[32] = {
    2890127753, 3623131505, 3730078463, 2152032372,
    3878122330, 99719295,   2841680931, 1634157816,
    587950636,  995002598,  2471334732, 3120724469,
    2726393524, 3435400464, 703518081,  3477778895,
    3165700845, 4267957007, 1207602662, 431601683,
    264022507,  1853857795, 2691787100, 3426863262,
    1029935929, 2189542274, 180288151,  1842937087,
    2748106285, 69207839,   258978281,  446256628
};

const uint32_t DataSet_1_32u::outputs::HBAND[32] = {
    2890127753, 2218825985, 2218792961, 67112961,
    67112960,   0,          0,          0,
    0,          0,          0,          0,
    0,          0,          0,          0,
    0,          0,          0,          0,
    0,          0,          0,          0,
    0,          0,          0,          0,
    0,          0,          0,          0
};

const uint32_t DataSet_1_32u::outputs::MHBAND[32] = {
    4294967295,     4294967295,     4294967295,     2142934923,
    2142934923,     2047477632,     1376387968,     1376387968,
    1376387968,     1073742592,     1073741824,     1073741824,
    1073741824,     1073741824,     1073741824,     0,
    0,              0,              0,              0,
    0,              0,              0,              0,
    0,              0,              0,              0,
    0,              0,              0,              0
};
const uint32_t DataSet_1_32u::outputs::HBANDS[32] = {
    102792, 37120,  4096,   4096,
    4096,   0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0
};

const uint32_t DataSet_1_32u::outputs::MHBANDS[32] = {
    636364, 636364, 636364, 562568,
    562568, 525696, 524672, 524672,
    524672, 256,    0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0
};
const uint32_t DataSet_1_32u::outputs::HBOR[32] = {
    2890127753, 4294433273, 4294441983, 4294967295,
    4294967295, 4294967295, 4294967295, 4294967295,
    4294967295, 4294967295, 4294967295, 4294967295,
    4294967295, 4294967295, 4294967295, 4294967295,
    4294967295, 4294967295, 4294967295, 4294967295,
    4294967295, 4294967295, 4294967295, 4294967295,
    4294967295, 4294967295, 4294967295, 4294967295,
    4294967295, 4294967295, 4294967295, 4294967295
};

const uint32_t DataSet_1_32u::outputs::MHBOR[32] = {
    0,              0,              0,              2142934923,
    2142934923,     4290705291,     4290770911,     4290770911,
    4290770911,     4290770911,     4290772991,     4290772991,
    4294967295,     4294967295,     4294967295,     4294967295,
    4294967295,     4294967295,     4294967295,     4294967295,
    4294967295,     4294967295,     4294967295,     4294967295,
    4294967295,     4294967295,     4294967295,     4294967295,
    4294967295,     4294967295,     4294967295,     4294967295
};
const uint32_t DataSet_1_32u::outputs::HBORS[32] = {
    2890661325,     4294966781,     4294967295,     4294967295,
    4294967295,     4294967295,     4294967295,     4294967295,
    4294967295,     4294967295,     4294967295,     4294967295,
    4294967295,     4294967295,     4294967295,     4294967295,
    4294967295,     4294967295,     4294967295,     4294967295,
    4294967295,     4294967295,     4294967295,     4294967295,
    4294967295,     4294967295,     4294967295,     4294967295,
    4294967295,     4294967295,     4294967295,     4294967295
};

const uint32_t DataSet_1_32u::outputs::MHBORS[32] = {
    636364,         636364,         636364,         2143008719,
    2143008719,     4290770895,     4290770911,     4290770911,
    4290770911,     4290770911,     4290772991,     4290772991,
    4294967295,     4294967295,     4294967295,     4294967295,
    4294967295,     4294967295,     4294967295,     4294967295,
    4294967295,     4294967295,     4294967295,     4294967295,
    4294967295,     4294967295,     4294967295,     4294967295,
    4294967295,     4294967295,     4294967295,     4294967295
};

const uint32_t DataSet_1_32u::outputs::HBXOR[32] = {
    2890127753, 2075607288, 2783132167, 3663310220,
    1031724758, 3346053462, 2448415370, 4035485298,
    3548640350, 389186375,  2072047604, 3246560769,
    2633768778, 1345851994, 2044119003, 1231260139,
    4124265222, 4098846710, 1279925743, 1442031100,
    2779837416, 3419362283, 1807628983, 1476398550,
    2593939728, 404595346,  312694789,  2156157701,
    591961384,  3633800648, 671558622,  849247786
};

const uint32_t DataSet_1_32u::outputs::MHBXOR[32] = {
    0,              0,              0,              2142934923,
    2142934923,     2243227659,     3542848471,     3542848471,
    3542848471,     396027086,      2066253949,     2066253949,
    643170614,      643170614,      643170614,      383995654,
    383995654,      394226678,      2944014831,     2944014831,
    1597591547,     1597591547,     1597591547,     1820817562,
    2921010268,     2921010268,     2921010268,     1010571100,
    1010571100,     3353604028,     930324906,      930324906
};
const uint32_t DataSet_1_32u::outputs::HBXORS[32] = {
    2890558533,     2076112180,     2783612875,     3662682176,
    1031236378,     3346599066,     2447798086,     4034997182,
    3549053330,     389756555,      2072599096,     3247121357,
    2633345670,     1345431446,     2044686871,     1231698983,
    4124769994,     4098284090,     1279502371,     1442451504,
    2780382756,     3418809895,     1807191931,     1477031962,
    2593510620,     403969886,      313191881,      2156785353,
    591472868,      3634286596,     672046610,      848808934
};

const uint32_t DataSet_1_32u::outputs::MHBXORS[32] = {
    636364,         636364,         636364,         2142446151,
    2142446151,     2243773895,     3542230555,     3542230555,
    3542230555,     395530498,      2065776049,     2065776049,
    643806458,      643806458,      643806458,      384498378,
    384498378,      393666106,      2943593507,     2943593507,
    1597045303,     1597045303,     1597045303,     1821298006,
    2920456592,     2920456592,     2920456592,     1010147984,
    1010147984,     3354024560,     930747494,      930747494
};

const uint32_t DataSet_1_32u::outputs::FMULADDV[32] = {
    1942988697, 2356664642, 1528700295, 2388448928,
    4000487527, 1037487451, 4066970895, 566313154,
    268980374,  3813586845, 2259689749, 2149342336,
    2066267167, 3012158361, 1918601634, 1082878735,
    705506807,  1818447072, 222366606,  143815174,
    633398309,  2475836120, 636411601,  75140087,
    158031659,  3555238107, 3193562210, 3804567107,
    1038658381, 3163522157, 2213147511, 697840289
};

const uint32_t DataSet_1_32u::outputs::MFMULADDV[32] = {
    2890127753, 3623131505, 3730078463, 2388448928,
    3878122330, 1037487451, 4066970895, 1634157816,
    587950636,  3813586845, 2259689749, 3120724469,
    2066267167, 3435400464, 703518081,  1082878735,
    3165700845, 1818447072, 222366606,  431601683,
    633398309,  1853857795, 2691787100, 75140087,
    158031659,  2189542274, 180288151,  3804567107,
    2748106285, 3163522157, 2213147511, 446256628
};

const uint32_t DataSet_1_32u::outputs::FMULSUBV[32] = {
    2858150375, 1954158346, 3534174615, 3544073664,
    2445387733, 3888263077, 487460937,  1327242094,
    1572571978, 3524713817, 665998229,  3390796656,
    307117533,  606841671,  972558124,  1663219345,
    3581200061, 4177438272, 248842226,  1016452660,
    1739706323, 1677498566, 1087100535, 3195340431,
    4112204685, 1882858117, 3447412344, 3367929277,
    2963142593, 1675732819, 3362540657, 1596647111
};

const uint32_t DataSet_1_32u::outputs::MFMULSUBV[32] = {
    2890127753, 3623131505, 3730078463, 3544073664,
    3878122330, 3888263077, 487460937,  1634157816,
    587950636,  3524713817, 665998229,  3120724469,
    307117533,  3435400464, 703518081,  1663219345,
    3165700845, 4177438272, 248842226,  431601683,
    1739706323, 1853857795, 2691787100, 3195340431,
    4112204685, 2189542274, 180288151,  3367929277,
    2748106285, 1675732819, 3362540657, 446256628
};

const uint32_t DataSet_1_32u::outputs::FADDMULV[32] = {
    1037211617, 2119201284, 1552747648, 776479696,
    3341134061, 512474849,  1968937659, 1245297778,
    3592447168, 29396248,   3561235328, 1825366760,
    1139088165, 3733878127, 3342743800, 2008433217,
    2752386579, 1077710640, 3089782174, 2368999922,
    3053201503, 2366075272, 450632703,  757152976,
    1453963840, 1777709854, 3852861146, 4193979214,
    2334874400, 891817957,  1822000332, 2303265665
};

const uint32_t DataSet_1_32u::outputs::MFADDMULV[32] = {
    2890127753, 3623131505, 3730078463, 776479696,
    3878122330, 512474849,  1968937659, 1634157816,
    587950636,  29396248,   3561235328, 3120724469,
    1139088165, 3435400464, 703518081,  2008433217,
    3165700845, 1077710640, 3089782174, 431601683,
    3053201503, 1853857795, 2691787100, 757152976,
    1453963840, 2189542274, 180288151,  4193979214,
    2748106285, 891817957,  1822000332, 446256628
};

const uint32_t DataSet_1_32u::outputs::FSUBMULV[32] = {
    3934244449, 3878011060, 624975760,  3582240208,
    2942206055, 2469090847, 2542809453, 2088287470,
    1996946512, 2620219276, 2363788544, 2723854696,
    2150315057, 1459173809, 662350462,  805304671,
    3139661471, 1057892048, 1812561054, 1129082532,
    1702558217, 2800929966, 2346831449, 1171236760,
    3141159412, 1949824654, 4225907244, 2317387442,
    1231322236, 3905984219, 2356586936, 710713927
};

const uint32_t DataSet_1_32u::outputs::MFSUBMULV[32] = {
    2890127753, 3623131505, 3730078463, 3582240208,
    3878122330, 2469090847, 2542809453, 1634157816,
    587950636,  2620219276, 2363788544, 3120724469,
    2150315057, 3435400464, 703518081,  805304671,
    3165700845, 1057892048, 1812561054, 431601683,
    1702558217, 1853857795, 2691787100, 1171236760,
    3141159412, 2189542274, 180288151,  2317387442,
    2748106285, 3905984219, 2356586936, 446256628
};

const uint32_t DataSet_1_32u::outputs::MAXV[32] = {
    2890127753, 3623131505, 3730078463, 2249388176,
    3878122330, 4195248000, 4226445805, 1634157816,
    2084138484, 3299964697, 4146090199, 3120724469,
    2261116474, 3435400464, 2743838183, 2028729679,
    3165700845, 444702103,  3207991232, 460691727,
    4030944788, 2667605829, 2691787100, 868104033,
    3265031366, 2189542274, 2953755803, 3129831066,
    2748106285, 4225759456, 4035989014, 1398010737
};

const uint32_t DataSet_1_32u::outputs::MMAXV[32] = {
    2890127753, 3623131505, 3730078463, 2249388176,
    3878122330, 4195248000, 4226445805, 1634157816,
    587950636,  3299964697, 4146090199, 3120724469,
    2261116474, 3435400464, 703518081,  2028729679,
    3165700845, 444702103,  3207991232, 431601683,
    4030944788, 1853857795, 2691787100, 868104033,
    3265031366, 2189542274, 180288151,  3129831066,
    2748106285, 4225759456, 4035989014, 446256628
};

const uint32_t DataSet_1_32u::outputs::MAXS[32] = {
    2890127753, 3623131505, 3730078463, 2142934923,
    3878122330, 4195248000, 1453286364, 1634157816,
    587950636,  3299964697, 1823632563, 3120724469,
    1568573771, 3435400464, 703518081,  817188400,
    3165700845, 27010288,   3087364633, 431601683,
    4030944788, 1853857795, 2691787100, 868104033,
    3265031366, 2189542274, 180288151,  2452030208,
    2748106285, 4225759456, 4035989014, 446256628
};

const uint32_t DataSet_1_32u::outputs::MMAXS[32] = {
    2890127753, 3623131505, 3730078463, 2142934923,
    3878122330, 4195248000, 1453286364, 1634157816,
    587950636,  3299964697, 1823632563, 3120724469,
    1568573771, 3435400464, 703518081,  817188400,
    3165700845, 27010288,   3087364633, 431601683,
    4030944788, 1853857795, 2691787100, 868104033,
    3265031366, 2189542274, 180288151,  2452030208,
    2748106285, 4225759456, 4035989014, 446256628
};

const uint32_t DataSet_1_32u::outputs::MINV[32] = {
    15667904,   794793862,  147627889,  2142934923,
    1586257963, 3765622899, 1453286364, 469642333,
    587950636,  157623475,  1823632563, 2263217432,
    1568573771, 289745607,  703518081,  817188400,
    2086680194, 27010288,   3087364633, 431601683,
    3667110195, 1853857795, 2649569023, 189568803,
    2603418874, 1317779160, 180288151,  2452030208,
    1022843651, 904356825,  2189632558, 446256628
};

const uint32_t DataSet_1_32u::outputs::MMINV[32] = {
    2890127753, 3623131505, 3730078463, 2142934923,
    3878122330, 3765622899, 1453286364, 1634157816,
    587950636,  157623475,  1823632563, 3120724469,
    1568573771, 3435400464, 703518081,  817188400,
    3165700845, 27010288,   3087364633, 431601683,
    3667110195, 1853857795, 2691787100, 189568803,
    2603418874, 2189542274, 180288151,  2452030208,
    2748106285, 904356825,  2189632558, 446256628
};

const uint32_t DataSet_1_32u::outputs::MINS[32] = {
    636364, 636364, 636364, 636364,
    636364, 636364, 636364, 636364,
    636364, 636364, 636364, 636364,
    636364, 636364, 636364, 636364,
    636364, 636364, 636364, 636364,
    636364, 636364, 636364, 636364,
    636364, 636364, 636364, 636364,
    636364, 636364, 636364, 636364
};

const uint32_t DataSet_1_32u::outputs::MMINS[32] = {
    2890127753, 3623131505, 3730078463, 636364,
    3878122330, 636364,     636364,     1634157816,
    587950636,  636364,     636364,     3120724469,
    636364,     3435400464, 703518081,  636364,
    3165700845, 636364,     636364,     431601683,
    636364,     1853857795, 2691787100, 636364,
    636364,     2189542274, 180288151,  636364,
    2748106285, 636364,     636364,     446256628
};

const uint32_t DataSet_1_32u::outputs::HMAX[32] = {
    2890127753, 3623131505, 3730078463, 3730078463,
    3878122330, 4195248000, 4195248000, 4195248000,
    4195248000, 4195248000, 4195248000, 4195248000,
    4195248000, 4195248000, 4195248000, 4195248000,
    4195248000, 4195248000, 4195248000, 4195248000,
    4195248000, 4195248000, 4195248000, 4195248000,
    4195248000, 4195248000, 4195248000, 4195248000,
    4195248000, 4225759456, 4225759456, 4225759456
};

const uint32_t DataSet_1_32u::outputs::MHMAX[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

const uint32_t DataSet_1_32u::outputs::HMIN[32] = {
    2890127753, 2890127753, 2890127753, 2142934923,
    2142934923, 2142934923, 1453286364, 1453286364,
    587950636,  587950636,  587950636,  587950636,
    587950636,  587950636,  587950636,  587950636,
    587950636,  27010288,   27010288,   27010288,
    27010288,   27010288,   27010288,   27010288,
    27010288,   27010288,   27010288,   27010288,
    27010288,   27010288,   27010288,   27010288
};

const uint32_t DataSet_1_32u::outputs::MHMIN[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

const uint32_t DataSet_1_32u::outputs::SQR[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

const uint32_t DataSet_1_32u::outputs::MSQR[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

const uint32_t DataSet_1_32u::outputs::SQRT[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

const uint32_t DataSet_1_32u::outputs::MSQRT[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

const uint32_t DataSet_1_32u::outputs::LSHV[32] = {
    0x98900000,     0x7f499710,     0xe0000000,     0xa978b000,
    0x93b9ad00,     0xccf00000,     0xfb1ee000,     0x6744f800,
    0xda8b0000,     0xe3200000,     0x4d966000,     0x40000000,
    0xbf44a580,     0x40000000,     0xac080000,     0xb54e3000,
    0x185da000,     0x849e0000,     0xb90c8000,     0xb8130000,
    0x85000000,     0x40060000,     0x57000000,     0xe3761000,
    0x18000000,     0x80000000,     0xbea5c000,     0x00000000,
    0x998c5a00,     0xdff8e000,     0x70b00000,     0xafa00000
};
const uint32_t DataSet_1_32u::outputs::MLSHV[32] = {
    0xac43d989,     0xd7f49971,     0xde547aff,     0xa978b000,
    0xe727735a,     0xccf00000,     0xfb1ee000,     0x616744f8,
    0x230b6a2c,     0xe3200000,     0x4d966000,     0xba0279f5,
    0xbf44a580,     0xccc40d10,     0x29eed581,     0xb54e3000,
    0xbcb0c2ed,     0x849e0000,     0xb90c8000,     0x19b9b813,
    0x85000000,     0x6e7fa003,     0xa071695c,     0xe3761000,
    0x18000000,     0x8281c382,     0x0abefa97,     0x00000000,
    0xa3ccc62d,     0xdff8e000,     0x70b00000,     0x1a9955f4
};
const uint32_t DataSet_1_32u::outputs::LSHS[32] = {
    0x48000000,     0x88000000,     0xf8000000,     0x58000000,
    0xd0000000,     0x00000000,     0xe0000000,     0xc0000000,
    0x60000000,     0xc8000000,     0x98000000,     0xa8000000,
    0x58000000,     0x80000000,     0x08000000,     0x80000000,
    0x68000000,     0x80000000,     0xc8000000,     0x98000000,
    0xa0000000,     0x18000000,     0xe0000000,     0x08000000,
    0x30000000,     0x10000000,     0xb8000000,     0x00000000,
    0x68000000,     0x00000000,     0xb0000000,     0xa0000000
};
const uint32_t DataSet_1_32u::outputs::MLSHS[32] = {
    0xac43d989,     0xd7f49971,     0xde547aff,     0x58000000,
    0xe727735a,     0x00000000,     0xe0000000,     0x616744f8,
    0x230b6a2c,     0xc8000000,     0x98000000,     0xba0279f5,
    0x58000000,     0xccc40d10,     0x29eed581,     0x80000000,
    0xbcb0c2ed,     0x80000000,     0xc8000000,     0x19b9b813,
    0xa0000000,     0x6e7fa003,     0xa071695c,     0x08000000,
    0x30000000,     0x8281c382,     0x0abefa97,     0x00000000,
    0xa3ccc62d,     0x00000000,     0xb0000000,     0x1a9955f4
};
const uint32_t DataSet_1_32u::outputs::RSHV[32] = {
    0x00000ac4,     0x0d7f4997,     0x00000006,     0x0007fba9,
    0x01ce4ee6,     0x0007d073,     0x000ad3ec,     0x00616744,
    0x00008c2d,     0x00000625,     0x00036593,     0x00000002,
    0x00bafd12,     0x00000033,     0x0000053d,     0x0030b54e,
    0x0005e586,     0x00000ce1,     0x0001700a,     0x000019b9,
    0x000003c1,     0x0000373f,     0x00000281,     0x00033be3,
    0x00000030,     0x00000002,     0x00002afb,     0x00000049,
    0x0051e663,     0x00fbdff8,     0x00001e12,     0x00000353
};
const uint32_t DataSet_1_32u::outputs::MRSHV[32] = {
    0xac43d989,     0xd7f49971,     0xde547aff,     0x0007fba9,
    0xe727735a,     0x0007d073,     0x000ad3ec,     0x616744f8,
    0x230b6a2c,     0x00000625,     0x00036593,     0xba0279f5,
    0x00bafd12,     0xccc40d10,     0x29eed581,     0x0030b54e,
    0xbcb0c2ed,     0x00000ce1,     0x0001700a,     0x19b9b813,
    0x000003c1,     0x6e7fa003,     0xa071695c,     0x00033be3,
    0x00000030,     0x8281c382,     0x0abefa97,     0x00000049,
    0xa3ccc62d,     0x00fbdff8,     0x00001e12,     0x1a9955f4
};
const uint32_t DataSet_1_32u::outputs::RSHS[32] = {
    0x00000015,     0x0000001a,     0x0000001b,     0x0000000f,
    0x0000001c,     0x0000001f,     0x0000000a,     0x0000000c,
    0x00000004,     0x00000018,     0x0000000d,     0x00000017,
    0x0000000b,     0x00000019,     0x00000005,     0x00000006,
    0x00000017,     0x00000000,     0x00000017,     0x00000003,
    0x0000001e,     0x0000000d,     0x00000014,     0x00000006,
    0x00000018,     0x00000010,     0x00000001,     0x00000012,
    0x00000014,     0x0000001f,     0x0000001e,     0x00000003
};
const uint32_t DataSet_1_32u::outputs::MRSHS[32] = {
    0xac43d989,     0xd7f49971,     0xde547aff,     0x0000000f,
    0xe727735a,     0x0000001f,     0x0000000a,     0x616744f8,
    0x230b6a2c,     0x00000018,     0x0000000d,     0xba0279f5,
    0x0000000b,     0xccc40d10,     0x29eed581,     0x00000006,
    0xbcb0c2ed,     0x00000000,     0x00000017,     0x19b9b813,
    0x0000001e,     0x6e7fa003,     0xa071695c,     0x00000006,
    0x00000018,     0x8281c382,     0x0abefa97,     0x00000012,
    0xa3ccc62d,     0x0000001f,     0x0000001e,     0x1a9955f4
};
const uint32_t DataSet_1_32u::outputs::ROLV[32] = {
    0x989ac43d,     0x7f49971d,     0xfbca8f5f,     0xa978b7fb,
    0x93b9ad73,     0xccf01f41,     0xfb1ee2b4,     0x6744f861,
    0xda8b08c2,     0xe338962e,     0x4d966d96,     0x6e809e7d,
    0xbf44a5ae,     0x43331034,     0xac094f76,     0xb54e3030,
    0x185db796,     0x849e0033,     0xb90cdc02,     0xb81319b9,
    0x853c10d5,     0x4006dcff,     0x57281c5a,     0xe376133b,
    0x1b0a71b3,     0xa0a070e0,     0xbea5c2af,     0x01244e06,
    0x998c5b47,     0xdff8e0fb,     0x70b78482,     0xafa0d4ca
};
const uint32_t DataSet_1_32u::outputs::MROLV[32] = {
    0xac43d989,     0xd7f49971,     0xde547aff,     0xa978b7fb,
    0xe727735a,     0xccf01f41,     0xfb1ee2b4,     0x616744f8,
    0x230b6a2c,     0xe338962e,     0x4d966d96,     0xba0279f5,
    0xbf44a5ae,     0xccc40d10,     0x29eed581,     0xb54e3030,
    0xbcb0c2ed,     0x849e0033,     0xb90cdc02,     0x19b9b813,
    0x853c10d5,     0x6e7fa003,     0xa071695c,     0xe376133b,
    0x1b0a71b3,     0x8281c382,     0x0abefa97,     0x01244e06,
    0xa3ccc62d,     0xdff8e0fb,     0x70b78482,     0x1a9955f4
};
const uint32_t DataSet_1_32u::outputs::ROLS[32] = {
    0x4d621ecc,     0x8ebfa4cb,     0xfef2a3d7,     0x5bfdd4bc,
    0xd7393b9a,     0x07d0733c,     0xe2b4fb1e,     0xc30b3a27,
    0x61185b51,     0xce258bb8,     0x9b659365,     0xadd013cf,
    0x5aebf44a,     0x86662068,     0x094f76ac,     0x8185aa71,
    0x6de58617,     0x800ce127,     0xcdc02b90,     0x98cdcdc0,
    0xa7821ab0,     0x1b73fd00,     0xe5038b4a,     0x099df1bb,
    0x3614e366,     0x14140e1c,     0xb855f7d4,     0x04913818,
    0x6d1e6631,     0x07deffc7,     0xb7848270,     0xa0d4caaf
};
const uint32_t DataSet_1_32u::outputs::MROLS[32] = {
    0xac43d989,     0xd7f49971,     0xde547aff,     0x5bfdd4bc,
    0xe727735a,     0x07d0733c,     0xe2b4fb1e,     0x616744f8,
    0x230b6a2c,     0xce258bb8,     0x9b659365,     0xba0279f5,
    0x5aebf44a,     0xccc40d10,     0x29eed581,     0x8185aa71,
    0xbcb0c2ed,     0x800ce127,     0xcdc02b90,     0x19b9b813,
    0xa7821ab0,     0x6e7fa003,     0xa071695c,     0x099df1bb,
    0x3614e366,     0x8281c382,     0x0abefa97,     0x04913818,
    0xa3ccc62d,     0x07deffc7,     0xb7848270,     0x1a9955f4
};
const uint32_t DataSet_1_32u::outputs::RORV[32] = {
    0x3d989ac4,     0x1d7f4997,     0xf2a3d7fe,     0x78b7fba9,
    0xb5ce4ee6,     0x3c07d073,     0x7b8ad3ec,     0xf8616744,
    0xa8b08c2d,     0x8bb8ce25,     0x659b6593,     0xe809e7d6,
    0x96bafd12,     0x31034433,     0xdab0253d,     0x3030b54e,
    0x176de586,     0x27800ce1,     0xe433700a,     0xb81319b9,
    0x0d5853c1,     0xd001b73f,     0xc5a57281,     0x76133be3,
    0xa71b31b0,     0x0a070e0a,     0xea5c2afb,     0x13818049,
    0x16d1e663,     0xe0fbdff8,     0x09c2de12,     0x2abe8353
};
const uint32_t DataSet_1_32u::outputs::MRORV[32] = {
    0xac43d989,     0xd7f49971,     0xde547aff,     0x78b7fba9,
    0xe727735a,     0x3c07d073,     0x7b8ad3ec,     0x616744f8,
    0x230b6a2c,     0x8bb8ce25,     0x659b6593,     0xba0279f5,
    0x96bafd12,     0xccc40d10,     0x29eed581,     0x3030b54e,
    0xbcb0c2ed,     0x27800ce1,     0xe433700a,     0x19b9b813,
    0x0d5853c1,     0x6e7fa003,     0xa071695c,     0x76133be3,
    0xa71b31b0,     0x8281c382,     0x0abefa97,     0x13818049,
    0xa3ccc62d,     0xe0fbdff8,     0x09c2de12,     0x1a9955f4
};
const uint32_t DataSet_1_32u::outputs::RORS[32] = {
    0x887b3135,     0xfe932e3a,     0xca8f5ffb,     0xf752f16f,
    0xe4ee6b5c,     0x41ccf01f,     0xd3ec7b8a,     0x2ce89f0c,
    0x616d4584,     0x962ee338,     0x964d966d,     0x404f3eb7,
    0xafd1296b,     0x9881a219,     0x3ddab025,     0x16a9c606,
    0x96185db7,     0x33849e00,     0x00ae4337,     0x37370263,
    0x086ac29e,     0xcff4006d,     0x0e2d2b94,     0x77c6ec26,
    0x538d98d8,     0x50387050,     0x57df52e1,     0x44e06012,
    0x7998c5b4,     0x7bff1c1f,     0x1209c2de,     0x532abe83
};
const uint32_t DataSet_1_32u::outputs::MRORS[32] = {
    0xac43d989,     0xd7f49971,     0xde547aff,     0xf752f16f,
    0xe727735a,     0x41ccf01f,     0xd3ec7b8a,     0x616744f8,
    0x230b6a2c,     0x962ee338,     0x964d966d,     0xba0279f5,
    0xafd1296b,     0xccc40d10,     0x29eed581,     0x16a9c606,
    0xbcb0c2ed,     0x33849e00,     0x00ae4337,     0x19b9b813,
    0x086ac29e,     0x6e7fa003,     0xa071695c,     0x77c6ec26,
    0x538d98d8,     0x8281c382,     0x0abefa97,     0x44e06012,
    0xa3ccc62d,     0x7bff1c1f,     0x1209c2de,     0x1a9955f4
};

const int32_t DataSet_1_32u::outputs::UTOI[32] = {
    (int32_t)-1404839543,   (int32_t)-671835791,    (int32_t)-564888833,    (int32_t)2142934923,
    (int32_t)-416844966,    (int32_t)-99719296,     (int32_t)1453286364,    (int32_t)1634157816,
    (int32_t)587950636,     (int32_t)-995002599,    (int32_t)1823632563,    (int32_t)-1174242827,
    (int32_t)1568573771,    (int32_t)-859566832,    (int32_t)703518081,     (int32_t)817188400,
    (int32_t)-1129266451,   (int32_t)27010288,      (int32_t)-1207602663,   (int32_t)431601683,
    (int32_t)-264022508,    (int32_t)1853857795,    (int32_t)-1603180196,   (int32_t)868104033,
    (int32_t)-1029935930,   (int32_t)-2105425022,   (int32_t)180288151,     (int32_t)-1842937088,
    (int32_t)-1546861011,   (int32_t)-69207840,     (int32_t)-258978282,    (int32_t)446256628
};

const float DataSet_1_32u::outputs::UTOF[32] = {
    2890127872.000000000f, 3623131392.000000000f, 3730078464.000000000f, 2142934912.000000000f,
    3878122240.000000000f, 4195248128.000000000f, 1453286400.000000000f, 1634157824.000000000f,
    587950656.000000000f, 3299964672.000000000f, 1823632512.000000000f, 3120724480.000000000f,
    1568573824.000000000f, 3435400448.000000000f, 703518080.000000000f, 817188416.000000000f,
    3165700864.000000000f, 27010288.000000000f, 3087364608.000000000f, 431601696.000000000f,
    4030944768.000000000f, 1853857792.000000000f, 2691787008.000000000f, 868104064.000000000f,
    3265031424.000000000f, 2189542400.000000000f, 180288144.000000000f, 2452030208.000000000f,
    2748106240.000000000f, 4225759488.000000000f, 4035988992.000000000f, 446256640.000000000
};

const int32_t DataSet_1_32i::inputs::inputA[32] = {
    948484139,      325061806,      1092824755,     -301586865,
    -873226993,     -1181626453,    68035273,       -1934370231,
    -817803908,     1170504758,     1961487688,     -363273962,
    950418613,      -582437808,     1132387793,     -777060985,
    -1489706084,    -1949446102,    -368683059,     1400938468, 
    -1302030944,    1882764301,     2345500,        1205584894, 
    447369865,      2002189952,     1814442606,     -1252331378,
    -991493688,     1779725208,     620782193,      841973243
};

const int32_t DataSet_1_32i::inputs::inputB[32] = {
     -1887618805,   -1934674551,    2084427857,     -1915569657,
    1858735609,     -405360036,     -962338641,     -1119657200,
    897529765,      -100128913,     -106890800,     -1762804186,
    -1640197658,    -960604164,     -294752717,     -167644897,
    767853874,      1224769104,     -230940897,     -497969136,
    -1500185509,    -1813830137,    600068247,      1457801838,
    882484781,      1351720796,     -1236796659,    -1976020177,
    -1786532694,    -724637555,     258541995,      1192251394
};

const int32_t DataSet_1_32i::inputs::inputC[32] = {
    -1775224743,    -154858053,     777821817,      -468149921,
    822243362,      1119220373,     1488111679,     935142279,
    1590669966,     608031794,      971176515,      1105608663,
    -2094385941,    -192541271,     710475075,      -1712382093,
    1055367934,     2000079676,     -1315499548,    -265930722,
    1856542308,     -2110584009,    -406607791,     159329147,
    909065427,      255613429,      -972120948,     -1588088154,
    -357313165,     1966155567,     1819410331,     -1064864049
};

const uint32_t DataSet_1_32i::inputs::inputShiftA[32] = {
    52,     4,      29,     44,
    7,      13,     11,     8,
    46,     53,     13,     62,
    39,     26,     19,     40,
    45,     13,     47,     1008,
    694,    529,    790,    492,
    58,     702,    206,    761,
    201,    8,      851,    691
};

const int32_t DataSet_1_32i::inputs::scalarA = 234123148;
const uint32_t DataSet_1_32i::inputs::inputShiftScalarA = 27;

const bool    DataSet_1_32i::inputs::maskA[32] = {
    false,   false,  false,  true,   // 4
    false,  true,   true,   false,  // 8

    false,  true,   true,   false,  
    true,   false,  false,  true,   // 16
    
    false,  true,   true,   false,
    true,   false,  false,  true,
    true,   false,  false,  true,
    false,  true,   true,   false,  // 32
};

const int32_t DataSet_1_32i::outputs::ADDV[32] = {
    -939134666,     -1609612745,    -1117714684,    2077810774,
    985508616,      -1586986489,    -894303368,     1240939865,
    79725857,       1070375845,     1854596888,     -2126078148,
    -689779045,     -1543041972,    837635076,      -944705882,
    -721852210,     -724676998,     -599623956,     902969332,
    1492750843,     68934164,       602413747,      -1631580564,
    1329854646,     -941056548,     577645947,      1066615741,
    1516940914,     1055087653,     879324188,      2034224637
};
const int32_t DataSet_1_32i::outputs::MADDV[32] = {
    948484139,      325061806,      1092824755,     2077810774,
    -873226993,     -1586986489,    -894303368,     -1934370231,
    -817803908,     1070375845,     1854596888,     -363273962,
    -689779045,     -582437808,     1132387793,     -944705882,
    -1489706084,    -724676998,     -599623956,     1400938468,
    1492750843,     1882764301,     2345500,        -1631580564,
    1329854646,     2002189952,     1814442606,     1066615741,
    -991493688,     1055087653,     879324188,      841973243
};
const int32_t DataSet_1_32i::outputs::ADDS[32] = {
    1182607287,     559184954,      1326947903,     -67463717,
    -639103845,     -947503305,     302158421,      -1700247083,
    -583680760,     1404627906,     -2099356460,    -129150814,
    1184541761,     -348314660,     1366510941,     -542937837,
    -1255582936,    -1715322954,    -134559911,     1635061616,
    -1067907796,    2116887449,     236468648,      1439708042,
    681493013,      -2058654196,    2048565754,     -1018208230,
    -757370540,     2013848356,     854905341,      1076096391
};
const int32_t DataSet_1_32i::outputs::MADDS[32] = {
    948484139,      325061806,      1092824755,     -67463717,
    -873226993,     -947503305,     302158421,      -1934370231,
    -817803908,     1404627906,     -2099356460,    -363273962,
    1184541761,     -582437808,     1132387793,     -542937837,
    -1489706084,    -1715322954,    -134559911,     1400938468,
    -1067907796,    1882764301,     2345500,        1439708042,
    681493013,      2002189952,     1814442606,     -1018208230,
    -991493688,     2013848356,     854905341,      841973243
};
const int32_t DataSet_1_32i::outputs::POSTPREFINC[32] = {
    948484140,      325061807,      1092824756,     -301586864,
    -873226992,     -1181626452,    68035274,       -1934370230,
    -817803907,     1170504759,     1961487689,     -363273961,
    950418614,      -582437807,     1132387794,     -777060984,
    -1489706083,    -1949446101,    -368683058,     1400938469,
    -1302030943,    1882764302,     2345501,        1205584895,
    447369866,      2002189953,     1814442607,     -1252331377,
    -991493687,     1779725209,     620782194,      841973244
};
const int32_t DataSet_1_32i::outputs::MPOSTPREFINC[32] = {
    948484139,      325061806,      1092824755,     -301586864,
    -873226993,     -1181626452,    68035274,       -1934370231,
    -817803908,     1170504759,     1961487689,     -363273962,
    950418614,      -582437808,     1132387793,     -777060984,
    -1489706084,    -1949446101,    -368683058,     1400938468,
    -1302030943,    1882764301,     2345500,        1205584895,
    447369866,      2002189952,     1814442606,     -1252331377,
    -991493688,     1779725209,     620782194,      841973243,
};
const int32_t DataSet_1_32i::outputs::SUBV[32] = {
    -1458864352,    -2035230939,    -991603102,     1613982792,
    1563004694,     -776266417,     1030373914,     -814713031,
    -1715333673,    1270633671,     2068378488,     1399530224,
    -1704351025,    378166356,      1427140510,     -609416088,
    2037407338,     1120752090,     -137742162,     1898907604,
    198154565,      -598372858,     -597722747,     -252216944,
    -435114916,     650469156,      -1243728031,    723688799,
    795039006,      -1790604533,    362240198,      -350278151
};
const int32_t DataSet_1_32i::outputs::MSUBV[32] = {
    948484139,      325061806,      1092824755,     1613982792,
    -873226993,     -776266417,     1030373914,     -1934370231,
    -817803908,     1270633671,     2068378488,     -363273962,
    -1704351025,    -582437808,     1132387793,     -609416088,
    -1489706084,    1120752090,     -137742162,     1400938468,
    198154565,      1882764301,     2345500,        -252216944,
    -435114916,     2002189952,     1814442606,     723688799,
    -991493688,     -1790604533,    362240198,      841973243
};
const int32_t DataSet_1_32i::outputs::SUBS[32] = {
    714360991,      90938658,       858701607,      -535710013,
    -1107350141,    -1415749601,    -166087875,     2126473917,
    -1051927056,    936381610,      1727364540,     -597397110,
    716295465,      -816560956,     898264645,      -1011184133,
    -1723829232,    2111398046,     -602806207,     1166815320,
    -1536154092,    1648641153,     -231777648,     971461746,
    213246717,      1768066804,     1580319458,     -1486454526,
    -1225616836,    1545602060,     386659045,      607850095
};
const int32_t DataSet_1_32i::outputs::MSUBS[32] = {
    948484139,      325061806,      1092824755,     -535710013,
    -873226993,     -1415749601,    -166087875,     -1934370231,
    -817803908,     936381610,      1727364540,     -363273962,
    716295465,      -582437808,     1132387793,     -1011184133,
    -1489706084,    2111398046,     -602806207,     1400938468,
    -1536154092,    1882764301,     2345500,        971461746,
    213246717,      2002189952,     1814442606,     -1486454526,
    -991493688,     1545602060,     386659045,      841973243
};
const int32_t DataSet_1_32i::outputs::SUBFROMV[32] = {
    1458864352,     2035230939,     991603102,      -1613982792,
    -1563004694,    776266417,      -1030373914,    814713031,
    1715333673,     -1270633671,    -2068378488,    -1399530224,
    1704351025,     -378166356,     -1427140510,    609416088,
    -2037407338,    -1120752090,    137742162,      -1898907604,
    -198154565,     598372858,      597722747,      252216944,
    435114916,      -650469156,     1243728031,     -723688799,
    -795039006,     1790604533,     -362240198,     350278151
};
const int32_t DataSet_1_32i::outputs::MSUBFROMV[32] = {
    -1887618805,    -1934674551,    2084427857,     -1613982792,
    1858735609,     776266417,      -1030373914,    -1119657200,
    897529765,      -1270633671,    -2068378488,    -1762804186,
    1704351025,     -960604164,     -294752717,     609416088,
    767853874,      -1120752090,    137742162,      -497969136,
    -198154565,     -1813830137,    600068247,      252216944,
    435114916,      1351720796,     -1236796659,    -723688799,
    -1786532694,    1790604533,     -362240198,     1192251394
};
const int32_t DataSet_1_32i::outputs::SUBFROMS[32] = {
    -714360991,     -90938658,      -858701607,     535710013,
    1107350141,     1415749601,     166087875,      -2126473917,
    1051927056,     -936381610,     -1727364540,    597397110,
    -716295465,     816560956,      -898264645,     1011184133,
    1723829232,     -2111398046,    602806207,      -1166815320,
    1536154092,     -1648641153,    231777648,      -971461746,
    -213246717,     -1768066804,    -1580319458,    1486454526,
    1225616836,     -1545602060,    -386659045,     -607850095
};
const int32_t DataSet_1_32i::outputs::MSUBFROMS[32] = {
    234123148,      234123148,      234123148,      535710013,
    234123148,      1415749601,     166087875,      234123148,
    234123148,      -936381610,     -1727364540,    234123148,
    -716295465,     234123148,      234123148,      1011184133,
    234123148,      -2111398046,    602806207,      234123148,
    1536154092,     234123148,      234123148,      -971461746,
    -213246717,     234123148,      234123148,      1486454526,
    234123148,      -1545602060,    -386659045,     234123148
};
const int32_t DataSet_1_32i::outputs::POSTPREFDEC[32] = {
    948484138,      325061805,      1092824754,     -301586866,
    -873226994,     -1181626454,    68035272,       -1934370232,
    -817803909,     1170504757,     1961487687,     -363273963,
    950418612,      -582437809,     1132387792,     -777060986,
    -1489706085,    -1949446103,    -368683060,     1400938467,
    -1302030945,    1882764300,     2345499,        1205584893,
    447369864,      2002189951,     1814442605,     -1252331379,
    -991493689,     1779725207,     620782192,      841973242
};
const int32_t DataSet_1_32i::outputs::MPOSTPREFDEC[32] = {
    948484139,      325061806,      1092824755,     -301586866,
    -873226993,     -1181626454,    68035272,       -1934370231,
    -817803908,     1170504757,     1961487687,     -363273962,
    950418612,      -582437808,     1132387793,     -777060986,
    -1489706084,    -1949446103,    -368683060,     1400938468,
    -1302030945,    1882764301,     2345500,        1205584893,
    447369864,      2002189952,     1814442606,     -1252331379,
    -991493688,     1779725207,     620782192,      841973243
};
const int32_t DataSet_1_32i::outputs::MULV[32] = {
    -790594343,     335073054,      540004003,      -372346327,
    516130455,      -121796748,     -1072388249,    1589829520,
    -1186949908,    -858909334,     -623127936,     94223684,
    -30863458,      1010347712,     -1822163293,    -664548775,
    -771456904,     -154814176,     2037544147,     1981636160,
    -1383419936,    -1697785253,    -709560700,     1229824804,
    1230395413,     299169280,      568617366,      -222284782,
    1207013072,     -631936328,     -1173308805,    -1200788490
};
const int32_t DataSet_1_32i::outputs::MMULV[32] = {
    948484139,      325061806,      1092824755,     -372346327,
    -873226993,     -121796748,     -1072388249,    -1934370231,
    -817803908,     -858909334,     -623127936,     -363273962,
    -30863458,      -582437808,     1132387793,     -664548775,
    -1489706084,    -154814176,     2037544147,     1400938468,
    -1383419936,    1882764301,     2345500,        1229824804,
    1230395413,     2002189952,     1814442606,     -222284782,
    -991493688,     -631936328,     -1173308805,    841973243
};
const int32_t DataSet_1_32i::outputs::MULS[32] = {
    -358916988,     702439720,      2048468708,     1548039220,
    -1616943820,    1388623492,     -1251726100,    -1332410644,
    -830528560,     2059884424,     779090784,      1496031752,
    482323964,      1982942144,     -205329076,     189542100,
    1011686736,     1004014840,     -739383524,     1761299632,
    -396217472,     980949532,      -1494963376,    1245860072,
    243436012,      233614848,      1536771624,     425795496,
    -1296387744,    -970055904,     -140978996,     -606815676
};
const int32_t DataSet_1_32i::outputs::MMULS[32] = {
    948484139,      325061806,      1092824755,     1548039220,
    -873226993,     1388623492,     -1251726100,    -1934370231,
    -817803908,     2059884424,     779090784,      -363273962,
    482323964,      -582437808,     1132387793,     189542100,
    -1489706084,    1004014840,     -739383524,     1400938468,
    -396217472,     1882764301,     2345500,        1245860072,
    243436012,      2002189952,     1814442606,     425795496,
    -991493688,     -970055904,     -140978996,     841973243
};
const int32_t DataSet_1_32i::outputs::DIVV[32] = {
    0,      0,      0,      0,  0,      2,      0,      1,
    0,      -11,    -18,    0,  0,      0,      -3,     4,
    -1,     -1,     1,      -2, 0,      -1,     0,      0,
    0,      1,      -1,     0,  0,      -2,     2,      0
};
const int32_t DataSet_1_32i::outputs::MDIVV[32] = {
    948484139,      325061806,      1092824755,     0,
    -873226993,     2,              0,              -1934370231,
    -817803908,     -11,            -18,            -363273962,
    0,              -582437808,     1132387793,     4,
    -1489706084,    -1,             1,              1400938468,
    0,              1882764301,     2345500,        0,
    0,              2002189952,     1814442606,     0,
    -991493688,     -2,             2,              841973243
};
const int32_t DataSet_1_32i::outputs::DIVS[32] = {
    4,      1,      4,      -1,     -3,     -5,     0,      -8,
    -3,     4,      8,      -1,     4,      -2,     4,      -3,
    -6,     -8,     -1,     5,      -5,     8,      0,      5,
    1,      8,      7,      -5,     -4,     7,      2,      3
};
const int32_t DataSet_1_32i::outputs::MDIVS[32] = {
    948484139,      325061806,      1092824755,     -1,
    -873226993,     -5,             0,              -1934370231,
    -817803908,     4,              8,              -363273962,
    4,              -582437808,     1132387793,     -3,
    -1489706084,    -8,             -1,             1400938468,
    -5,             1882764301,     2345500,        5,
    1,              2002189952,     1814442606,     -5,
    -991493688,     7,              2,              841973243
};
const int32_t DataSet_1_32i::outputs::RCP[32] = {
    0,      0,      0,      0,      0,      0,      0,      0,
    0,      0,      0,      0,      0,      0,      0,      0,
    0,      0,      0,      0,      0,      0,      0,      0,
    0,      0,      0,      0,      0,      0,      0,      0
};
const int32_t DataSet_1_32i::outputs::MRCP[32] = {
    948484139,      325061806,      1092824755,     0,
    -873226993,     0,              0,              -1934370231,
    -817803908,     0,              0,              -363273962,
    0,              -582437808,     1132387793,     0,
    -1489706084,    0,              0,              1400938468,
    0,              1882764301,     2345500,        0,
    0,              2002189952,     1814442606,     0,
    -991493688,     0,              0,              841973243
};
const int32_t DataSet_1_32i::outputs::RCPS[32] = {
    0,      0,      0,      0,
    0,      0,      3,      0,
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      99,     0,
    0,      0,      0,      0,
    0,      0,      0,      0
};
const int32_t DataSet_1_32i::outputs::MRCPS[32] = {
    948484139,      325061806,      1092824755,     0,
    -873226993,     0,              3,              -1934370231,
    -817803908,     0,              0,              -363273962,
    0,              -582437808,     1132387793,     0,
    -1489706084,    0,              0,              1400938468,
    0,              1882764301,     2345500,        0,
    0,              2002189952,     1814442606,     0,
    -991493688,     0,              0,              841973243
};
const bool  DataSet_1_32i::outputs::CMPEQV[32] = {
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false
};
const bool  DataSet_1_32i::outputs::CMPEQS[32] = {
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false,
    false,  false,  false,  false
};
const bool  DataSet_1_32i::outputs::CMPNEV[32] = {
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true
};
const bool  DataSet_1_32i::outputs::CMPNES[32] = {
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true,
    true,   true,   true,   true
};
const bool  DataSet_1_32i::outputs::CMPGTV[32] = {
    true,   true,   false,  true,
    false,  false,  true,   false,
    false,  true,   true,   true,
    true,   true,   true,   false,
    false,  false,  false,  true,
    true,   true,   false,  false,
    false,  true,   true,   true,
    true,   true,   true,   false
};
const bool  DataSet_1_32i::outputs::CMPGTS[32] = {
    true,   true,   true,   false,
    false,  false,  false,  false,
    false,  true,   true,   false,
    true,   false,  true,   false,
    false,  false,  false,  true,
    false,  true,   false,  true,
    true,   true,   true,   false,
    false,  true,   true,   true
};
const bool  DataSet_1_32i::outputs::CMPLTV[32] = {
    false,  false,  true,   false,
    true,   true,   false,  true,
    true,   false,  false,  false,
    false,  false,  false,  true,
    true,   true,   true,   false,
    false,  false,  true,   true,
    true,   false,  false,  false,
    false,  false,  false,  true
};
const bool  DataSet_1_32i::outputs::CMPLTS[32] = {
    false,  false,  false,  true,
    true,   true,   true,   true,
    true,   false,  false,  true,
    false,  true,   false,  true,
    true,   true,   true,   false,
    true,   false,  true,   false,
    false,  false,  false,  true,
    true,   false,  false,  false
};
const bool  DataSet_1_32i::outputs::CMPGEV[32] = {
    true,   true,   false,  true,
    false,  false,  true,   false,
    false,  true,   true,   true,
    true,   true,   true,   false,
    false,  false,  false,  true,
    true,   true,   false,  false,
    false,  true,   true,   true,
    true,   true,   true,   false
};
const bool  DataSet_1_32i::outputs::CMPGES[32] = {
    true,   true,   true,   false,
    false,  false,  false,  false,
    false,  true,   true,   false,
    true,   false,  true,   false,
    false,  false,  false,  true,
    false,  true,   false,  true,
    true,   true,   true,   false,
    false,  true,   true,   true
};
const bool  DataSet_1_32i::outputs::CMPLEV[32] = {
    false,  false,  true,   false,
    true,   true,   false,  true,
    true,   false,  false,  false,
    false,  false,  false,  true,
    true,   true,   true,   false,
    false,  false,  true,   true,
    true,   false,  false,  false,
    false,  false,  false,  true
};
const bool  DataSet_1_32i::outputs::CMPLES[32] = {
    false,  false,  false,  true,
    true,   true,   true,   true,
    true,   false,  false,  true,
    false,  true,   false,  true,
    true,   true,   true,   false,
    true,   false,  true,   false,
    false,  false,  false,  true,
    true,   false,  false,  false
};

const bool  DataSet_1_32i::outputs::CMPEV = false;
const bool  DataSet_1_32i::outputs::CMPES = false;

const int32_t DataSet_1_32i::outputs::HADD[32] = {
    948484139,      1273545945,     -1928596596,    2064783835,
    1191556842,     9930389,        77965662,       -1856404569,
    1620758819,     -1503703719,    457783969,      94510007,
    1044928620,     462490812,      1594878605,     817817620,
    -671888464,     1673632730,     1304949671,     -1589079157,
    1403857195,     -1008345800,    -1006000300,    199584594,
    646954459,      -1645822885,    168619721,      -1083711657,
    -2075205345,    -295480137,     325302056,      1167275299
};
const int32_t DataSet_1_32i::outputs::MHADD[32] = {
    0,              0,              0,              -301586865,
    -301586865,     -1483213318,    -1415178045,    -1415178045,
    -1415178045,    -244673287,     1716814401,     1716814401,
    -1627734282,    -1627734282,    -1627734282,    1890172029,
    1890172029,     -59274073,      -427957132,     -427957132,
    -1729988076,    -1729988076,    -1729988076,    -524403182,
    -77033317,      -77033317,      -77033317,      -1329364695,
    -1329364695,    450360513,      1071142706,     1071142706
};
const int32_t DataSet_1_32i::outputs::HMUL[32] = {
    948484139,      380580154,      -867838066,     2027985106,
    1748483662,     -666161126,     -656918422,     1607490106,
    -490133992,     -415127280,     -208970624,     910586624,
    1958956800,     143962112,      -44847104,      -334376960,
    -431308800,     1145274368,     -1702002688,    -2004484096,
    1371537408,     650117120,      -1124073472,    -2046820352,
    -1241513984,    0,              0,              0,
    0,              0,              0,              0

};
const int32_t DataSet_1_32i::outputs::MHMUL[32] = {
    1,              1,              1,              -301586865,
    -301586865,     -1059817531,    1065366445,     1065366445,
    1065366445,     -1350126978,    -1647388304,    -1647388304,
    -1932133328,    -1932133328,    -1932133328,    1471376720,
    1471376720,     626061088,      1037261984,     1037261984,
    394462208,      394462208,      394462208,      -1945503744,
    -758794240,     -758794240,     -758794240,     -164884480,
    -164884480,     895057920,      -813596672,     -813596672
};

const int32_t DataSet_1_32i::outputs::FMULADDV[32] = {
    1729148210,     180215001,      1317825820,     -840496248,
    1338373817,     997423625,      415723430,      -1769995497,
    403720058,      -250877540,     348048579,      1199832347,
    -2125249399,    817806441,      -1111688218,    1918036428,
    283911030,      1845265500,     722044599,      1715705438,
    473122372,      486598034,      -1116168491,    1389153951,
    2139460840,     554782709,      -403503582,     -1810372936,
    849699907,      1334219239,     646101526,      2029314757
};
const int32_t DataSet_1_32i::outputs::MFMULADDV[32] = {
    948484139,      325061806,      1092824755,     -840496248,
    -873226993,     997423625,      415723430,      -1934370231,
    -817803908,     -250877540,     348048579,      -363273962,
    -2125249399,    -582437808,     1132387793,     1918036428,
    -1489706084,    1845265500,     722044599,      1400938468,
    473122372,      1882764301,     2345500,        1389153951,
    2139460840,     2002189952,     1814442606,     -1810372936,
    -991493688,     1334219239,     646101526,      841973243
};
const int32_t DataSet_1_32i::outputs::FMULSUBV[32] = {
    984630400,      489931107,      -237817814,     95803594,
    -306112907,     -1241017121,    1734467368,     654687241,
    1517347422,     -1466941128,    -1594304451,    -1011384979,
    2063522483,     1202888983,     1762328928,     1047833318,
    -1826824838,    2140073444,     -941923601,     -2047400414,
    1055005052,     412798756,      -302952909,     1070495657,
    321329986,      43555851,       1540738314,     1365803372,
    1564326237,     1696875401,     1302248160,     -135924441

};
const int32_t DataSet_1_32i::outputs::MFMULSUBV[32] = {
    948484139,      325061806,      1092824755,     95803594,
    -873226993,     -1241017121,    1734467368,     -1934370231,
    -817803908,     -1466941128,    -1594304451,    -363273962,
    2063522483,     -582437808,     1132387793,     1047833318,
    -1489706084,    2140073444,     -941923601,     1400938468,
    1055005052,     1882764301,     2345500,        1070495657,
    321329986,      2002189952,     1814442606,     1365803372,
    -991493688,     1696875401,     1302248160,     841973243
};
const int32_t DataSet_1_32i::outputs::FADDMULV[32] = {
    -1259033658,    2144676141,     -1566521116,    866035178,
    -1508624624,    2041662483,     -1644914040,    947487983,
    1124715086,     533568058,      -281547960,     1063747428,
    -1969675191,    -2079393236,    1587206412,     609963666,
    -1716889500,    -1135533416,    -431253968,     1494740632,
    -1547823092,    -1370048948,    -1991316829,    2085032420,
    1354409474,     336629644,      -1766730940,    -2133698930,
    -1899546826,    -1159950901,    1189107956,     -738906733
};
const int32_t DataSet_1_32i::outputs::MFADDMULV[32] = {
    948484139,      325061806,      1092824755,     866035178,
    -873226993,     2041662483,     -1644914040,    -1934370231,
    -817803908,     533568058,      -281547960,     -363273962,
    -1969675191,    -582437808,     1132387793,     609963666,
    -1489706084,    -1135533416,    -431253968,     1400938468,
    -1547823092,    1882764301,     2345500,        2085032420,
    1354409474,     2002189952,     1814442606,     -2133698930,
    -991493688,     -1159950901,    1189107956,     841973243
};
const int32_t DataSet_1_32i::outputs::FSUBMULV[32] = {
    -556912096,     2073233671,     1122914386,     -1249102152,
    -498425620,     175619323,      115250278,      145795599,
    -676291774,     -261596450,     701163624,      -1070372976,
    682088453,      1275017076,     -873702310,     1616194232,
    1994572076,     302189848,      -629143304,     1991330520,
    2083409140,     1521337674,     -180110571,     -1228761040,
    -1794662444,    560435572,      -325884660,     1143413146,
    -546622854,     -1542573051,    -1955202590,    2075521623
};
const int32_t DataSet_1_32i::outputs::MFSUBMULV[32] = {
    948484139,      325061806,      1092824755,     -1249102152,
    -873226993,     175619323,      115250278,      -1934370231,
    -817803908,     -261596450,     701163624,      -363273962,
    682088453,      -582437808,     1132387793,     1616194232,
    -1489706084,    302189848,      -629143304,     1400938468,
    2083409140,     1882764301,     2345500,        -1228761040,
    -1794662444,    2002189952,     1814442606,     1143413146,
    -991493688,     -1542573051,    -1955202590,    841973243
};
const int32_t DataSet_1_32i::outputs::MAXV[32] = {
    948484139,      325061806,      2084427857,     -301586865,
    1858735609,     -405360036,     68035273,       -1119657200,
    897529765,      1170504758,     1961487688,     -363273962,
    950418613,      -582437808,     1132387793,     -167644897,
    767853874,      1224769104,     -230940897,     1400938468,
    -1302030944,    1882764301,     600068247,      1457801838,
    882484781,      2002189952,     1814442606,     -1252331378,
    -991493688,     1779725208,     620782193,      1192251394
};
const int32_t DataSet_1_32i::outputs::MMAXV[32] = {
    948484139,      325061806,      1092824755,     -301586865,
    -873226993,     -405360036,     68035273,       -1934370231,
    -817803908,     1170504758,     1961487688,     -363273962,
    950418613,      -582437808,     1132387793,     -167644897,
    -1489706084,    1224769104,     -230940897,     1400938468,
    -1302030944,    1882764301,     2345500,        1457801838,
    882484781,      2002189952,     1814442606,     -1252331378,
    -991493688,     1779725208,     620782193,      841973243
};
const int32_t DataSet_1_32i::outputs::MAXS[32] = {
    948484139,      325061806,      1092824755,     234123148,
    234123148,      234123148,      234123148,      234123148,
    234123148,      1170504758,     1961487688,     234123148,
    950418613,      234123148,      1132387793,     234123148,
    234123148,      234123148,      234123148,      1400938468,
    234123148,      1882764301,     234123148,      1205584894,
    447369865,      2002189952,     1814442606,     234123148,
    234123148,      1779725208,     620782193,      841973243
};
const int32_t DataSet_1_32i::outputs::MMAXS[32] = {
    948484139,      325061806,      1092824755,     234123148,
    -873226993,     234123148,      234123148,      -1934370231,
    -817803908,     1170504758,     1961487688,     -363273962,
    950418613,      -582437808,     1132387793,     234123148,
    -1489706084,    234123148,      234123148,      1400938468,
    234123148,      1882764301,     2345500,        1205584894,
    447369865,      2002189952,     1814442606,     234123148,
    -991493688,     1779725208,     620782193,      841973243
};
const int32_t DataSet_1_32i::outputs::MINV[32] = {
    -1887618805,    -1934674551,    1092824755,     -1915569657,
    -873226993,     -1181626453,    -962338641,     -1934370231,
    -817803908,     -100128913,     -106890800,     -1762804186,
    -1640197658,    -960604164,     -294752717,     -777060985,
    -1489706084,    -1949446102,    -368683059,     -497969136,
    -1500185509,    -1813830137,    2345500,        1205584894,
    447369865,      1351720796,     -1236796659,    -1976020177,
    -1786532694,    -724637555,     258541995,      841973243
};
const int32_t DataSet_1_32i::outputs::MMINV[32] = {
    948484139,      325061806,      1092824755,     -1915569657,
    -873226993,     -1181626453,    -962338641,     -1934370231,
    -817803908,     -100128913,     -106890800,     -363273962,
    -1640197658,    -582437808,     1132387793,     -777060985,
    -1489706084,    -1949446102,    -368683059,     1400938468,
    -1500185509,    1882764301,     2345500,        1205584894,
    447369865,      2002189952,     1814442606,     -1976020177,
    -991493688,     -724637555,     258541995,      841973243
};
const int32_t DataSet_1_32i::outputs::MINS[32] = {
    234123148,      234123148,      234123148,      -301586865,
    -873226993,     -1181626453,    68035273,       -1934370231,
    -817803908,     234123148,      234123148,      -363273962,
    234123148,      -582437808,     234123148,      -777060985,
    -1489706084,    -1949446102,    -368683059,     234123148,
    -1302030944,    234123148,      2345500,        234123148,
    234123148,      234123148,      234123148,      -1252331378,
    -991493688,     234123148,      234123148,      234123148
};
const int32_t DataSet_1_32i::outputs::MMINS[32] = {
    948484139,      325061806,      1092824755,     -301586865,
    -873226993,     -1181626453,    68035273,       -1934370231,
    -817803908,     234123148,      234123148,      -363273962,
    234123148,      -582437808,     1132387793,     -777060985,
    -1489706084,    -1949446102,    -368683059,     1400938468,
    -1302030944,    1882764301,     2345500,        234123148,
    234123148,      2002189952,     1814442606,     -1252331378,
    -991493688,     234123148,      234123148,      841973243
};
const int32_t DataSet_1_32i::outputs::HMAX[32] = {
    948484139,      948484139,      1092824755,     1092824755,
    1092824755,     1092824755,     1092824755,     1092824755,
    1092824755,     1170504758,     1961487688,     1961487688,
    1961487688,     1961487688,     1961487688,     1961487688,
    1961487688,     1961487688,     1961487688,     1961487688,
    1961487688,     1961487688,     1961487688,     1961487688,
    1961487688,     2002189952,     2002189952,     2002189952,
    2002189952,     2002189952,     2002189952,     2002189952
};
const int32_t DataSet_1_32i::outputs::MHMAX[32] = {
    -2147483647,    -2147483647,    -2147483647,    -301586865,
    -301586865,     -301586865,     68035273,       68035273,
    68035273,       1170504758,     1961487688,     1961487688,
    1961487688,     1961487688,     1961487688,     1961487688,
    1961487688,     1961487688,     1961487688,     1961487688,
    1961487688,     1961487688,     1961487688,     1961487688,
    1961487688,     1961487688,     1961487688,     1961487688,
    1961487688,     1961487688,     1961487688,     1961487688
};
const int32_t DataSet_1_32i::outputs::HMIN[32] = {
    948484139,      325061806,      325061806,      -301586865,
    -873226993,     -1181626453,    -1181626453,    -1934370231,
    -1934370231,    -1934370231,    -1934370231,    -1934370231,
    -1934370231,    -1934370231,    -1934370231,    -1934370231,
    -1934370231,    -1949446102,    -1949446102,    -1949446102,
    -1949446102,    -1949446102,    -1949446102,    -1949446102,
    -1949446102,    -1949446102,    -1949446102,    -1949446102,
    -1949446102,    -1949446102,    -1949446102,    -1949446102
};
const int32_t DataSet_1_32i::outputs::MHMIN[32] = {
    2147483647,     2147483647,     2147483647,     -301586865,
    -301586865,     -1181626453,    -1181626453,    -1181626453,
    -1181626453,    -1181626453,    -1181626453,    -1181626453,
    -1181626453,    -1181626453,    -1181626453,    -1181626453,
    -1181626453,    -1949446102,    -1949446102,    -1949446102,
    -1949446102,    -1949446102,    -1949446102,    -1949446102,
    -1949446102,    -1949446102,    -1949446102,    -1949446102,
    -1949446102,    -1949446102,    -1949446102,    -1949446102
};
const int32_t DataSet_1_32i::outputs::SQR[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};
const int32_t DataSet_1_32i::outputs::MSQR[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

const int32_t DataSet_1_32i::outputs::BANDV[32] = {
    134756363,      2098312,        1075906577,     -1946017273,
    1254231305,     -1584365048,    67248265,       -1945943552,
    88147236,       1073751078,     1889593664,     -2109169658,
    405016740,      -1006103472,    1114524689,     -805244665,
    621052176,      151015424,      -503048435,     1107335168,
    -1576758272,    270541829,      16404,          1187005038,
    277348873,      1343266304,     604385804,      -2145890290,
    -2071789432,    1074028680,     83886113,       33574914
};
const int32_t DataSet_1_32i::outputs::MBANDV[32] = {
    948484139,      325061806,      1092824755,     -1946017273,
    -873226993,     -1584365048,    67248265,       -1934370231,
    -817803908,     1073751078,     1889593664,     -363273962,
    405016740,      -582437808,     1132387793,     -805244665,
    -1489706084,    151015424,      -503048435,     1400938468,
    -1576758272,    1882764301,     2345500,        1187005038,
    277348873,      2002189952,     1814442606,     -2145890290,
    -991493688,     1074028680,     83886113,       841973243
};
const int32_t DataSet_1_32i::outputs::BANDS[32] = {
    142616584,      23071884,       18886272,       201598476,
    166726924,      160452488,      67379848,       212879880,
    222317836,      96758788,       81816840,       139485444,
    144977028,      222307328,      24399232,       27553156,
    87319436,       163856392,      134498188,      25169796,
    6555008,        3154956,        2116108,        97534860,
    144720520,      89419392,       203694604,      89153676,
    82051464,       135553928,      83911168,       2387336
};
const int32_t DataSet_1_32i::outputs::MBANDS[32] = {
    948484139,      325061806,      1092824755,     201598476,
    -873226993,     160452488,      67379848,       -1934370231,
    -817803908,     96758788,       81816840,       -363273962,
    144977028,      -582437808,     1132387793,     27553156,
    -1489706084,    163856392,      134498188,      1400938468,
    6555008,        1882764301,     2345500,        97534860,
    144720520,      2002189952,     1814442606,     89153676,
    -991493688,     135553928,      83911168,       841973243
};
const int32_t DataSet_1_32i::outputs::BORV[32] = {
    -1073891029,    -1611711057,    2101346035,     -271139249,
    -268722689,     -2621441,       -961551633,     -1108083879,
    -8421379,       -3375233,       -34996776,      -16908490,
    -1094795785,    -536938500,     -276889613,     -139461217,
    -1342904386,    -875692422,     -96575521,      -204365836,
    -1225458181,    -201607665,     602397343,      1476381694,
    1052505773,     2010644444,     -26739857,      -1082461265,
    -706236950,     -18941027,      795438075,      2000649723
};
const int32_t DataSet_1_32i::outputs::MBORV[32] = {
    948484139,      325061806,      1092824755,     -271139249,
    -873226993,     -2621441,       -961551633,     -1934370231,
    -817803908,     -3375233,       -34996776,      -363273962,
    -1094795785,    -582437808,     1132387793,     -139461217,
    -1489706084,    -875692422,     -96575521,      1400938468,
    -1225458181,    1882764301,     2345500,        1476381694,
    1052505773,     2002189952,     1814442606,     -1082461265,
    -991493688,     -18941027,      795438075,      841973243
};
const int32_t DataSet_1_32i::outputs::BORS[32] = {
    1039990703,     536113070,      1308061631,     -269062193,
    -805830769,     -1107955793,    234778573,      -1913126963,
    -805998596,     1307869118,     2113793996,     -268636258,
    1039564733,     -570621988,     1342111709,     -570490993,
    -1342902372,    -1879179346,    -269058099,     1609891820,
    -1074462804,    2113732493,     234352540,      1342173182,
    536772493,      2146893708,     1844871150,     -1107361906,
    -839422004,     1878294428,     770994173,      1073709055
};
const int32_t DataSet_1_32i::outputs::MBORS[32] = {
    948484139,      325061806,      1092824755,     -269062193,
    -873226993,     -1107955793,    234778573,      -1934370231,
    -817803908,     1307869118,     2113793996,     -363273962,
    1039564733,     -582437808,     1132387793,     -570490993,
    -1489706084,    -1879179346,    -269058099,     1400938468,
    -1074462804,    1882764301,     2345500,        1342173182,
    536772493,      2002189952,     1814442606,     -1107361906,
    -991493688,     1878294428,     770994173,      841973243
};
const int32_t DataSet_1_32i::outputs::BXORV[32] = {
    -1208647392,    -1613809369,    1025439458,     1674878024,
    -1522953994,    1581743607,     -1028799898,    837859673,
    -96568615,      -1077126311,    -1924590440,    2092261168,
    -1499812525,    469164972,      -1391414302,    665783448,
    -1963956562,    -1026707846,    406472914,      -1311701004,
    351300091,      -472149494,     602380939,      289376656,
    775156900,      667378140,      -631125661,     1063429025,
    1365552482,     -1092969707,    711551962,      1967074809
};
const int32_t DataSet_1_32i::outputs::MBXORV[32] = {
    948484139,      325061806,      1092824755,     1674878024,
    -873226993,     1581743607,     -1028799898,    -1934370231,
    -817803908,     -1077126311,    -1924590440,    -363273962,
    -1499812525,    -582437808,     1132387793,     665783448,
    -1489706084,    -1026707846,    406472914,      1400938468,
    351300091,      1882764301,     2345500,        289376656,
    775156900,      2002189952,     1814442606,     1063429025,
    -991493688,     -1092969707,    711551962,      841973243
};
const int32_t DataSet_1_32i::outputs::BXORS[32] = {
    897374119,      513041186,      1289175359,     -470660669,
    -972557693,     -1268408281,    167398725,      -2126006843,
    -1028316432,    1211110330,     2031977156,     -408121702,
    894587705,      -792929316,     1317712477,     -598044149,
    -1430221808,    -2043035738,    -403556287,     1584722024,
    -1081017812,    2110577537,     232236432,      1244638322,
    392051973,      2057474316,     1641176546,     -1196515582,
    -921473468,     1742740500,     687083005,      1071321719
};
const int32_t DataSet_1_32i::outputs::MBXORS[32] = {
    948484139,      325061806,      1092824755,     -470660669,
    -873226993,     -1268408281,    167398725,      -1934370231,
    -817803908,     1211110330,     2031977156,     -363273962,
    894587705,      -582437808,     1132387793,     -598044149,
    -1489706084,    -2043035738,    -403556287,     1400938468,
    -1081017812,    1882764301,     2345500,        1244638322,
    392051973,      2002189952,     1814442606,     -1196515582,
    -991493688,     1742740500,     687083005,      841973243
};
const int32_t DataSet_1_32i::outputs::BNOT[32] = {
    -948484140,     -325061807,     -1092824756,    301586864,
    873226992,      1181626452,     -68035274,      1934370230,
    817803907,      -1170504759,    -1961487689,    363273961,
    -950418614,     582437807,      -1132387794,    777060984,
    1489706083,     1949446101,     368683058,      -1400938469,
    1302030943,     -1882764302,    -2345501,       -1205584895,
    -447369866,     -2002189953,    -1814442607,    1252331377,
    991493687,      -1779725209,    -620782194,     -841973244
};
const int32_t DataSet_1_32i::outputs::MBNOT[32] = {
    948484139,      325061806,      1092824755,     301586864,
    -873226993,     1181626452,     -68035274,      -1934370231,
    -817803908,     -1170504759,    -1961487689,    -363273962,
    -950418614,     -582437808,     1132387793,     777060984,
    -1489706084,    1949446101,     368683058,      1400938468,
    1302030943,     1882764301,     2345500,        -1205584895,
    -447369866,     2002189952,     1814442606,     1252331377,
    -991493688,     -1779725209,    -620782194,     841973243
};

const int32_t DataSet_1_32i::outputs::HBAND[32] = {
    948484139,      268437546,      2082,           2,
    2,              2,              0,              0,
    0,              0,              0,              0,
    0,              0,              0,              0,
    0,              0,              0,              0,
    0,              0,              0,              0,
    0,              0,              0,              0,
    0,              0,              0,              0
};
const int32_t DataSet_1_32i::outputs::MHBAND[32] = {
    -1,             -1,             -1,             -301586865,
    -301586865,     -1476393461,    521,            521,
    521,            0,              0,              0,
    0,              0,              0,              0,
    0,              0,              0,              0,
    0,              0,              0,              0,
    0,              0,              0,              0,
    0,              0,              0,              0
};

const int32_t DataSet_1_32i::outputs::HBANDS[32] = {
    142616584,      2056,   2048,   0,
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0,
    0,      0,      0,      0
};
const int32_t DataSet_1_32i::outputs::MHBANDS[32] = {
    234123148,      234123148,      234123148,      201598476,
    201598476,      134219272,      520,            520,
    520,            0,              0,              0,
    0,              0,              0,              0,
    0,              0,              0,              0,
    0,              0,              0,              0,
    0,              0,              0,              0,
    0,              0,              0,              0
};
const int32_t DataSet_1_32i::outputs::HBOR[32] = {
    948484139,      1005108399,     2079047359,     -1065217,
    -16385,         -1,             -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1
};
const int32_t DataSet_1_32i::outputs::MHBOR[32] = {
    0,              0,              0,              -301586865,
    -301586865,     -6819857,       -6295569,       -6295569,
    -6295569,       -2097153,       -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1
};

const int32_t DataSet_1_32i::outputs::HBORS[32] = {
    1039990703,     1073545135,     2147483583,     -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1
};
const int32_t DataSet_1_32i::outputs::MHBORS[32] = {
    234123148,      234123148,      234123148,      -269062193,
    -269062193,     -528401,        -4113,          -4113,
    -4113,          -1,             -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1,
    -1,             -1,             -1,             -1
};
const int32_t DataSet_1_32i::outputs::HBXOR[32] = {
    948484139,      736670853,      1791728182,     -2066891655,
    1329471862,     -156242211,     -224277484,     2115114589,
    -1319937247,    -191420649,     -2139174305,    1780886345,
    1384126460,     -1882676308,    -860455299,     488145914,
    -1171496858,    836874316,      -605558911,     -2006484891,
    973336005,      1245403592,     1243124692,     230903850,
    392757923,      1614806051,     203017805,      -1186775357,
    2107916043,     397429907,      850407138,      10441497
};
const int32_t DataSet_1_32i::outputs::MHBXOR[32] = {
    0,              0,              0,              -301586865,
    -301586865,     1469573604,     1402587949,     1402587949,
    1402587949,     375240475,      1655986771,     1655986771,
    1511155430,     1511155430,     1511155430,     -1950573727,
    -1950573727,    7424843,        -361292666,     -361292666,
    1477675302,     1477675302,     1477675302,     533220056,
    90314833,       90314833,       90314833,       -1338447649,
    -1338447649,    -634610873,     -13829834,      -13829834
};
const int32_t DataSet_1_32i::outputs::HBXORS[32] = {
    897374119,      639425289,      1732244922,     -1992698891,
    1120554746,     -77889199,      -11164776,      1944484305,
    -1129892691,    -110933861,     -1920293421,    1741833413,
    1601468528,     -2109944800,    -1052634639,    283962486,
    -1210576918,    1008066496,     -703363059,     -2053957655,
    938967625,      1204762180,     1206713432,     3620774,
    446524719,      1842058159,     32350657,       -1263054513,
    1884311687,     440673055,      1061437806,     225131669
};
const int32_t DataSet_1_32i::outputs::MHBXORS[32] = {
    234123148,      234123148,      234123148,      -470660669,
    -470660669,     1516471912,     1584243873,     1584243873,
    1584243873,     464115863,      1866479071,     1866479071,
    1474693482,     1474693482,     1474693482,     -2042039059,
    -2042039059,    226829511,      -410815734,     -410815734,
    1441261226,     1441261226,     1441261226,     305931604,
    144078813,      144078813,      144078813,      -1110669485,
    -1110669485,    -673647413,     -220686662,     -220686662
};


const uint32_t DataSet_1_32i::outputs::LSHV[32] = {
    0x82b00000,     0x3600cae0,     0x60000000,     0x6264f000,
    0xf9ce8780,     0x39f56000,     0x71164800,     0xb3da4900,
    0x535f0000,     0x86c00000,     0x3da90000,     0x80000000,
    0x531e5a80,     0x40000000,     0xee880000,     0xaefd8700,
    0x9c738000,     0xba054000,     0x2be68000,     0x9fe40000,
    0x68000000,     0x681a0000,     0x87000000,     0xbc3fe000,
    0x24000000,     0x00000000,     0x8c9b8000,     0x1c000000,
    0xce039000,     0x14739800,     0x13880000,     0xefd80000
};
const uint32_t DataSet_1_32i::outputs::MLSHV[32] = {
    0x3888b82b,     0x13600cae,     0x41232eb3,     0x6264f000,
    0xcbf39d0f,     0x39f56000,     0x71164800,     0x8cb3da49,
    0xcf414d7c,     0x86c00000,     0x3da90000,     0xea58e116,
    0x531e5a80,     0xdd48b450,     0x437eddd1,     0xaefd8700,
    0xa734e39c,     0xba054000,     0x2be68000,     0x53809fe4,
    0x68000000,     0x7038b40d,     0x0023ca1c,     0xbc3fe000,
    0x24000000,     0x7756fe80,     0x6c26326e,     0x1c000000,
    0xc4e701c8,     0x14739800,     0x13880000,     0x322f7dfb
};
const uint32_t DataSet_1_32i::outputs::LSHS[32] = {
    0x58000000,     0x70000000,     0x98000000,     0x78000000,
    0x78000000,     0x58000000,     0x48000000,     0x48000000,
    0xe0000000,     0xb0000000,     0x40000000,     0xb0000000,
    0xa8000000,     0x80000000,     0x88000000,     0x38000000,
    0xe0000000,     0x50000000,     0x68000000,     0x20000000,
    0x00000000,     0x68000000,     0xe0000000,     0xf0000000,
    0x48000000,     0x00000000,     0x70000000,     0x70000000,
    0x40000000,     0xc0000000,     0x88000000,     0xd8000000
};
const uint32_t DataSet_1_32i::outputs::MLSHS[32] = {
    0x3888b82b,     0x13600cae,     0x41232eb3,     0x78000000,
    0xcbf39d0f,     0x58000000,     0x48000000,     0x8cb3da49,
    0xcf414d7c,     0xb0000000,     0x40000000,     0xea58e116,
    0xa8000000,     0xdd48b450,     0x437eddd1,     0x38000000,
    0xa734e39c,     0x50000000,     0x68000000,     0x53809fe4,
    0x00000000,     0x7038b40d,     0x0023ca1c,     0xf0000000,
    0x48000000,     0x7756fe80,     0x6c26326e,     0x70000000,
    0xc4e701c8,     0xc0000000,     0x88000000,     0x322f7dfb
};
const uint32_t DataSet_1_32i::outputs::RSHV[32] = {
    0x00000388,     0x013600ca,     0x00000002,     0xfffee062,
    0xff97e73a,     0xfffdcc8e,     0x000081c4,     0xff8cb3da,
    0xffff3d05,     0x0000022e,     0x0003a74f,     0xffffffff,
    0x00714c79,     0xfffffff7,     0x0000086f,     0xffd1aefd,
    0xfffd39a7,     0xfffc5e6e,     0xffffd40c,     0x00005380,
    0xfffffec9,     0x0000381c,     0x00000000,     0x00047dbc,
    0x00000006,     0x00000001,     0x0001b098,     0xffffffda,
    0xffe27380,     0x006a1473,     0x000004a0,     0x00000645
};
const uint32_t DataSet_1_32i::outputs::MRSHV[32] = {
    0x3888b82b,     0x13600cae,     0x41232eb3,     0xfffee062,
    0xcbf39d0f,     0xfffdcc8e,     0x000081c4,     0x8cb3da49,
    0xcf414d7c,     0x0000022e,     0x0003a74f,     0xea58e116,
    0x00714c79,     0xdd48b450,     0x437eddd1,     0xffd1aefd,
    0xa734e39c,     0xfffc5e6e,     0xffffd40c,     0x53809fe4,
    0xfffffec9,     0x7038b40d,     0x0023ca1c,     0x00047dbc,
    0x00000006,     0x7756fe80,     0x6c26326e,     0xffffffda,
    0xc4e701c8,     0x006a1473,     0x000004a0,     0x322f7dfb
};
const uint32_t DataSet_1_32i::outputs::RSHS[32] = {
    0x00000007,     0x00000002,     0x00000008,     0xfffffffd,
    0xfffffff9,     0xfffffff7,     0x00000000,     0xfffffff1,
    0xfffffff9,     0x00000008,     0x0000000e,     0xfffffffd,
    0x00000007,     0xfffffffb,     0x00000008,     0xfffffffa,
    0xfffffff4,     0xfffffff1,     0xfffffffd,     0x0000000a,
    0xfffffff6,     0x0000000e,     0x00000000,     0x00000008,
    0x00000003,     0x0000000e,     0x0000000d,     0xfffffff6,
    0xfffffff8,     0x0000000d,     0x00000004,     0x00000006
};
const uint32_t DataSet_1_32i::outputs::MRSHS[32] = {
    0x3888b82b,     0x13600cae,     0x41232eb3,     0xfffffffd,
    0xcbf39d0f,     0xfffffff7,     0x00000000,     0x8cb3da49,
    0xcf414d7c,     0x00000008,     0x0000000e,     0xea58e116,
    0x00000007,     0xdd48b450,     0x437eddd1,     0xfffffffa,
    0xa734e39c,     0xfffffff1,     0xfffffffd,     0x53809fe4,
    0xfffffff6,     0x7038b40d,     0x0023ca1c,     0x00000008,
    0x00000003,     0x7756fe80,     0x6c26326e,     0xfffffff6,
    0xc4e701c8,     0x0000000d,     0x00000004,     0x322f7dfb
};
const uint32_t DataSet_1_32i::outputs::ROLV[32] = {
    0x82b3888b,     0x3600cae1,     0x682465d6,     0x6264fee0,
    0xf9ce87e5,     0x39f57732,     0x71164820,     0xb3da498c,
    0x535f33d0,     0x86c8b88f,     0x3da90e9d,     0xba963845,
    0x531e5a9c,     0x437522d1,     0xee8a1bf6,     0xaefd87d1,
    0x9c7394e6,     0xba055179,     0x2be6f503,     0x9fe45380,
    0x682c9925,     0x681ae071,     0x870008f2,     0xbc3fe47d,
    0x246aa94a,     0x1dd5bfa0,     0x8c9b9b09,     0x1d6ab5e1,
    0xce039189,     0x1473986a,     0x13892803,     0xefd9917b
};
const uint32_t DataSet_1_32i::outputs::MROLV[32] = {
    0x3888b82b,     0x13600cae,     0x41232eb3,     0x6264fee0,
    0xcbf39d0f,     0x39f57732,     0x71164820,     0x8cb3da49,
    0xcf414d7c,     0x86c8b88f,     0x3da90e9d,     0xea58e116,
    0x531e5a9c,     0xdd48b450,     0x437eddd1,     0xaefd87d1,
    0xa734e39c,     0xba055179,     0x2be6f503,     0x53809fe4,
    0x682c9925,     0x7038b40d,     0x0023ca1c,     0xbc3fe47d,
    0x246aa94a,     0x7756fe80,     0x6c26326e,     0x1d6ab5e1,
    0xc4e701c8,     0x1473986a,     0x13892803,     0x322f7dfb
};
const uint32_t DataSet_1_32i::outputs::ROLS[32] = {
    0x59c445c1,     0x709b0065,     0x9a091975,     0x7f703132,
    0x7e5f9ce8,     0x5dcc8e7d,     0x48207116,     0x4c659ed2,
    0xe67a0a6b,     0xb22e23e1,     0x43a74f6a,     0xb752c708,
    0xa9c531e5,     0x86ea45a2,     0x8a1bf6ee,     0x3e8d77ec,
    0xe539a71c,     0x545e6e81,     0x6f5032be,     0x229c04ff,
    0x059324ad,     0x6b81c5a0,     0xe0011e50,     0xf23ede1f,
    0x48d55294,     0x03bab7f4,     0x73613193,     0x75aad784,
    0x4627380e,     0xc350a39c,     0x89280313,     0xd9917bef
};
const uint32_t DataSet_1_32i::outputs::MROLS[32] = {
    0x3888b82b,     0x13600cae,     0x41232eb3,     0x7f703132,
    0xcbf39d0f,     0x5dcc8e7d,     0x48207116,     0x8cb3da49,
    0xcf414d7c,     0xb22e23e1,     0x43a74f6a,     0xea58e116,
    0xa9c531e5,     0xdd48b450,     0x437eddd1,     0x3e8d77ec,
    0xa734e39c,     0x545e6e81,     0x6f5032be,     0x53809fe4,
    0x059324ad,     0x7038b40d,     0x0023ca1c,     0xf23ede1f,
    0x48d55294,     0x7756fe80,     0x6c26326e,     0x75aad784,
    0xc4e701c8,     0xc350a39c,     0x89280313,     0x322f7dfb
};
const uint32_t DataSet_1_32i::outputs::RORV[32] = {
    0x8b82b388,     0xe13600ca,     0x0919759a,     0x64fee062,
    0x1f97e73a,     0x7d5dcc8e,     0x592081c4,     0x498cb3da,
    0x35f33d05,     0x23e1b22e,     0x6a43a74f,     0xa963845b,
    0x6a714c79,     0x522d1437,     0xdbba286f,     0x87d1aefd,
    0x1ce539a7,     0x81545e6e,     0xaf9bd40c,     0x9fe45380,
    0x925682c9,     0x5a06b81c,     0x8f287000,     0x3fe47dbc,
    0xaa94a246,     0xdd5bfa01,     0xc9b9b098,     0xad78475a,
    0xe4627380,     0x986a1473,     0x0c4e24a0,     0xefbf6645
};
const uint32_t DataSet_1_32i::outputs::MRORV[32] = {
    0x3888b82b,     0x13600cae,     0x41232eb3,     0x64fee062,
    0xcbf39d0f,     0x7d5dcc8e,     0x592081c4,     0x8cb3da49,
    0xcf414d7c,     0x23e1b22e,     0x6a43a74f,     0xea58e116,
    0x6a714c79,     0xdd48b450,     0x437eddd1,     0x87d1aefd,
    0xa734e39c,     0x81545e6e,     0xaf9bd40c,     0x53809fe4,
    0x925682c9,     0x7038b40d,     0x0023ca1c,     0x3fe47dbc,
    0xaa94a246,     0x7756fe80,     0x6c26326e,     0xad78475a,
    0xc4e701c8,     0x986a1473,     0x0c4e24a0,     0x322f7dfb
};
const uint32_t DataSet_1_32i::outputs::RORS[32] = {
    0x11170567,     0x6c0195c2,     0x2465d668,     0xc0c4c9fd,
    0x7e73a1f9,     0x3239f577,     0x81c45920,     0x967b4931,
    0xe829af99,     0xb88f86c8,     0x9d3da90e,     0x4b1c22dd,
    0x14c796a7,     0xa9168a1b,     0x6fdbba28,     0x35dfb0fa,
    0xe69c7394,     0x79ba0551,     0x40caf9bd,     0x7013fc8a,
    0x4c92b416,     0x071681ae,     0x04794380,     0xfb787fc8,
    0x554a5123,     0xeadfd00e,     0x84c64dcd,     0xab5e11d6,
    0x9ce03918,     0x428e730d,     0xa00c4e24,     0x45efbf66
};
const uint32_t DataSet_1_32i::outputs::MRORS[32] = {
    0x3888b82b,     0x13600cae,     0x41232eb3,     0xc0c4c9fd,
    0xcbf39d0f,     0x3239f577,     0x81c45920,     0x8cb3da49,
    0xcf414d7c,     0xb88f86c8,     0x9d3da90e,     0xea58e116,
    0x14c796a7,     0xdd48b450,     0x437eddd1,     0x35dfb0fa,
    0xa734e39c,     0x79ba0551,     0x40caf9bd,     0x53809fe4,
    0x4c92b416,     0x7038b40d,     0x0023ca1c,     0xfb787fc8,
    0x554a5123,     0x7756fe80,     0x6c26326e,     0xab5e11d6,
    0xc4e701c8,     0x428e730d,     0xa00c4e24,     0x322f7dfb
};


const int32_t DataSet_1_32i::outputs::NEG[32] = {
    -948484139,     -325061806,     -1092824755,    301586865,
    873226993,      1181626453,     -68035273,      1934370231,
    817803908,      -1170504758,    -1961487688,    363273962,
    -950418613,     582437808,      -1132387793,    777060985,
    1489706084,     1949446102,     368683059,      -1400938468,
    1302030944,     -1882764301,    -2345500,       -1205584894,
    -447369865,     -2002189952,    -1814442606,    1252331378,
    991493688,      -1779725208,    -620782193,     -841973243
};
const int32_t DataSet_1_32i::outputs::MNEG[32] = {
    948484139,      325061806,      1092824755,     301586865,
    -873226993,     1181626453,     -68035273,      -1934370231,
    -817803908,     -1170504758,    -1961487688,    -363273962,
    -950418613,     -582437808,     1132387793,     777060985,
    -1489706084,    1949446102,     368683059,      1400938468,
    1302030944,     1882764301,     2345500,        -1205584894,
    -447369865,     2002189952,     1814442606,     1252331378,
    -991493688,     -1779725208,    -620782193,     841973243
};
const int32_t DataSet_1_32i::outputs::ABS[32] = {
    948484139,      325061806,      1092824755,     301586865,
    873226993,      1181626453,     68035273,       1934370231,
    817803908,      1170504758,     1961487688,     363273962,
    950418613,      582437808,      1132387793,     777060985,
    1489706084,     1949446102,     368683059,      1400938468,
    1302030944,     1882764301,     2345500,        1205584894,
    447369865,      2002189952,     1814442606,     1252331378,
    991493688,      1779725208,     620782193,      841973243
};
const int32_t DataSet_1_32i::outputs::MABS[32] = {
    948484139,      325061806,      1092824755,     301586865,
    -873226993,     1181626453,     68035273,       -1934370231,
    -817803908,     1170504758,     1961487688,     -363273962,
    950418613,      -582437808,     1132387793,     777060985,
    -1489706084,    1949446102,     368683059,      1400938468,
    1302030944,     1882764301,     2345500,        1205584894,
    447369865,      2002189952,     1814442606,     1252331378,
    -991493688,     1779725208,     620782193,      841973243
};

const uint32_t DataSet_1_32i::outputs::ITOU[32] = {
    0x3888b82b,     0x13600cae,     0x41232eb3,     0xee06264f,
    0xcbf39d0f,     0xb991cfab,     0x040e22c9,     0x8cb3da49,
    0xcf414d7c,     0x45c47c36,     0x74e9ed48,     0xea58e116,
    0x38a63cb5,     0xdd48b450,     0x437eddd1,     0xd1aefd87,
    0xa734e39c,     0x8bcdd02a,     0xea0657cd,     0x53809fe4,
    0xb26495a0,     0x7038b40d,     0x0023ca1c,     0x47dbc3fe,
    0x1aaa5289,     0x7756fe80,     0x6c26326e,     0xb55af08e,
    0xc4e701c8,     0x6a147398,     0x25006271,     0x322f7dfb
};

const float DataSet_1_32i::outputs::ITOF[32] = {
    948484160.000000000f,   325061792.000000000f,   1092824704.000000000f,  -301586880.000000000f,
    -873227008.000000000f,  -1181626496.000000000f, 68035272.000000000f,    -1934370176.000000000f,
    -817803904.000000000f,  1170504704.000000000f,  1961487744.000000000f,  -363273952.000000000f,
    950418624.000000000f,   -582437824.000000000f,  1132387840.000000000f,  -777060992.000000000f,
    -1489706112.000000000f, -1949446144.000000000f, -368683072.000000000f,  1400938496.000000000f,
    -1302030976.000000000f, 1882764288.000000000f,  2345500.000000000f,     1205584896.000000000f,
    447369856.000000000f,   2002189952.000000000f,  1814442624.000000000f,  -1252331392.000000000f,
    -991493696.000000000f,  1779725184.000000000f,  620782208.000000000f,   841973248.000000000
};




const float DataSet_1_32f::inputs::inputA[32] = {
     -100.558593750000f,    2686.696777343750f,     -136.265136718750f,     4406.415039062500f,
    -514.084472656250f,     2339.091308593750f,     1863.917968750000f,     -3039.185791015625f,
    1879.787597656250f,     -2183.446777343750f,    -1044.496093750000f,    601.367187500000f,
    -252.540527343750f,     -2759.941406250000f,    3421.887695312500f,     3541.825195312500f,
    -1742.759521484375f,    674.611816406250f,      3355.357421875000f,     1645.405273437500f,
    4394.818359375000f,     -1560.869140625000f,    -1027.100341796875f,    -4268.776367187500f,
    -3708.151367187500f,    -636.768554687500f,     -893.734375000000f,     -1646.626220703125f,
    4145.176757812500f,     4291.054687500000f,     1929.837890625000f,     -2136.753417968750f
};

const float DataSet_1_32f::inputs::inputB[32] = {
    -19.989746093750f,      -1620.990722656250f,    1452.528320312500f,     -766.777343750000f,
    4454.329101562500f,     -595.263671875000f,     -1198.614501953125f,    4600.817382812500f,
    33.723144531250f,       -4159.214843750000f,    -4700.918457031250f,    1690.267578125000f,
    -2996.459960937500f,    -3747.215332031250f,    1995.147460937500f,     2126.071777343750f,
    1177.861816406250f,     -3868.068359375000f,    -1390.881103515625f,    3157.292480468750f,
    -3422.803466796875f,    1116.214355468750f,     128.024902343750f,      -873.287109375000f,
    -2733.695556640625f,    2728.202148437500f,     -4162.877441406250f,    -4380.474121093750f,
    1163.518066406250f,     -1805.627685546875f,    -3889.431396484375f,    347.758300781250f
};

const float DataSet_1_32f::inputs::inputC[32] = {
    74.617675781250f,       -6.256347656250f,       -634.937500000000f,     -3130.741210937500f,
    3491.469726562500f,     3247.322265625000f,     -963.622070312500f,     -265.968750000000f,
    -1267.281250000000f,    -2401.959228515625f,    1011.230957031250f,     2967.162109375000f,
    -1152.836669921875f,    -4041.108398437500f,    -4616.687500000000f,    4786.980468750000f,
    -1323.129882812500f,    310.525878906250f,      1684.774414062500f,     -4822.687500000000f,
    -2467.879394531250f,    -65.157226562500f,      -4566.942382812500f,    2491.988769531250f,
    631.275390625000f,      3341.013671875000f,     3608.355468750000f,     3277.840820312500f,
    876.338867187500f,      -807.977539062500f,     3113.345703125000f,     3137.760742187500f
};

const uint32_t DataSet_1_32f::inputs::inputUintA[32] = {
    3273082796, 4034370560, 3724343109, 3837331920,
    4039792685, 3251975234, 2975186003, 2501147154,
    729672903,  3346757884, 2929511781, 1454646875,
    3604265798, 978895957,  1251551287, 1898914091,
    2792756694, 1441485571, 3332554660, 2854188413,
    3917849407, 3356278803, 363973866,  315720241,
    3083762480, 3595285540, 3751632854, 2121976320,
    2438295964, 1083364314, 36816391,   3766746593
};

const int32_t DataSet_1_32f::inputs::inputIntA[32] = {
    1196535124,     726590047,      -199236052,     906531650,
    1096768198,     -1668390515,    -1515839972,    -1060143288,
    -1392312974,    -1116745340,    -772215731,     -1046462947,
    -841197175,     618367478,      -1377362169,    1269201977,
    447678805,      -293813370,     1272230454,     1756815974,
    2107971637,     -720187790,     264742751,      1672053848,
    1481089269,     -1135701836,    -331170375,     1013950633,
    9647458,        102792793,      -1660519627,    1275239050
};


const float DataSet_1_32f::inputs::scalarA = 255.897461f;

const bool DataSet_1_32f::inputs::maskA[32] = {
    true,   false,  false,  true,   // 4
    false,  true,   true,   false,  // 8

    false,  true,   true,   false,  
    true,   false,  false,  true,   // 16
    
    false,  true,   true,   false,
    true,   false,  false,  true,
    true,   false,  false,  true,
    false,  true,   true,   false,  // 32
};

const float DataSet_1_32f::outputs::ADDV[32] = {
    -120.548339844f,        1065.706054688f,        1316.263183594f,        3639.637695313f,
    3940.244628906f,        1743.827636719f,        665.303466797f,         1561.631591797f,
    1913.510742188f,        -6342.661621094f,       -5745.414550781f,       2291.634765625f,
    -3249.000488281f,       -6507.156738281f,       5417.035156250f,        5667.896972656f,
    -564.897705078f,        -3193.456542969f,       1964.476318359f,        4802.697753906f,
    972.014892578f,         -444.654785156f,        -899.075439453f,        -5142.063476563f,
    -6441.846679688f,       2091.433593750f,        -5056.611816406f,       -6027.100585938f,
    5308.694824219f,        2485.427001953f,        -1959.593505859f,       -1788.995117188f
}; 

const float DataSet_1_32f::outputs::MADDV[32] = {
    -120.548339844f,        2686.696777344f,        -136.265136719f,        3639.637695313f,
    -514.084472656f,        1743.827636719f,        665.303466797f,         -3039.185791016f,
    1879.787597656f,        -6342.661621094f,       -5745.414550781f,       601.367187500f,
    -3249.000488281f,       -2759.941406250f,       3421.887695313f,        5667.896972656f,
    -1742.759521484f,       -3193.456542969f,       1964.476318359f,        1645.405273438f,
    972.014892578f,         -1560.869140625f,       -1027.100341797f,       -5142.063476563f,
    -6441.846679688f,       -636.768554688f,        -893.734375000f,        -6027.100585938f,
    4145.176757813f,        2485.427001953f,        -1959.593505859f,       -2136.753417969f
};

const float DataSet_1_32f::outputs::ADDS[32] = {
    155.338867188f,         2942.594238281f,        119.632324219f,         4662.312500000f,
    -258.187011719f,        2594.988769531f,        2119.815429688f,        -2783.288330078f,
    2135.685058594f,        -1927.549316406f,       -788.598632813f,        857.264648438f,
    3.356933594f,           -2504.043945313f,       3677.785156250f,        3797.722656250f,
    -1486.862060547f,       930.509277344f,         3611.254882813f,        1901.302734375f,
    4650.715820313f,        -1304.971679688f,       -771.202880859f,        -4012.878906250f,
    -3452.253906250f,       -380.871093750f,        -637.836914063f,        -1390.728759766f,
    4401.074218750f,        4546.952148438f,        2185.735351563f,        -1880.855957031f
};

const float DataSet_1_32f::outputs::MADDS[32] = {
    155.338867188f,         2686.696777344f,        -136.265136719f,        4662.312500000f,
    -514.084472656f,        2594.988769531f,        2119.815429688f,        -3039.185791016f,
    1879.787597656f,        -1927.549316406f,       -788.598632813f,        601.367187500f,
    3.356933594f,           -2759.941406250f,       3421.887695313f,        3797.722656250f,
    -1742.759521484f,       930.509277344f,         3611.254882813f,        1645.405273438f,
    4650.715820313f,        -1560.869140625f,       -1027.100341797f,       -4012.878906250f,
    -3452.253906250f,       -636.768554688f,        -893.734375000f,        -1390.728759766f,
    4145.176757813f,        4546.952148438f,        2185.735351563f,        -2136.753417969f
};

const float DataSet_1_32f::outputs::POSTPREFINC[32] = {
    -99.558593750f,         2687.696777344f,        -135.265136719f,        4407.415039063f,
    -513.084472656f,        2340.091308594f,        1864.917968750f,        -3038.185791016f,
    1880.787597656f,        -2182.446777344f,       -1043.496093750f,       602.367187500f,
    -251.540527344f,        -2758.941406250f,       3422.887695313f,        3542.825195313f,
    -1741.759521484f,       675.611816406f,         3356.357421875f,        1646.405273438f,
    4395.818359375f,        -1559.869140625f,       -1026.100341797f,       -4267.776367188f,
    -3707.151367188f,       -635.768554688f,        -892.734375000f,        -1645.626220703f,
    4146.176757813f,        4292.054687500f,        1930.837890625f,        -2135.753417969f
};

const float DataSet_1_32f::outputs::MPOSTPREFINC[32] = {
    -99.558593750f,         2686.696777344f,        -136.265136719f,        4407.415039063f,
    -514.084472656f,        2340.091308594f,        1864.917968750f,        -3039.185791016f,
    1879.787597656f,        -2182.446777344f,       -1043.496093750f,       601.367187500f,
    -251.540527344f,        -2759.941406250f,       3421.887695313f,        3542.825195313f,
    -1742.759521484f,       675.611816406f,         3356.357421875f,        1645.405273438f,
    4395.818359375f,        -1560.869140625f,       -1027.100341797f,       -4267.776367188f,
    -3707.151367188f,       -636.768554688f,        -893.734375000f,        -1645.626220703f,
    4145.176757813f,        4292.054687500f,        1930.837890625f,        -2136.753417969f
};

const float DataSet_1_32f::outputs::SUBV[32] = {
    -80.568847656f,         4307.687500000f,        -1588.793457031f,       5173.192382813f,
    -4968.413574219f,       2934.354980469f,        3062.532470703f,        -7640.002929688f,
    1846.064453125f,        1975.768066406f,        3656.422363281f,        -1088.900390625f,
    2743.919433594f,        987.273925781f,         1426.740234375f,        1415.753417969f,
    -2920.621337891f,       4542.680175781f,        4746.238281250f,        -1511.887207031f,
    7817.622070313f,        -2677.083496094f,       -1155.125244141f,       -3395.489257813f,
    -974.455810547f,        -3364.970703125f,       3269.143066406f,        2733.847900391f,
    2981.658691406f,        6096.682617188f,        5819.269531250f,        -2484.511718750f
};

const float DataSet_1_32f::outputs::MSUBV[32] = {
    -80.568847656f,         2686.696777344f,        -136.265136719f,        5173.192382813f,
    -514.084472656f,        2934.354980469f,        3062.532470703f,        -3039.185791016f,
    1879.787597656f,        1975.768066406f,        3656.422363281f,        601.367187500f,
    2743.919433594f,        -2759.941406250f,       3421.887695313f,        1415.753417969f,
    -1742.759521484f,       4542.680175781f,        4746.238281250f,        1645.405273438f,
    7817.622070313f,        -1560.869140625f,       -1027.100341797f,       -3395.489257813f,
    -974.455810547f,        -636.768554688f,        -893.734375000f,        2733.847900391f,
    4145.176757813f,        6096.682617188f,        5819.269531250f,        -2136.753417969f
};

const float DataSet_1_32f::outputs::SUBS[32] = {
    -356.456054688f,        2430.799316406f,        -392.162597656f,        4150.517578125f,
    -769.981933594f,        2083.193847656f,        1608.020507813f,        -3295.083251953f,
    1623.890136719f,        -2439.344238281f,       -1300.393554688f,       345.469726563f,
    -508.437988281f,        -3015.838867188f,       3165.990234375f,        3285.927734375f,
    -1998.656982422f,       418.714355469f,         3099.459960938f,        1389.507812500f,
    4138.920898438f,        -1816.766601563f,       -1282.997802734f,       -4524.673828125f,
    -3964.048828125f,       -892.666015625f,        -1149.631835938f,       -1902.523681641f,
    3889.279296875f,        4035.157226563f,        1673.940429688f,        -2392.650878906f
};

const float DataSet_1_32f::outputs::MSUBS[32] = {
    -356.456054688f,        2686.696777344f,        -136.265136719f,        4150.517578125f,
    -514.084472656f,        2083.193847656f,        1608.020507813f,        -3039.185791016f,
    1879.787597656f,        -2439.344238281f,       -1300.393554688f,       601.367187500f,
    -508.437988281f,        -2759.941406250f,       3421.887695313f,        3285.927734375f,
    -1742.759521484f,       418.714355469f,         3099.459960938f,        1645.405273438f,
    4138.920898438f,        -1560.869140625f,       -1027.100341797f,       -4524.673828125f,
    -3964.048828125f,       -636.768554688f,        -893.734375000f,        -1902.523681641f,
    4145.176757813f,        4035.157226563f,        1673.940429688f,        -2136.753417969f
};

const float DataSet_1_32f::outputs::SUBFROMV[32] = {
    80.568847656f,          -4307.687500000f,       1588.793457031f,        -5173.192382813f,
    4968.413574219f,        -2934.354980469f,       -3062.532470703f,       7640.002929688f,
    -1846.064453125f,       -1975.768066406f,       -3656.422363281f,       1088.900390625f,
    -2743.919433594f,       -987.273925781f,        -1426.740234375f,       -1415.753417969f,
    2920.621337891f,        -4542.680175781f,       -4746.238281250f,       1511.887207031f,
    -7817.622070313f,       2677.083496094f,        1155.125244141f,        3395.489257813f,
    974.455810547f,         3364.970703125f,        -3269.143066406f,       -2733.847900391f,
    -2981.658691406f,       -6096.682617188f,       -5819.269531250f,       2484.511718750f
};

const float DataSet_1_32f::outputs::MSUBFROMV[32] = {
    80.568847656f,          -1620.990722656f,       1452.528320313f,        -5173.192382813f,
    4454.329101563f,        -2934.354980469f,       -3062.532470703f,       4600.817382813f,
    33.723144531f,          -1975.768066406f,       -3656.422363281f,       1690.267578125f,
    -2743.919433594f,       -3747.215332031f,       1995.147460938f,        -1415.753417969f,
    1177.861816406f,        -4542.680175781f,       -4746.238281250f,       3157.292480469f,
    -7817.622070313f,       1116.214355469f,        128.024902344f,         3395.489257813f,
    974.455810547f,         2728.202148438f,        -4162.877441406f,       -2733.847900391f,
    1163.518066406f,        -6096.682617188f,       -5819.269531250f,       347.758300781f
};

const float DataSet_1_32f::outputs::SUBFROMS[32] = {
    356.456054688f,         -2430.799316406f,       392.162597656f,         -4150.517578125f,
    769.981933594f,         -2083.193847656f,       -1608.020507813f,       3295.083251953f,
    -1623.890136719f,       2439.344238281f,        1300.393554688f,        -345.469726563f,
    508.437988281f,         3015.838867188f,        -3165.990234375f,       -3285.927734375f,
    1998.656982422f,        -418.714355469f,        -3099.459960938f,       -1389.507812500f,
    -4138.920898438f,       1816.766601563f,        1282.997802734f,        4524.673828125f,
    3964.048828125f,        892.666015625f,         1149.631835938f,        1902.523681641f,
    -3889.279296875f,       -4035.157226563f,       -1673.940429688f,       2392.650878906f
};

const float DataSet_1_32f::outputs::MSUBFROMS[32] = {
    356.456054688f,         255.897460938f,         255.897460938f,         -4150.517578125f,
    255.897460938f,         -2083.193847656f,       -1608.020507813f,       255.897460938f,
    255.897460938f,         2439.344238281f,        1300.393554688f,        255.897460938f,
    508.437988281f,         255.897460938f,         255.897460938f,         -3285.927734375f,
    255.897460938f,         -418.714355469f,        -3099.459960938f,       255.897460938f,
    -4138.920898438f,       255.897460938f,         255.897460938f,         4524.673828125f,
    3964.048828125f,        255.897460938f,         255.897460938f,         1902.523681641f,
    255.897460938f,         -4035.157226563f,       -1673.940429688f,       255.897460938f
};

const float DataSet_1_32f::outputs::POSTPREFDEC[32] = {
    -101.558593750f,        2685.696777344f,        -137.265136719f,        4405.415039063f,
    -515.084472656f,        2338.091308594f,        1862.917968750f,        -3040.185791016f,
    1878.787597656f,        -2184.446777344f,       -1045.496093750f,       600.367187500f,
    -253.540527344f,        -2760.941406250f,       3420.887695313f,        3540.825195313f,
    -1743.759521484f,       673.611816406f,         3354.357421875f,        1644.405273438f,
    4393.818359375f,        -1561.869140625f,       -1028.100341797f,       -4269.776367188f,
    -3709.151367188f,       -637.768554688f,        -894.734375000f,        -1647.626220703f,
    4144.176757813f,        4290.054687500f,        1928.837890625f,        -2137.753417969f
};

const float DataSet_1_32f::outputs::MPOSTPREFDEC[32] = {
    -101.558593750f,        2686.696777344f,        -136.265136719f,        4405.415039063f,
    -514.084472656f,        2338.091308594f,        1862.917968750f,        -3039.185791016f,
    1879.787597656f,        -2184.446777344f,       -1045.496093750f,       601.367187500f,
    -253.540527344f,        -2759.941406250f,       3421.887695313f,        3540.825195313f,
    -1742.759521484f,       673.611816406f,         3354.357421875f,        1645.405273438f,
    4393.818359375f,        -1560.869140625f,       -1027.100341797f,       -4269.776367188f,
    -3709.151367188f,       -636.768554688f,        -893.734375000f,        -1647.626220703f,
    4145.176757813f,        4290.054687500f,        1928.837890625f,        -2136.753417969f
};

const float DataSet_1_32f::outputs::MULV[32] = {
    2010.140747070f,        -4355110.500000000f,    -197928.968750000f,     -3378739.250000000f,
    -2289901.500000000f,    -1392376.125000000f,    -2234119.000000000f,    -13982739.000000000f,
    63392.347656250f,       9081424.000000000f,     4910091.000000000f,     1016471.437500000f,
    756727.562500000f,      10342095.000000000f,    6827170.500000000f,     7530174.500000000f,
    -2052729.875000000f,    -2609444.500000000f,    -4666903.000000000f,    5195025.500000000f,
    -15042600.000000000f,   -1742264.500000000f,    -131494.421875000f,     3727867.250000000f,
    10136957.000000000f,    -1737233.375000000f,    3720506.750000000f,     7213003.500000000f,
    4822988.000000000f,     -7748047.000000000f,    -7505972.000000000f,    -743073.750000000f
};

const float DataSet_1_32f::outputs::MMULV[32] = {
    2010.140747070f,        2686.696777344f,        -136.265136719f,        -3378739.250000000f,
    -514.084472656f,        -1392376.125000000f,    -2234119.000000000f,    -3039.185791016f,
    1879.787597656f,        9081424.000000000f,     4910091.000000000f,     601.367187500f,
    756727.562500000f,      -2759.941406250f,       3421.887695313f,        7530174.500000000f,
    -1742.759521484f,       -2609444.500000000f,    -4666903.000000000f,    1645.405273438f,
    -15042600.000000000f,   -1560.869140625f,       -1027.100341797f,       3727867.250000000f,
    10136957.000000000f,    -636.768554688f,        -893.734375000f,        7213003.500000000f,
    4145.176757813f,        -7748047.000000000f,    -7505972.000000000f,    -2136.753417969f
};

const float DataSet_1_32f::outputs::MULS[32] = {
    -25732.689453125f,      687518.875000000f,      -34869.902343750f,      1127590.375000000f,
    -131552.906250000f,     598567.500000000f,      476971.875000000f,      -777719.937500000f,
    481032.875000000f,      -558738.500000000f,     -267283.906250000f,     153888.343750000f,
    -64624.480468750f,      -706262.000000000f,     875652.375000000f,      906344.062500000f,
    -445967.750000000f,     172631.453125000f,      858627.437500000f,      421055.031250000f,
    1124622.875000000f,     -399422.437500000f,     -262832.375000000f,     -1092369.000000000f,
    -948906.500000000f,     -162947.453125000f,     -228704.359375000f,     -421367.468750000f,
    1060740.250000000f,     1098070.000000000f,     493840.625000000f,      -546789.750000000f
};

const float DataSet_1_32f::outputs::MMULS[32] = {
    -25732.689453125f,      2686.696777344f,        -136.265136719f,        1127590.375000000f,
    -514.084472656f,        598567.500000000f,      476971.875000000f,      -3039.185791016f,
    1879.787597656f,        -558738.500000000f,     -267283.906250000f,     601.367187500f,
    -64624.480468750f,      -2759.941406250f,       3421.887695313f,        906344.062500000f,
    -1742.759521484f,       172631.453125000f,      858627.437500000f,      1645.405273438f,
    1124622.875000000f,     -1560.869140625f,       -1027.100341797f,       -1092369.000000000f,
    -948906.500000000f,     -636.768554688f,        -893.734375000f,        -421367.468750000f,
    4145.176757813f,        1098070.000000000f,     493840.625000000f,      -2136.753417969f
};

const float DataSet_1_32f::outputs::DIVV[32] = {
    5.030508995f,   -1.657441139f,  -0.093812376f,  -5.746668339f,
    -0.115412325f,  -3.929504633f,  -1.555060387f,  -0.660575211f,
    55.741764069f,  0.524966121f,   0.222189799f,   0.355782241f,
    0.084279627f,   0.736531317f,   1.715105176f,   1.665901065f,
    -1.479595900f,  -0.174405351f,  -2.412396908f,  0.521144390f,
    -1.283982038f,  -1.398359656f,  -8.022660255f,  4.888170719f,
    1.356460929f,   -0.233402267f,  0.214691490f,   0.375901371f,
    3.562623501f,   -2.376489162f,  -0.496174812f,  -6.144363403f
};

const float DataSet_1_32f::outputs::MDIVV[32] = {
    5.030508995f,           2686.696777344f,        -136.265136719f,        -5.746668339f,
    -514.084472656f,        -3.929504633f,          -1.555060387f,          -3039.185791016f,
    1879.787597656f,        0.524966121f,           0.222189799f,           601.367187500f,
    0.084279627f,           -2759.941406250f,       3421.887695313f,        1.665901065f,
    -1742.759521484f,       -0.174405351f,          -2.412396908f,          1645.405273438f,
    -1.283982038f,          -1560.869140625f,       -1027.100341797f,       4.888170719f,
    1.356460929f,           -636.768554688f,        -893.734375000f,        0.375901371f,
    4145.176757813f,        -2.376489162f,          -0.496174812f,          -2136.753417969f
};

const float DataSet_1_32f::outputs::DIVS[32] = {
    -0.392964393f,  10.499114990f,  -0.532498956f,  17.219455719f,
    -2.008947134f,  9.140736580f,   7.283846855f,   -11.876576424f,
    7.345862865f,   -8.532506943f,  -4.081697941f,  2.350031853f,
    -0.986881733f,  -10.785341263f, 13.372104645f,  13.840798378f,
    -6.810382366f,  2.636258364f,   13.112116814f,  6.429939747f,
    17.174139023f,  -6.099588394f,  -4.013718605f,  -16.681589127f,
    -14.490770340f, -2.488373756f,  -3.492548704f,  -6.434710979f,
    16.198585510f,  16.768648148f,  7.541450024f,   -8.350037575f
};

const float DataSet_1_32f::outputs::MDIVS[32] = {
    -0.392964393f,          2686.696777344f,        -136.265136719f,        17.219455719f,
    -514.084472656f,        9.140736580f,           7.283846855f,           -3039.185791016f,
    1879.787597656f,        -8.532506943f,          -4.081697941f,          601.367187500f,
    -0.986881733f,          -2759.941406250f,       3421.887695313f,        13.840798378f,
    -1742.759521484f,       2.636258364f,           13.112116814f,          1645.405273438f,
    17.174139023f,          -1560.869140625f,       -1027.100341797f,       -16.681589127f,
    -14.490770340f,         -636.768554688f,        -893.734375000f,        -6.434710979f,
    4145.176757813f,        16.768648148f,          7.541450024f,           -2136.753417969f
};

const float DataSet_1_32f::outputs::RCP[32] = {
    -0.009944451f,  0.000372204f,   -0.007338634f,  0.000226942f,
    -0.001945206f,  0.000427516f,   0.000536504f,   -0.000329036f,
    0.000531975f,   -0.000457991f,  -0.000957399f,  0.001662878f,
    -0.003959761f,  -0.000362327f,  0.000292236f,   0.000282340f,
    -0.000573803f,  0.001482334f,   0.000298031f,   0.000607753f,
    0.000227541f,   -0.000640669f,  -0.000973615f,  -0.000234259f,
    -0.000269676f,  -0.001570429f,  -0.001118901f,  -0.000607302f,
    0.000241244f,   0.000233043f,   0.000518178f,   -0.000468000f
};

const float DataSet_1_32f::outputs::MRCP[32] = {
    -0.009944451f,          2686.696777344f,        -136.265136719f,        0.000226942f,
    -514.084472656f,        0.000427516f,           0.000536504f,           -3039.185791016f,
    1879.787597656f,        -0.000457991f,          -0.000957399f,          601.367187500f,
    -0.003959761f,          -2759.941406250f,       3421.887695313f,        0.000282340f,
    -1742.759521484f,       0.001482334f,           0.000298031f,           1645.405273438f,
    0.000227541f,           -1560.869140625f,       -1027.100341797f,       -0.000234259f,
    -0.000269676f,          -636.768554688f,        -893.734375000f,        -0.000607302f,
    4145.176757813f,        0.000233043f,           0.000518178f,           -2136.753417969f
};

const float DataSet_1_32f::outputs::RCPS[32] = {
    -2.544759750f,  0.095246129f,   -1.877937913f,  0.058073845f,
    -0.497773170f,  0.109400369f,   0.137290090f,   -0.084199347f,
    0.136131048f,   -0.117198855f,  -0.244996086f,  0.425526142f,
    -1.013292670f,  -0.092718437f,  0.074782543f,   0.072250165f,
    -0.146834642f,  0.379325509f,   0.076265335f,   0.155522451f,
    0.058227085f,   -0.163945496f,  -0.249145538f,  -0.059946325f,
    -0.069009446f,  -0.401868880f,  -0.286323845f,  -0.155407131f,
    0.061733786f,   0.059635095f,   0.132600501f,   -0.119759940f
};

const float DataSet_1_32f::outputs::MRCPS[32] = {
    -2.544759750f,          2686.696777344f,        -136.265136719f,        0.058073845f,
    -514.084472656f,        0.109400369f,           0.137290090f,           -3039.185791016f,
    1879.787597656f,        -0.117198855f,          -0.244996086f,          601.367187500f,
    -1.013292670f,          -2759.941406250f,       3421.887695313f,        0.072250165f,
    -1742.759521484f,       0.379325509f,           0.076265335f,           1645.405273438f,
    0.058227085f,           -1560.869140625f,       -1027.100341797f,       -0.059946325f,
    -0.069009446f,          -636.768554688f,        -893.734375000f,        -0.155407131f,
    4145.176757813f,        0.059635095f,           0.132600501f,           -2136.753417969f
};

const bool DataSet_1_32f::outputs::CMPEQV[32] = {
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false
};

const bool DataSet_1_32f::outputs::CMPEQS[32] = {
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false
};

const bool DataSet_1_32f::outputs::CMPNEV[32] = {
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true
};

const bool DataSet_1_32f::outputs::CMPNES[32] = {
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true
};

const bool DataSet_1_32f::outputs::CMPGTV[32] = {
    false,  true,   false,  true,   false,  true,   true,   false,
    true,   true,   true,   false,  true,   true,   true,   true,
    false,  true,   true,   false,  true,   false,  false,  false,
    false,  false,  true,   true,   true,   true,   true,   false
};

const bool DataSet_1_32f::outputs::CMPGTS[32] = {
    false,  true,   false,  true,   false,  true,   true,   false,
    true,   false,  false,  true,   false,  false,  true,   true,
    false,  true,   true,   true,   true,   false,  false,  false,
    false,  false,  false,  false,  true,   true,   true,   false
};

const bool DataSet_1_32f::outputs::CMPLTV[32] = {
    true,   false,  true,   false,  true,   false,  false,  true,
    false,  false,  false,  true,   false,  false,  false,  false,
    true,   false,  false,  true,   false,  true,   true,   true,
    true,   true,   false,  false,  false,  false,  false,  true
};

const bool DataSet_1_32f::outputs::CMPLTS[32] = {
    true,   false,  true,   false,  true,   false,  false,  true,
    false,  true,   true,   false,  true,   true,   false,  false,
    true,   false,  false,  false,  false,  true,   true,   true,
    true,   true,   true,   true,   false,  false,  false,  true
};

const bool DataSet_1_32f::outputs::CMPGEV[32] = {
    false,  true,   false,  true,   false,  true,   true,   false,
    true,   true,   true,   false,  true,   true,   true,   true,
    false,  true,   true,   false,  true,   false,  false,  false,
    false,  false,  true,   true,   true,   true,   true,   false
};

const bool DataSet_1_32f::outputs::CMPGES[32] = {
    false,  true,   false,  true,   false,  true,   true,   false,
    true,   false,  false,  true,   false,  false,  true,   true,
    false,  true,   true,   true,   true,   false,  false,  false,
    false,  false,  false,  false,  true,   true,   true,   false
};

const bool DataSet_1_32f::outputs::CMPLEV[32] = {
    true,   false,  true,   false,  true,   false,  false,  true,
    false,  false,  false,  true,   false,  false,  false,  false,
    true,   false,  false,  true,   false,  true,   true,   true,
    true,   true,   false,  false,  false,  false,  false,  true
};

const bool DataSet_1_32f::outputs::CMPLES[32] = {
    true,   false,  true,   false,  true,   false,  false,  true,
    false,  true,   true,   false,  true,   true,   false,  false,
    true,   false,  false,  false,  false,  true,   true,   true,
    true,   true,   true,   true,   false,  false,  false,  true
};

const bool DataSet_1_32f::outputs::CMPEV = false;
const bool DataSet_1_32f::outputs::CMPES = false;

const float DataSet_1_32f::outputs::HADD[32] = {
    -100.558593750f,        2586.138183594f,        2449.873046875f,        6856.288085938f,
    6342.203613281f,        8681.294921875f,        10545.212890625f,       7506.027343750f,
    9385.814453125f,        7202.367675781f,        6157.871582031f,        6759.238769531f,
    6506.698242188f,        3746.756835938f,        7168.644531250f,        10710.469726563f,
    8967.709960938f,        9642.322265625f,        12997.679687500f,       14643.084960938f,
    19037.902343750f,       17477.033203125f,       16449.933593750f,       12181.157226563f,
    8473.005859375f,        7836.237304688f,        6942.502929688f,        5295.876953125f,
    9441.053710938f,        13732.108398438f,       15661.946289063f,       13525.193359375f
};

const float DataSet_1_32f::outputs::MHADD[32] = {
    -100.558593750f,        -100.558593750f,        -100.558593750f,        4305.856445313f,
    4305.856445313f,        6644.947753906f,        8508.865234375f,        8508.865234375f,
    8508.865234375f,        6325.418457031f,        5280.922363281f,        5280.922363281f,
    5028.381835938f,        5028.381835938f,        5028.381835938f,        8570.207031250f,
    8570.207031250f,        9244.818359375f,        12600.175781250f,       12600.175781250f,
    16994.994140625f,       16994.994140625f,       16994.994140625f,       12726.217773438f,
    9018.066406250f,        9018.066406250f,        9018.066406250f,        7371.440429688f,
    7371.440429688f,        11662.495117188f,       13592.333007813f,       13592.333007813f
};

const float DataSet_1_32f::outputs::HMUL[32] = {
    -100.558593750f,                                -270170.437500000f,     
    36814812.000000000f,                            162221342720.000000000f,
    -83395472261120.000000000f,                     -195069632103579650.000000000f, 
    -363593788237990070000.000000000f,              1105029073247884500000000.000000000f,
    2077219966122241400000000000.000000000f,        -4535499192154353100000000000000.000000000f,    
    4737311185166095000000000000000000.000000000f,  2848863449977112800000000000000000000.000000000f,
    -std::numeric_limits<float>::infinity(),        std::numeric_limits<float>::infinity(),   
    std::numeric_limits<float>::infinity(),         std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),        -std::numeric_limits<float>::infinity(),  
    -std::numeric_limits<float>::infinity(),        -std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),        std::numeric_limits<float>::infinity(),   
    -std::numeric_limits<float>::infinity(),        std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),        std::numeric_limits<float>::infinity(),   
    -std::numeric_limits<float>::infinity(),        std::numeric_limits<float>::infinity(),
    std::numeric_limits<float>::infinity(),         std::numeric_limits<float>::infinity(),   
    std::numeric_limits<float>::infinity(),         -std::numeric_limits<float>::infinity()
};

const float DataSet_1_32f::outputs::MHMUL[32] = {
    -100.558593750f,                                -100.558593750f,        
    -100.558593750f,                                -443102.906250000f,
    -443102.906250000f,                             -1036458176.000000000f, 
    -1931872960512.000000000f,                      -1931872960512.000000000f,
    -1931872960512.000000000f,                      4218141873799168.000000000f,    
    -4405832702696095700.000000000f,                -4405832702696095700.000000000f,
    1112651301901318500000.000000000f,              1112651301901318500000.000000000f,      
    1112651301901318500000.000000000f,              3940816271106599700000000.000000000f,
    3940816271106599700000000.000000000f,           2658521144070606200000000000.000000000f,        
    8920288855932268100000000000000.000000000f,     8920288855932268100000000000000.000000000f,
    39203048025888244000000000000000000.000000000f, 39203048025888244000000000000000000.000000000f, 
    39203048025888244000000000000000000.000000000f, -167349046107201740000000000000000000000.000000000f,
    std::numeric_limits<float>::infinity(),         std::numeric_limits<float>::infinity(),   
    std::numeric_limits<float>::infinity(),         -std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),        -std::numeric_limits<float>::infinity(),  
    -std::numeric_limits<float>::infinity(),        -std::numeric_limits<float>::infinity()
};

const float DataSet_1_32f::outputs::FMULADDV[32] = {
    2084.758300781f,        -4355117.000000000f,    -198563.906250000f,     -3381870.000000000f,
    -2286410.000000000f,    -1389128.750000000f,    -2235082.500000000f,    -13983005.000000000f,
    62125.066406250f,       9079022.000000000f,     4911102.000000000f,     1019438.625000000f,
    755574.750000000f,      10338054.000000000f,    6822554.000000000f,     7534961.500000000f,
    -2054053.000000000f,    -2609134.000000000f,    -4665218.000000000f,    5190203.000000000f,
    -15045068.000000000f,   -1742329.625000000f,    -136061.359375000f,     3730359.250000000f,
    10137588.000000000f,    -1733892.375000000f,    3724115.000000000f,     7216281.500000000f,
    4823864.500000000f,     -7748855.000000000f,    -7502858.500000000f,    -739936.000000000f
};

const float DataSet_1_32f::outputs::MFMULADDV[32] = {
    2084.758300781f,        2686.696777344f,        -136.265136719f,        -3381870.000000000f,
    -514.084472656f,        -1389128.750000000f,    -2235082.500000000f,    -3039.185791016f,
    1879.787597656f,        9079022.000000000f,     4911102.000000000f,     601.367187500f,
    755574.750000000f,      -2759.941406250f,       3421.887695313f,        7534961.500000000f,
    -1742.759521484f,       -2609134.000000000f,    -4665218.000000000f,    1645.405273438f,
    -15045068.000000000f,   -1560.869140625f,       -1027.100341797f,       3730359.250000000f,
    10137588.000000000f,    -636.768554688f,        -893.734375000f,        7216281.500000000f,
    4145.176757813f,        -7748855.000000000f,    -7502858.500000000f,    -2136.753417969f
};

const float DataSet_1_32f::outputs::FMULSUBV[32] = {
    1935.523071289f,        -4355104.000000000f,    -197294.031250000f,     -3375608.500000000f,
    -2293393.000000000f,    -1395623.500000000f,    -2233155.500000000f,    -13982473.000000000f,
    64659.628906250f,       9083826.000000000f,     4909080.000000000f,     1013504.250000000f,
    757880.375000000f,      10346136.000000000f,    6831787.000000000f,     7525387.500000000f,
    -2051406.750000000f,    -2609755.000000000f,    -4668588.000000000f,    5199848.000000000f,
    -15040132.000000000f,   -1742199.375000000f,    -126927.476562500f,     3725375.250000000f,
    10136326.000000000f,    -1740574.375000000f,    3716898.500000000f,     7209725.500000000f,
    4822111.500000000f,     -7747239.000000000f,    -7509085.500000000f,    -746211.500000000f
};

const float DataSet_1_32f::outputs::MFMULSUBV[32] = {
    1935.523071289f,        2686.696777344f,        -136.265136719f,        -3375608.500000000f,
    -514.084472656f,        -1395623.500000000f,    -2233155.500000000f,    -3039.185791016f,
    1879.787597656f,        9083826.000000000f,     4909080.000000000f,     601.367187500f,
    757880.375000000f,      -2759.941406250f,       3421.887695313f,        7525387.500000000f,
    -1742.759521484f,       -2609755.000000000f,    -4668588.000000000f,    1645.405273438f,
    -15040132.000000000f,   -1560.869140625f,       -1027.100341797f,       3725375.250000000f,
    10136326.000000000f,    -636.768554688f,        -893.734375000f,        7209725.500000000f,
    4145.176757813f,        -7747239.000000000f,    -7509085.500000000f,    -2136.753417969f
};

const float DataSet_1_32f::outputs::FADDMULV[32] = {
    -8995.037109375f,       -6667.427734375f,       -835744.875000000f,     -11394764.000000000f,
    13757245.000000000f,    5662770.500000000f,     -641101.125000000f,     -415345.187500000f,
    -2424956.250000000f,    15234815.000000000f,    -5809941.000000000f,    6799652.000000000f,
    3745567.000000000f,     26296126.000000000f,    -25008758.000000000f,   27132112.000000000f,
    747433.062500000f,      -991650.875000000f,     3309699.500000000f,     -23161910.000000000f,
    -2398815.500000000f,    28972.472656250f,       4106025.750000000f,     -12813964.000000000f,
    -4066579.250000000f,    6987508.000000000f,     -18246052.000000000f,   -19755876.000000000f,
    4652215.500000000f,     -2008169.250000000f,    -6100892.000000000f,    -5613438.500000000f
};

const float DataSet_1_32f::outputs::MFADDMULV[32] = {
    -8995.037109375f,       2686.696777344f,        -136.265136719f,        -11394764.000000000f,
    -514.084472656f,        5662770.500000000f,     -641101.125000000f,     -3039.185791016f,
    1879.787597656f,        15234815.000000000f,    -5809941.000000000f,    601.367187500f,
    3745567.000000000f,     -2759.941406250f,       3421.887695313f,        27132112.000000000f,
    -1742.759521484f,       -991650.875000000f,     3309699.500000000f,     1645.405273438f,
    -2398815.500000000f,    -1560.869140625f,       -1027.100341797f,       -12813964.000000000f,
    -4066579.250000000f,    -636.768554688f,        -893.734375000f,        -19755876.000000000f,
    4145.176757813f,        -2008169.250000000f,    -6100892.000000000f,    -2136.753417969f
};

const float DataSet_1_32f::outputs::FSUBMULV[32] = {
    -6011.860351563f,       -26950.390625000f,      1008784.562500000f,     -16195927.000000000f,
    -17347066.000000000f,   9528796.000000000f,     -2951124.000000000f,    2032002.000000000f,
    -2339482.750000000f,    -4745714.500000000f,    3697487.500000000f,     -3230944.000000000f,
    -3163291.000000000f,    -3989681.000000000f,    -6586814.000000000f,    6777184.000000000f,
    3864361.250000000f,     1410619.750000000f,     7996341.000000000f,     7291359.500000000f,
    -19292948.000000000f,   174431.328125000f,      5275390.500000000f,     -8461521.000000000f,
    -615150.000000000f,     -11242413.000000000f,   11796230.000000000f,    8961118.000000000f,
    2612943.500000000f,     -4925982.500000000f,    18117398.000000000f,    -7795803.500000000f
};

const float DataSet_1_32f::outputs::MFSUBMULV[32] = {
    -6011.860351563f,       2686.696777344f,        -136.265136719f,        -16195927.000000000f,
    -514.084472656f,        9528796.000000000f,     -2951124.000000000f,    -3039.185791016f,
    1879.787597656f,        -4745714.500000000f,    3697487.500000000f,     601.367187500f,
    -3163291.000000000f,    -2759.941406250f,       3421.887695313f,        6777184.000000000f,
    -1742.759521484f,       1410619.750000000f,     7996341.000000000f,     1645.405273438f,
    -19292948.000000000f,   -1560.869140625f,       -1027.100341797f,       -8461521.000000000f,
    -615150.000000000f,     -636.768554688f,        -893.734375000f,        8961118.000000000f,
    4145.176757813f,        -4925982.500000000f,    18117398.000000000f,    -2136.753417969f
};

const float DataSet_1_32f::outputs::MAXV[32] = {
    -19.989746094f,         2686.696777344f,        1452.528320313f,        4406.415039063f,
    4454.329101563f,        2339.091308594f,        1863.917968750f,        4600.817382813f,
    1879.787597656f,        -2183.446777344f,       -1044.496093750f,       1690.267578125f,
    -252.540527344f,        -2759.941406250f,       3421.887695313f,        3541.825195313f,
    1177.861816406f,        674.611816406f,         3355.357421875f,        3157.292480469f,
    4394.818359375f,        1116.214355469f,        128.024902344f,         -873.287109375f,
    -2733.695556641f,       2728.202148438f,        -893.734375000f,        -1646.626220703f,
    4145.176757813f,        4291.054687500f,        1929.837890625f,        347.758300781f
};

const float DataSet_1_32f::outputs::MMAXV[32] = {
    -19.989746094f,         2686.696777344f,        -136.265136719f,        4406.415039063f,
    -514.084472656f,        2339.091308594f,        1863.917968750f,        -3039.185791016f,
    1879.787597656f,        -2183.446777344f,       -1044.496093750f,       601.367187500f,
    -252.540527344f,        -2759.941406250f,       3421.887695313f,        3541.825195313f,
    -1742.759521484f,       674.611816406f,         3355.357421875f,        1645.405273438f,
    4394.818359375f,        -1560.869140625f,       -1027.100341797f,       -873.287109375f,
    -2733.695556641f,       -636.768554688f,        -893.734375000f,        -1646.626220703f,
    4145.176757813f,        4291.054687500f,        1929.837890625f,        -2136.753417969f
};

const float DataSet_1_32f::outputs::MAXS[32] = {
    255.897460938f,         2686.696777344f,        255.897460938f,         4406.415039063f,
    255.897460938f,         2339.091308594f,        1863.917968750f,        255.897460938f,
    1879.787597656f,        255.897460938f,         255.897460938f,         601.367187500f,
    255.897460938f,         255.897460938f,         3421.887695313f,        3541.825195313f,
    255.897460938f,         674.611816406f,         3355.357421875f,        1645.405273438f,
    4394.818359375f,        255.897460938f,         255.897460938f,         255.897460938f,
    255.897460938f,         255.897460938f,         255.897460938f,         255.897460938f,
    4145.176757813f,        4291.054687500f,        1929.837890625f,        255.897460938f
};

const float DataSet_1_32f::outputs::MMAXS[32] = {
    255.897460938f,         2686.696777344f,        -136.265136719f,        4406.415039063f,
    -514.084472656f,        2339.091308594f,        1863.917968750f,        -3039.185791016f,
    1879.787597656f,        255.897460938f,         255.897460938f,         601.367187500f,
    255.897460938f,         -2759.941406250f,       3421.887695313f,        3541.825195313f,
    -1742.759521484f,       674.611816406f,         3355.357421875f,        1645.405273438f,
    4394.818359375f,        -1560.869140625f,       -1027.100341797f,       255.897460938f,
    255.897460938f,         -636.768554688f,        -893.734375000f,        255.897460938f,
    4145.176757813f,        4291.054687500f,        1929.837890625f,        -2136.753417969f
};

const float DataSet_1_32f::outputs::MINV[32] = {
    -100.558593750f,        -1620.990722656f,       -136.265136719f,        -766.777343750f,
    -514.084472656f,        -595.263671875f,        -1198.614501953f,       -3039.185791016f,
    33.723144531f,          -4159.214843750f,       -4700.918457031f,       601.367187500f,
    -2996.459960938f,       -3747.215332031f,       1995.147460938f,        2126.071777344f,
    -1742.759521484f,       -3868.068359375f,       -1390.881103516f,       1645.405273438f,
    -3422.803466797f,       -1560.869140625f,       -1027.100341797f,       -4268.776367188f,
    -3708.151367188f,       -636.768554688f,        -4162.877441406f,       -4380.474121094f,
    1163.518066406f,        -1805.627685547f,       -3889.431396484f,       -2136.753417969f
};

const float DataSet_1_32f::outputs::MMINV[32] = {
    -100.558593750f,        2686.696777344f,        -136.265136719f,        -766.777343750f,
    -514.084472656f,        -595.263671875f,        -1198.614501953f,       -3039.185791016f,
    1879.787597656f,        -4159.214843750f,       -4700.918457031f,       601.367187500f,
    -2996.459960938f,       -2759.941406250f,       3421.887695313f,        2126.071777344f,
    -1742.759521484f,       -3868.068359375f,       -1390.881103516f,       1645.405273438f,
    -3422.803466797f,       -1560.869140625f,       -1027.100341797f,       -4268.776367188f,
    -3708.151367188f,       -636.768554688f,        -893.734375000f,        -4380.474121094f,
    4145.176757813f,        -1805.627685547f,       -3889.431396484f,       -2136.753417969f
};

const float DataSet_1_32f::outputs::MINS[32] = {
    -100.558593750f,        255.897460938f,         -136.265136719f,        255.897460938f,
    -514.084472656f,        255.897460938f,         255.897460938f,         -3039.185791016f,
    255.897460938f,         -2183.446777344f,       -1044.496093750f,       255.897460938f,
    -252.540527344f,        -2759.941406250f,       255.897460938f,         255.897460938f,
    -1742.759521484f,       255.897460938f,         255.897460938f,         255.897460938f,
    255.897460938f,         -1560.869140625f,       -1027.100341797f,       -4268.776367188f,
    -3708.151367188f,       -636.768554688f,        -893.734375000f,        -1646.626220703f,
    255.897460938f,         255.897460938f,         255.897460938f,         -2136.753417969f
};

const float DataSet_1_32f::outputs::MMINS[32] = {
    -100.558593750f,        2686.696777344f,        -136.265136719f,        255.897460938f,
    -514.084472656f,        255.897460938f,         255.897460938f,         -3039.185791016f,
    1879.787597656f,        -2183.446777344f,       -1044.496093750f,       601.367187500f,
    -252.540527344f,        -2759.941406250f,       3421.887695313f,        255.897460938f,
    -1742.759521484f,       255.897460938f,         255.897460938f,         1645.405273438f,
    255.897460938f,         -1560.869140625f,       -1027.100341797f,       -4268.776367188f,
    -3708.151367188f,       -636.768554688f,        -893.734375000f,        -1646.626220703f,
    4145.176757813f,        255.897460938f,         255.897460938f,         -2136.753417969f
};

const float DataSet_1_32f::outputs::HMAX[32] = {
    -100.558593750f,        2686.696777344f,        2686.696777344f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f
};

const float DataSet_1_32f::outputs::MHMAX[32] = {
    -100.558593750f,        -100.558593750f,        -100.558593750f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f,
    4406.415039063f,        4406.415039063f,        4406.415039063f,        4406.415039063f
};

const float DataSet_1_32f::outputs::HMIN[32] = {
    -100.558593750f,        -100.558593750f,        -136.265136719f,        -136.265136719f,
    -514.084472656f,        -514.084472656f,        -514.084472656f,        -3039.185791016f,
    -3039.185791016f,       -3039.185791016f,       -3039.185791016f,       -3039.185791016f,
    -3039.185791016f,       -3039.185791016f,       -3039.185791016f,       -3039.185791016f,
    -3039.185791016f,       -3039.185791016f,       -3039.185791016f,       -3039.185791016f,
    -3039.185791016f,       -3039.185791016f,       -3039.185791016f,       -4268.776367188f,
    -4268.776367188f,       -4268.776367188f,       -4268.776367188f,       -4268.776367188f,
    -4268.776367188f,       -4268.776367188f,       -4268.776367188f,       -4268.776367188f
};

const float DataSet_1_32f::outputs::MHMIN[32] = {
    -100.558593750f,        -100.558593750f,        -100.558593750f,        -100.558593750f,
    -100.558593750f,        -100.558593750f,        -100.558593750f,        -100.558593750f,
    -100.558593750f,        -2183.446777344f,       -2183.446777344f,       -2183.446777344f,
    -2183.446777344f,       -2183.446777344f,       -2183.446777344f,       -2183.446777344f,
    -2183.446777344f,       -2183.446777344f,       -2183.446777344f,       -2183.446777344f,
    -2183.446777344f,       -2183.446777344f,       -2183.446777344f,       -4268.776367188f,
    -4268.776367188f,       -4268.776367188f,       -4268.776367188f,       -4268.776367188f,
    -4268.776367188f,       -4268.776367188f,       -4268.776367188f,       -4268.776367188f
};

const float DataSet_1_32f::outputs::NEG[32] = {
    100.558593750f,         -2686.696777344f,       136.265136719f,         -4406.415039063f,
    514.084472656f,         -2339.091308594f,       -1863.917968750f,       3039.185791016f,
    -1879.787597656f,       2183.446777344f,        1044.496093750f,        -601.367187500f,
    252.540527344f,         2759.941406250f,        -3421.887695313f,       -3541.825195313f,
    1742.759521484f,        -674.611816406f,        -3355.357421875f,       -1645.405273438f,
    -4394.818359375f,       1560.869140625f,        1027.100341797f,        4268.776367188f,
    3708.151367188f,        636.768554688f,         893.734375000f,         1646.626220703f,
    -4145.176757813f,       -4291.054687500f,       -1929.837890625f,       2136.753417969f
};

const float DataSet_1_32f::outputs::MNEG[32] = {
    100.558593750f,         2686.696777344f,        -136.265136719f,        -4406.415039063f,
    -514.084472656f,        -2339.091308594f,       -1863.917968750f,       -3039.185791016f,
    1879.787597656f,        2183.446777344f,        1044.496093750f,        601.367187500f,
    252.540527344f,         -2759.941406250f,       3421.887695313f,        -3541.825195313f,
    -1742.759521484f,       -674.611816406f,        -3355.357421875f,       1645.405273438f,
    -4394.818359375f,       -1560.869140625f,       -1027.100341797f,       4268.776367188f,
    3708.151367188f,        -636.768554688f,        -893.734375000f,        1646.626220703f,
    4145.176757813f,        -4291.054687500f,       -1929.837890625f,       -2136.753417969f
};

const float DataSet_1_32f::outputs::ABS[32] = {
    100.558593750f,         2686.696777344f,        136.265136719f,         4406.415039063f,
    514.084472656f,         2339.091308594f,        1863.917968750f,        3039.185791016f,
    1879.787597656f,        2183.446777344f,        1044.496093750f,        601.367187500f,
    252.540527344f,         2759.941406250f,        3421.887695313f,        3541.825195313f,
    1742.759521484f,        674.611816406f,         3355.357421875f,        1645.405273438f,
    4394.818359375f,        1560.869140625f,        1027.100341797f,        4268.776367188f,
    3708.151367188f,        636.768554688f,         893.734375000f,         1646.626220703f,
    4145.176757813f,        4291.054687500f,        1929.837890625f,        2136.753417969f
};

const float DataSet_1_32f::outputs::MABS[32] = {
    100.558593750f,         2686.696777344f,        -136.265136719f,        4406.415039063f,
    -514.084472656f,        2339.091308594f,        1863.917968750f,        -3039.185791016f,
    1879.787597656f,        2183.446777344f,        1044.496093750f,        601.367187500f,
    252.540527344f,         -2759.941406250f,       3421.887695313f,        3541.825195313f,
    -1742.759521484f,       674.611816406f,         3355.357421875f,        1645.405273438f,
    4394.818359375f,        -1560.869140625f,       -1027.100341797f,       4268.776367188f,
    3708.151367188f,        -636.768554688f,        -893.734375000f,        1646.626220703f,
    4145.176757813f,        4291.054687500f,        1929.837890625f,        -2136.753417969f
};

const float DataSet_1_32f::outputs::SQR[32] = {
    10112.031250000f,       7218339.500000000f,     18568.187500000f,       19416494.000000000f,
    264282.843750000f,      5471348.000000000f,     3474190.250000000f,     9236650.000000000f,
    3533601.500000000f,     4767440.000000000f,     1090972.125000000f,     361642.500000000f,
    63776.718750000f,       7617276.500000000f,     11709315.000000000f,    12544526.000000000f,
    3037210.750000000f,     455101.093750000f,      11258423.000000000f,    2707358.500000000f,
    19314428.000000000f,    2436312.500000000f,     1054935.125000000f,     18222452.000000000f,
    13750387.000000000f,    405474.187500000f,      798761.125000000f,      2711378.000000000f,
    17182490.000000000f,    18413150.000000000f,    3724274.250000000f,     4565715.000000000f
};

const float DataSet_1_32f::outputs::MSQR[32] = {
    10112.031250000f,       2686.696777344f,        -136.265136719f,        19416494.000000000f,
    -514.084472656f,        5471348.000000000f,     3474190.250000000f,     -3039.185791016f,
    1879.787597656f,        4767440.000000000f,     1090972.125000000f,     601.367187500f,
    63776.718750000f,       -2759.941406250f,       3421.887695313f,        12544526.000000000f,
    -1742.759521484f,       455101.093750000f,      11258423.000000000f,    1645.405273438f,
    19314428.000000000f,    -1560.869140625f,       -1027.100341797f,       18222452.000000000f,
    13750387.000000000f,    -636.768554688f,        -893.734375000f,        2711378.000000000f,
    4145.176757813f,        18413150.000000000f,    3724274.250000000f,     -2136.753417969f
};

// SQRT(ABS(g_Init1_l));
const float DataSet_1_32f::outputs::SQRT[32] = {
    10.027891159f,  51.833354950f,  11.673265457f,  66.380836487f,
    22.673431396f,  48.364154816f,  43.173114777f,  55.128810883f,
    43.356517792f,  46.727367401f,  32.318664551f,  24.522789001f,
    15.891523361f,  52.535144806f,  58.496902466f,  59.513237000f,
    41.746372223f,  25.973289490f,  57.925445557f,  40.563594818f,
    66.293426514f,  39.507835388f,  32.048404694f,  65.335876465f,
    60.894592285f,  25.234273911f,  29.895391464f,  40.578643799f,
    64.383049011f,  65.506141663f,  43.929920197f,  46.225028992f
};

const float DataSet_1_32f::outputs::MSQRT[32] = {
    10.027891159f,          2686.696777344f,        -136.265136719f,        66.380836487f,
    -514.084472656f,        48.364154816f,          43.173114777f,          -3039.185791016f,
    1879.787597656f,        46.727367401f,          32.318664551f,          601.367187500f,
    15.891523361f,          -2759.941406250f,       3421.887695313f,        59.513237000f,
    -1742.759521484f,       25.973289490f,          57.925445557f,          1645.405273438f,
    66.293426514f,          -1560.869140625f,       -1027.100341797f,       65.335876465f,
    60.894592285f,          -636.768554688f,        -893.734375000f,        40.578643799f,
    4145.176757813f,        65.506141663f,          43.929920197f,          -2136.753417969f
};

// POWV
// MPOWV
// POWS
// MPOWS

const float DataSet_1_32f::outputs::ROUND[32] = {
    -100.000000000f,        2687.000000000f,        -135.000000000f,        4406.000000000f,
    -513.000000000f,        2339.000000000f,        1864.000000000f,        -3038.000000000f,
    1880.000000000f,        -2182.000000000f,       -1043.000000000f,       601.000000000f,
    -252.000000000f,        -2759.000000000f,       3422.000000000f,        3542.000000000f,
    -1742.000000000f,       675.000000000f,         3355.000000000f,        1645.000000000f,
    4395.000000000f,        -1560.000000000f,       -1026.000000000f,       -4268.000000000f,
    -3707.000000000f,       -636.000000000f,        -893.000000000f,        -1646.000000000f,
    4145.000000000f,        4291.000000000f,        1930.000000000f,        -2136.000000000f
};

const float DataSet_1_32f::outputs::MROUND[32] = {
    -100.000000000f,        2686.696777344f,        -136.265136719f,        4406.000000000f,
    -514.084472656f,        2339.000000000f,        1864.000000000f,        -3039.185791016f,
    1879.787597656f,        -2182.000000000f,       -1043.000000000f,       601.367187500f,
    -252.000000000f,        -2759.941406250f,       3421.887695313f,        3542.000000000f,
    -1742.759521484f,       675.000000000f,         3355.000000000f,        1645.405273438f,
    4395.000000000f,        -1560.869140625f,       -1027.100341797f,       -4268.000000000f,
    -3707.000000000f,       -636.768554688f,        -893.734375000f,        -1646.000000000f,
    4145.176757813f,        4291.000000000f,        1930.000000000f,        -2136.753417969f
};

const int32_t DataSet_1_32f::outputs::TRUNC[32] = {
    -100,   2686,   -136,   4406,
    -514,   2339,   1863,   -3039,
    1879,   -2183,  -1044,  601,
    -252,   -2759,  3421,   3541,
    -1742,  674,    3355,   1645,
    4394,   -1560,  -1027,  -4268,
    -3708,  -636,   -893,   -1646,
    4145,   4291,   1929,   -2136
};

const int32_t DataSet_1_32f::outputs::MTRUNC[32] = {
    -100,   0,      0,      4406,
    0,      2339,   1863,   0,
    0,      -2183,  -1044,  0,
    -252,   0,      0,      3541,
    0,      674,    3355,   0,
    4394,   0,      0,      -4268,
    -3708,  0,      0,      -1646,
    0,      4291,   1929,   0
};

const float DataSet_1_32f::outputs::FLOOR[32] = {
    -101.000000000f,        2686.000000000f,        -137.000000000f,        4406.000000000f,
    -515.000000000f,        2339.000000000f,        1863.000000000f,        -3040.000000000f,
    1879.000000000f,        -2184.000000000f,       -1045.000000000f,       601.000000000f,
    -253.000000000f,        -2760.000000000f,       3421.000000000f,        3541.000000000f,
    -1743.000000000f,       674.000000000f,         3355.000000000f,        1645.000000000f,
    4394.000000000f,        -1561.000000000f,       -1028.000000000f,       -4269.000000000f,
    -3709.000000000f,       -637.000000000f,        -894.000000000f,        -1647.000000000f,
    4145.000000000f,        4291.000000000f,        1929.000000000f,        -2137.000000000f
};

const float DataSet_1_32f::outputs::MFLOOR[32] = {
    -101.000000000f,        2686.696777344f,        -136.265136719f,        4406.000000000f,
    -514.084472656f,        2339.000000000f,        1863.000000000f,        -3039.185791016f,
    1879.787597656f,        -2184.000000000f,       -1045.000000000f,       601.367187500f,
    -253.000000000f,        -2759.941406250f,       3421.887695313f,        3541.000000000f,
    -1742.759521484f,       674.000000000f,         3355.000000000f,        1645.405273438f,
    4394.000000000f,        -1560.869140625f,       -1027.100341797f,       -4269.000000000f,
    -3709.000000000f,       -636.768554688f,        -893.734375000f,        -1647.000000000f,
    4145.176757813f,        4291.000000000f,        1929.000000000f,        -2136.753417969f
};

const float DataSet_1_32f::outputs::CEIL[32] = {
    -100.000000000f,        2687.000000000f,        -136.000000000f,        4407.000000000f,
    -514.000000000f,        2340.000000000f,        1864.000000000f,        -3039.000000000f,
    1880.000000000f,        -2183.000000000f,       -1044.000000000f,       602.000000000f,
    -252.000000000f,        -2759.000000000f,       3422.000000000f,        3542.000000000f,
    -1742.000000000f,       675.000000000f,         3356.000000000f,        1646.000000000f,
    4395.000000000f,        -1560.000000000f,       -1027.000000000f,       -4268.000000000f,
    -3708.000000000f,       -636.000000000f,        -893.000000000f,        -1646.000000000f,
    4146.000000000f,        4292.000000000f,        1930.000000000f,        -2136.000000000f
};

const float DataSet_1_32f::outputs::MCEIL[32] = {
    -100.000000000f,        2686.696777344f,        -136.265136719f,        4407.000000000f,
    -514.084472656f,        2340.000000000f,        1864.000000000f,        -3039.185791016f,
    1879.787597656f,        -2183.000000000f,       -1044.000000000f,       601.367187500f,
    -252.000000000f,        -2759.941406250f,       3421.887695313f,        3542.000000000f,
    -1742.759521484f,       675.000000000f,         3356.000000000f,        1645.405273438f,
    4395.000000000f,        -1560.869140625f,       -1027.100341797f,       -4268.000000000f,
    -3708.000000000f,       -636.768554688f,        -893.734375000f,        -1646.000000000f,
    4145.176757813f,        4292.000000000f,        1930.000000000f,        -2136.753417969f
};

const bool DataSet_1_32f::outputs::ISFIN[32] = {
    true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true
};

const bool DataSet_1_32f::outputs::ISINF[32] = {
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false
};

const bool DataSet_1_32f::outputs::ISAN[32] = {
    true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true
};

const bool DataSet_1_32f::outputs::ISNAN[32] = {
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false
};

const bool DataSet_1_32f::outputs::ISNORM[32] = {
    true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true
};

const bool DataSet_1_32f::outputs::ISSUB[32] = {
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false
};

const bool DataSet_1_32f::outputs::ISZERO[32] = { 
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false
};

const bool DataSet_1_32f::outputs::ISZEROSUB[32] = {
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false
};

const float DataSet_1_32f::outputs::SIN[32] = {
    -0.027625320f,  -0.593224645f,  0.923325717f,   0.945606470f,
    0.907259941f,   0.984625757f,   -0.815460980f,  0.953816533f,
    0.897994757f,   0.039872527f,   -0.996518910f,  -0.969452977f,
    -0.936711133f,  -0.998634756f,  -0.639075398f,  -0.949071229f,
    -0.734141350f,  0.738338947f,   0.136044651f,   -0.709844232f,
    0.266503006f,   -0.481537551f,  -0.199116156f,  -0.603632152f,
    -0.878176212f,  -0.827563822f,  -0.998812675f,  -0.418388397f,
    -0.988050282f,  -0.353095174f,  0.783327758f,   -0.453254938f
};

const float DataSet_1_32f::outputs::MSIN[32] = {
    -0.027625320f,          2686.696777344f,        -136.265136719f,        0.945606470f,
    -514.084472656f,        0.984625757f,           -0.815460980f,          -3039.185791016f,
    1879.787597656f,        0.039872527f,           -0.996518910f,          601.367187500f,
    -0.936711133f,          -2759.941406250f,       3421.887695313f,        -0.949071229f,
    -1742.759521484f,       0.738338947f,           0.136044651f,           1645.405273438f,
    0.266503006f,           -1560.869140625f,       -1027.100341797f,       -0.603632152f,
    -0.878176212f,          -636.768554688f,        -893.734375000f,        -0.418388397f,
    4145.176757813f,        -0.353095174f,          0.783327758f,           -2136.753417969f
};

const float DataSet_1_32f::outputs::COS[32] = {
    0.999618351f,   -0.805036962f,  -0.384017706f,  -0.325312704f,
    0.420570284f,   -0.174677283f,  -0.578812063f,  -0.300389826f,
    0.440006137f,   -0.999204755f,  0.083366700f,   -0.245277241f,
    0.350103199f,   -0.052236285f,  -0.769144118f,  -0.315061659f,
    -0.678996623f,  -0.674429834f,  0.990702689f,   0.704358697f,
    -0.963834107f,  -0.876425445f,  -0.979975879f,  -0.797262967f,
    0.478337288f,   -0.561371684f,  0.048715658f,   0.908268213f,
    -0.154131711f,  0.935587406f,   0.621608913f,   0.891380906f
};

const float DataSet_1_32f::outputs::MCOS[32] = {
    0.999618351f,           2686.696777344f,        -136.265136719f,        -0.325312704f,
    -514.084472656f,        -0.174677283f,          -0.578812063f,          -3039.185791016f,
    1879.787597656f,        -0.999204755f,          0.083366700f,           601.367187500f,
    0.350103199f,           -2759.941406250f,       3421.887695313f,        -0.315061659f,
    -1742.759521484f,       -0.674429834f,          0.990702689f,           1645.405273438f,
    -0.963834107f,          -1560.869140625f,       -1027.100341797f,       -0.797262967f,
    0.478337288f,           -636.768554688f,        -893.734375000f,        0.908268213f,
    4145.176757813f,        0.935587406f,           0.621608913f,           -2136.753417969f
};


const float DataSet_1_32f::outputs::TAN[32] = {
    -0.027635867f,  0.736891150f,   -2.404383183f,  -2.906761646f,
    2.157213688f,   -5.636827946f,  1.408852816f,   -3.175262451f,
    2.040868759f,   -0.039904259f,  -11.953441620f, 3.952478170f,
    -2.675528765f,  19.117645264f,  0.830891669f,   3.012334824f,
    1.081215024f,   -1.094760180f,  0.137321368f,   -1.007788062f,
    -0.276502997f,  0.549433529f,   0.203184739f,   0.757130504f,
    -1.835893154f,  1.474181652f,   -20.502908707f, -0.460644096f,
    6.410428524f,   -0.377404779f,  1.260161638f,   -0.508486211f
};

const float DataSet_1_32f::outputs::MTAN[32] = {
    -0.027635867f,          2686.696777344f,        -136.265136719f,        -2.906761646f,
    -514.084472656f,        -5.636827946f,          1.408852816f,           -3039.185791016f,
    1879.787597656f,        -0.039904259f,          -11.953441620f,         601.367187500f,
    -2.675528765f,          -2759.941406250f,       3421.887695313f,        3.012334824f,
    -1742.759521484f,       -1.094760180f,          0.137321368f,           1645.405273438f,
    -0.276502997f,          -1560.869140625f,       -1027.100341797f,       0.757130504f,
    -1.835893154f,          -636.768554688f,        -893.734375000f,        -0.460644096f,
    4145.176757813f,        -0.377404779f,          1.260161638f,           -2136.753417969f
};

const float DataSet_1_32f::outputs::CTAN[32] = {
    -36.184860229f, 1.357052565f,   -0.415907085f,  -0.344025463f,
    0.463560939f,   -0.177404746f,  0.709797382f,   -0.314934611f,
    0.489987403f,   -25.059982300f, -0.083657913f,  0.253005832f,
    -0.373757899f,  0.052307699f,   1.203526378f,   0.331968397f,
    0.924885392f,   -0.913442075f,  7.282187939f,   -0.992272139f,
    -3.616597414f,  1.820056438f,   4.921629429f,   1.320776224f,
    -0.544694006f,  0.678342462f,   -0.048773568f,  -2.170873404f,
    0.155995816f,   -2.649674892f,  0.793549001f,   -1.966621637f
};

const float DataSet_1_32f::outputs::MCTAN[32] = {
    -36.184860229f,         2686.696777344f,        -136.265136719f,        -0.344025463f,
    -514.084472656f,        -0.177404746f,          0.709797382f,           -3039.185791016f,
    1879.787597656f,        -25.059982300f,         -0.083657913f,          601.367187500f,
    -0.373757899f,          -2759.941406250f,       3421.887695313f,        0.331968397f,
    -1742.759521484f,       -0.913442075f,          7.282187939f,           1645.405273438f,
    -3.616597414f,          -1560.869140625f,       -1027.100341797f,       1.320776224f,
    -0.544694006f,          -636.768554688f,        -893.734375000f,        -2.170873404f,
    4145.176757813f,        -2.649674892f,          0.793549001f,           -2136.753417969f
};

const uint32_t DataSet_1_32f::outputs::FTOU[32] = {
    0xffffff9c,     0x00000a7e,     0xffffff78,     0x00001136,
    0xfffffdfe,     0x00000923,     0x00000747,     0xfffff421,
    0x00000757,     0xfffff779,     0xfffffbec,     0x00000259,
    0xffffff04,     0xfffff539,     0x00000d5d,     0x00000dd5,
    0xfffff932,     0x000002a2,     0x00000d1b,     0x0000066d,
    0x0000112a,     0xfffff9e8,     0xfffffbfd,     0xffffef54,
    0xfffff184,     0xfffffd84,     0xfffffc83,     0xfffff992,
    0x00001031,     0x000010c3,     0x00000789,     0xfffff7a8
};

const int32_t DataSet_1_32f::outputs::FTOI[32] = {
    -100,   2686,   -136,   4406,
    -514,   2339,   1863,   -3039,
    1879,   -2183,  -1044,  601,
    -252,   -2759,  3421,   3541,
    -1742,  674,    3355,   1645,
    4394,   -1560,  -1027,  -4268,
    -3708,  -636,   -893,   -1646,
    4145,   4291,   1929,   -2136
};
