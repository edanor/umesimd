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

#include "UMEUnitTestDataSets16.h"

#include <limits>

const uint16_t DataSet_1_16u::inputs::inputA[64] = {
    61968, 27063, 31277, 30445, 42314, 23190, 17687, 56878,
    42539, 53306, 6977,  201,   19778, 6222,  3601,  17175,
    9655,  10268, 40025, 18577, 6045,  44489, 61092, 17417,
    32431, 41860, 56665, 59650, 57058, 46484, 37628, 62628,
    5921,  52314, 1180,  585,   36293, 23901, 31544, 54323,
    6500,  45926, 61828, 40834, 16937, 61175, 12830, 4872,
    55562, 27109, 30778, 49889, 50092, 3924,  24004, 49241,
    30581, 36168, 43153, 35033, 744,   41859, 14226, 45959
};

const uint16_t DataSet_1_16u::inputs::inputB[64] = {
    25647, 64293, 9824,  12754, 35639, 40441, 38916, 48922,
    33870, 31995, 62750, 36048, 12613, 19841, 8556,  18442,
    43489, 17999, 47898, 26745, 28457, 61246, 20310, 37176,
    54625, 16543, 33874, 57415, 64975, 50728, 1491,  62068,
    33022, 18442, 12817, 28401, 23909, 41449, 50059, 22277,
    54238, 35378, 4556,  52114, 45421, 35909, 65136, 41626,
    29657, 40267, 8003,  28613, 65397, 36469, 36199, 48289,
    19261, 53203, 50459, 18955, 32885, 18439, 58953, 6655
};

const uint16_t DataSet_1_16u::inputs::inputC[64] = {
    37878, 57155, 35231, 49700, 21772, 52077, 48494, 56016,
    56229, 8605,  60497, 7341,  10568, 3151,  21244, 38777,
    10584, 60659, 6024,  38104, 53152, 29243, 56127, 26027,
    43059, 59955, 18796, 60859, 6757,  19229, 29121, 34957,
    57577, 11244, 17325, 54958, 8888,  36159, 3359,  2850,
    46130, 13653, 56805, 58851, 56260, 54625, 17465, 42203,
    4860,  50238, 64933, 37331, 64507, 11384, 7051,  55067,
    15325, 52926, 19073, 30010, 53119, 51705, 23765, 11678
};

const uint16_t DataSet_1_16u::inputs::inputShiftA[64] = {
    3,  15, 6,  14, 10, 1,  2,  12,
    8,  3,  3,  10, 3,  13, 3,  11,
    1,  14, 1,  9,  10, 6,  9,  6,
    8,  8,  7,  9,  12, 2,  1,  14,
    11, 4,  2,  2,  9,  2,  12, 7,
    13, 4,  10, 14, 13, 9,  13, 8,
    10, 2,  13, 10, 2,  13, 4,  14,
    14, 9,  1,  6,  3,  4,  14, 3
};

const uint16_t DataSet_1_16u::inputs::scalarA = 28734;
const uint16_t DataSet_1_16u::inputs::inputShiftScalarA = 9;

const bool DataSet_1_16u::inputs::maskA[64] = {
    false,                              // 1
    true,                               // 2
    true,   false,                      // 4
    true,   false,  false,  true,       // 8
    true,   false,  false,  true,
    false,  true,   true,   false,      // 16
    false,  true,   true,   false,
    true,   false,  false,  true,
    true,   false,  false,  true,
    false,  true,   true,   false,      // 32
    false,  true,   true,   false,
    true,   false,  false,  true,
    true,   false,  false,  true,
    false,  true,   true,   false,
    false,  true,   true,   false,
    true,   false,  false,  true,
    true,   false,  false,  true,
    false,  true,   true,   false       // 64
};

const uint16_t DataSet_1_16u::outputs::ADDV[64] = {
    0x563f, 0x64dc, 0xa08d, 0xa8bf, 0x3081, 0xf88f, 0xdd1b, 0x9d48,
    0x2a79, 0x4d35, 0x105f, 0x8d99, 0x7e87, 0x65cf, 0x2f7d, 0x8b21,
    0xcf98, 0x6e6b, 0x5773, 0xb10a, 0x86c6, 0x9d07, 0x3dfa, 0xd541,
    0x5410, 0xe423, 0x61ab, 0xc949, 0xdcb1, 0x7bbc, 0x98cf, 0xe718,
    0x981f, 0x1464, 0x36ad, 0x713a, 0xeb2a, 0xff46, 0x3ec3, 0x2b38,
    0xed42, 0x3d98, 0x0350, 0x6b14, 0xf396, 0x7b3c, 0x308e, 0xb5a2,
    0x4ce3, 0x0730, 0x977d, 0x32a6, 0xc321, 0x9dc9, 0xeb2b, 0x7cfa,
    0xc2b2, 0x5d1b, 0x6dac, 0xd2e4, 0x835d, 0xeb8a, 0x1ddb, 0xcd86
};

const uint16_t DataSet_1_16u::outputs::MADDV[64] = {
    0xf210, 0x64dc, 0xa08d, 0x76ed, 0x3081, 0x5a96, 0x4517, 0x9d48,
    0x2a79, 0xd03a, 0x1b41, 0x8d99, 0x4d42, 0x65cf, 0x2f7d, 0x4317,
    0x25b7, 0x6e6b, 0x5773, 0x4891, 0x86c6, 0xadc9, 0xeea4, 0xd541,
    0x5410, 0xa384, 0xdd59, 0xc949, 0xdee2, 0x7bbc, 0x98cf, 0xf4a4,
    0x1721, 0x1464, 0x36ad, 0x0249, 0xeb2a, 0x5d5d, 0x7b38, 0x2b38,
    0xed42, 0xb366, 0xf184, 0x6b14, 0x4229, 0x7b3c, 0x308e, 0x1308,
    0xd90a, 0x0730, 0x977d, 0xc2e1, 0xc321, 0x0f54, 0x5dc4, 0x7cfa,
    0xc2b2, 0x8d48, 0xa891, 0xd2e4, 0x02e8, 0xeb8a, 0x1ddb, 0xb387
};

const uint16_t DataSet_1_16u::outputs::ADDS[64] = {
    0x624e, 0xd9f5, 0xea6b, 0xe72b, 0x1588, 0xcad4, 0xb555, 0x4e6c,
    0x1669, 0x4078, 0x8b7f, 0x7107, 0xbd80, 0x888c, 0x7e4f, 0xb355,
    0x95f5, 0x985a, 0x0c97, 0xb8cf, 0x87db, 0x1e07, 0x5ee2, 0xb447,
    0xeeed, 0x13c2, 0x4d97, 0x5940, 0x4f20, 0x25d2, 0x033a, 0x64e2,
    0x875f, 0x3c98, 0x74da, 0x7287, 0xfe03, 0xcd9b, 0xeb76, 0x4471,
    0x89a2, 0x23a4, 0x61c2, 0x0fc0, 0xb267, 0x5f35, 0xa25c, 0x8346,
    0x4948, 0xda23, 0xe878, 0x331f, 0x33ea, 0x7f92, 0xce02, 0x3097,
    0xe7b3, 0xfd86, 0x18cf, 0xf917, 0x7326, 0x13c1, 0xa7d0, 0x23c5
};

const uint16_t DataSet_1_16u::outputs::MADDS[64] = {
    0xf210, 0xd9f5, 0xea6b, 0x76ed, 0x1588, 0x5a96, 0x4517, 0x4e6c,
    0x1669, 0xd03a, 0x1b41, 0x7107, 0x4d42, 0x888c, 0x7e4f, 0x4317,
    0x25b7, 0x985a, 0x0c97, 0x4891, 0x87db, 0xadc9, 0xeea4, 0xb447,
    0xeeed, 0xa384, 0xdd59, 0x5940, 0xdee2, 0x25d2, 0x033a, 0xf4a4,
    0x1721, 0x3c98, 0x74da, 0x0249, 0xfe03, 0x5d5d, 0x7b38, 0x4471,
    0x89a2, 0xb366, 0xf184, 0x0fc0, 0x4229, 0x5f35, 0xa25c, 0x1308,
    0xd90a, 0xda23, 0xe878, 0xc2e1, 0x33ea, 0x0f54, 0x5dc4, 0x3097,
    0xe7b3, 0x8d48, 0xa891, 0xf917, 0x02e8, 0x13c1, 0xa7d0, 0xb387
};

const uint16_t DataSet_1_16u::outputs::POSTPREFINC[64] = {
    0xf211, 0x69b8, 0x7a2e, 0x76ee, 0xa54b, 0x5a97, 0x4518, 0xde2f,
    0xa62c, 0xd03b, 0x1b42, 0x00ca, 0x4d43, 0x184f, 0x0e12, 0x4318,
    0x25b8, 0x281d, 0x9c5a, 0x4892, 0x179e, 0xadca, 0xeea5, 0x440a,
    0x7eb0, 0xa385, 0xdd5a, 0xe903, 0xdee3, 0xb595, 0x92fd, 0xf4a5,
    0x1722, 0xcc5b, 0x049d, 0x024a, 0x8dc6, 0x5d5e, 0x7b39, 0xd434,
    0x1965, 0xb367, 0xf185, 0x9f83, 0x422a, 0xeef8, 0x321f, 0x1309,
    0xd90b, 0x69e6, 0x783b, 0xc2e2, 0xc3ad, 0x0f55, 0x5dc5, 0xc05a,
    0x7776, 0x8d49, 0xa892, 0x88da, 0x02e9, 0xa384, 0x3793, 0xb388
};

const uint16_t DataSet_1_16u::outputs::MPOSTPREFINC[64] = {
    0xf210, 0x69b8, 0x7a2e, 0x76ed, 0xa54b, 0x5a96, 0x4517, 0xde2f,
    0xa62c, 0xd03a, 0x1b41, 0x00ca, 0x4d42, 0x184f, 0x0e12, 0x4317,
    0x25b7, 0x281d, 0x9c5a, 0x4891, 0x179e, 0xadc9, 0xeea4, 0x440a,
    0x7eb0, 0xa384, 0xdd59, 0xe903, 0xdee2, 0xb595, 0x92fd, 0xf4a4,
    0x1721, 0xcc5b, 0x049d, 0x0249, 0x8dc6, 0x5d5d, 0x7b38, 0xd434,
    0x1965, 0xb366, 0xf184, 0x9f83, 0x4229, 0xeef8, 0x321f, 0x1308,
    0xd90a, 0x69e6, 0x783b, 0xc2e1, 0xc3ad, 0x0f54, 0x5dc4, 0xc05a,
    0x7776, 0x8d48, 0xa891, 0x88da, 0x02e8, 0xa384, 0x3793, 0xb387
};

const uint16_t DataSet_1_16u::outputs::SUBV[64] = {
    0x8de1, 0x6e92, 0x53cd, 0x451b, 0x1a13, 0xbc9d, 0xad13, 0x1f14,
    0x21dd, 0x533f, 0x2623, 0x73f9, 0x1bfd, 0xcacd, 0xeca5, 0xfb0d,
    0x7bd6, 0xe1cd, 0xe13f, 0xe018, 0xa874, 0xbe8b, 0x9f4e, 0xb2d1,
    0xa94e, 0x62e5, 0x5907, 0x08bb, 0xe113, 0xef6c, 0x8d29, 0x0230,
    0x9623, 0x8450, 0xd28b, 0x9358, 0x3060, 0xbb74, 0xb7ad, 0x7d2e,
    0x4586, 0x2934, 0xdfb8, 0xd3f0, 0x90bc, 0x62b2, 0x33ae, 0x706e,
    0x6531, 0xcc9a, 0x58f7, 0x531c, 0xc437, 0x80df, 0xd05d, 0x03b8,
    0x2c38, 0xbd75, 0xe376, 0x3ece, 0x8273, 0x5b7c, 0x5149, 0x9988
};

const uint16_t DataSet_1_16u::outputs::MSUBV[64] = {
    0xf210, 0x6e92, 0x53cd, 0x76ed, 0x1a13, 0x5a96, 0x4517, 0x1f14,
    0x21dd, 0xd03a, 0x1b41, 0x73f9, 0x4d42, 0xcacd, 0xeca5, 0x4317,
    0x25b7, 0xe1cd, 0xe13f, 0x4891, 0xa874, 0xadc9, 0xeea4, 0xb2d1,
    0xa94e, 0xa384, 0xdd59, 0x08bb, 0xdee2, 0xef6c, 0x8d29, 0xf4a4,
    0x1721, 0x8450, 0xd28b, 0x0249, 0x3060, 0x5d5d, 0x7b38, 0x7d2e,
    0x4586, 0xb366, 0xf184, 0xd3f0, 0x4229, 0x62b2, 0x33ae, 0x1308,
    0xd90a, 0xcc9a, 0x58f7, 0xc2e1, 0xc437, 0x0f54, 0x5dc4, 0x03b8,
    0x2c38, 0x8d48, 0xa891, 0x3ece, 0x02e8, 0x5b7c, 0x5149, 0xb387
};

const uint16_t DataSet_1_16u::outputs::SUBS[64] = {
    0x81d2, 0xf979, 0x09ef, 0x06af, 0x350c, 0xea58, 0xd4d9, 0x6df0,
    0x35ed, 0x5ffc, 0xab03, 0x908b, 0xdd04, 0xa810, 0x9dd3, 0xd2d9,
    0xb579, 0xb7de, 0x2c1b, 0xd853, 0xa75f, 0x3d8b, 0x7e66, 0xd3cb,
    0x0e71, 0x3346, 0x6d1b, 0x78c4, 0x6ea4, 0x4556, 0x22be, 0x8466,
    0xa6e3, 0x5c1c, 0x945e, 0x920b, 0x1d87, 0xed1f, 0x0afa, 0x63f5,
    0xa926, 0x4328, 0x8146, 0x2f44, 0xd1eb, 0x7eb9, 0xc1e0, 0xa2ca,
    0x68cc, 0xf9a7, 0x07fc, 0x52a3, 0x536e, 0x9f16, 0xed86, 0x501b,
    0x0737, 0x1d0a, 0x3853, 0x189b, 0x92aa, 0x3345, 0xc754, 0x4349
};

const uint16_t DataSet_1_16u::outputs::MSUBS[64] = {
    0xf210, 0xf979, 0x09ef, 0x76ed, 0x350c, 0x5a96, 0x4517, 0x6df0,
    0x35ed, 0xd03a, 0x1b41, 0x908b, 0x4d42, 0xa810, 0x9dd3, 0x4317,
    0x25b7, 0xb7de, 0x2c1b, 0x4891, 0xa75f, 0xadc9, 0xeea4, 0xd3cb,
    0x0e71, 0xa384, 0xdd59, 0x78c4, 0xdee2, 0x4556, 0x22be, 0xf4a4,
    0x1721, 0x5c1c, 0x945e, 0x0249, 0x1d87, 0x5d5d, 0x7b38, 0x63f5,
    0xa926, 0xb366, 0xf184, 0x2f44, 0x4229, 0x7eb9, 0xc1e0, 0x1308,
    0xd90a, 0xf9a7, 0x07fc, 0xc2e1, 0x536e, 0x0f54, 0x5dc4, 0x501b,
    0x0737, 0x8d48, 0xa891, 0x189b, 0x02e8, 0x3345, 0xc754, 0xb387
};

const uint16_t DataSet_1_16u::outputs::SUBFROMV[64] = {
    0x721f, 0x916e, 0xac33, 0xbae5, 0xe5ed, 0x4363, 0x52ed, 0xe0ec,
    0xde23, 0xacc1, 0xd9dd, 0x8c07, 0xe403, 0x3533, 0x135b, 0x04f3,
    0x842a, 0x1e33, 0x1ec1, 0x1fe8, 0x578c, 0x4175, 0x60b2, 0x4d2f,
    0x56b2, 0x9d1b, 0xa6f9, 0xf745, 0x1eed, 0x1094, 0x72d7, 0xfdd0,
    0x69dd, 0x7bb0, 0x2d75, 0x6ca8, 0xcfa0, 0x448c, 0x4853, 0x82d2,
    0xba7a, 0xd6cc, 0x2048, 0x2c10, 0x6f44, 0x9d4e, 0xcc52, 0x8f92,
    0x9acf, 0x3366, 0xa709, 0xace4, 0x3bc9, 0x7f21, 0x2fa3, 0xfc48,
    0xd3c8, 0x428b, 0x1c8a, 0xc132, 0x7d8d, 0xa484, 0xaeb7, 0x6678
};

const uint16_t DataSet_1_16u::outputs::MSUBFROMV[64] = {
    0x642f, 0x916e, 0xac33, 0x31d2, 0xe5ed, 0x9df9, 0x9804, 0xe0ec,
    0xde23, 0x7cfb, 0xf51e, 0x8c07, 0x3145, 0x3533, 0x135b, 0x480a,
    0xa9e1, 0x1e33, 0x1ec1, 0x6879, 0x578c, 0xef3e, 0x4f56, 0x4d2f,
    0x56b2, 0x409f, 0x8452, 0xf745, 0xfdcf, 0x1094, 0x72d7, 0xf274,
    0x80fe, 0x7bb0, 0x2d75, 0x6ef1, 0xcfa0, 0xa1e9, 0xc38b, 0x82d2,
    0xba7a, 0x8a32, 0x11cc, 0x2c10, 0xb16d, 0x9d4e, 0xcc52, 0xa29a,
    0x73d9, 0x3366, 0xa709, 0x6fc5, 0x3bc9, 0x8e75, 0x8d67, 0xfc48,
    0xd3c8, 0xcfd3, 0xc51b, 0xc132, 0x8075, 0xa484, 0xaeb7, 0x19ff
};

const uint16_t DataSet_1_16u::outputs::SUBFROMS[64] = {
    0x7e2e, 0x0687, 0xf611, 0xf951, 0xcaf4, 0x15a8, 0x2b27, 0x9210,
    0xca13, 0xa004, 0x54fd, 0x6f75, 0x22fc, 0x57f0, 0x622d, 0x2d27,
    0x4a87, 0x4822, 0xd3e5, 0x27ad, 0x58a1, 0xc275, 0x819a, 0x2c35,
    0xf18f, 0xccba, 0x92e5, 0x873c, 0x915c, 0xbaaa, 0xdd42, 0x7b9a,
    0x591d, 0xa3e4, 0x6ba2, 0x6df5, 0xe279, 0x12e1, 0xf506, 0x9c0b,
    0x56da, 0xbcd8, 0x7eba, 0xd0bc, 0x2e15, 0x8147, 0x3e20, 0x5d36,
    0x9734, 0x0659, 0xf804, 0xad5d, 0xac92, 0x60ea, 0x127a, 0xafe5,
    0xf8c9, 0xe2f6, 0xc7ad, 0xe765, 0x6d56, 0xccbb, 0x38ac, 0xbcb7
};

const uint16_t DataSet_1_16u::outputs::MSUBFROMS[64] = {
    0x703e, 0x0687, 0xf611, 0x703e, 0xcaf4, 0x703e, 0x703e, 0x9210,
    0xca13, 0x703e, 0x703e, 0x6f75, 0x703e, 0x57f0, 0x622d, 0x703e,
    0x703e, 0x4822, 0xd3e5, 0x703e, 0x58a1, 0x703e, 0x703e, 0x2c35,
    0xf18f, 0x703e, 0x703e, 0x873c, 0x703e, 0xbaaa, 0xdd42, 0x703e,
    0x703e, 0xa3e4, 0x6ba2, 0x703e, 0xe279, 0x703e, 0x703e, 0x9c0b,
    0x56da, 0x703e, 0x703e, 0xd0bc, 0x703e, 0x8147, 0x3e20, 0x703e,
    0x703e, 0x0659, 0xf804, 0x703e, 0xac92, 0x703e, 0x703e, 0xafe5,
    0xf8c9, 0x703e, 0x703e, 0xe765, 0x703e, 0xccbb, 0x38ac, 0x703e
};

const uint16_t DataSet_1_16u::outputs::POSTPREFDEC[64] = {
    0xf20f, 0x69b6, 0x7a2c, 0x76ec, 0xa549, 0x5a95, 0x4516, 0xde2d,
    0xa62a, 0xd039, 0x1b40, 0x00c8, 0x4d41, 0x184d, 0x0e10, 0x4316,
    0x25b6, 0x281b, 0x9c58, 0x4890, 0x179c, 0xadc8, 0xeea3, 0x4408,
    0x7eae, 0xa383, 0xdd58, 0xe901, 0xdee1, 0xb593, 0x92fb, 0xf4a3,
    0x1720, 0xcc59, 0x049b, 0x0248, 0x8dc4, 0x5d5c, 0x7b37, 0xd432,
    0x1963, 0xb365, 0xf183, 0x9f81, 0x4228, 0xeef6, 0x321d, 0x1307,
    0xd909, 0x69e4, 0x7839, 0xc2e0, 0xc3ab, 0x0f53, 0x5dc3, 0xc058,
    0x7774, 0x8d47, 0xa890, 0x88d8, 0x02e7, 0xa382, 0x3791, 0xb386
};

const uint16_t DataSet_1_16u::outputs::MPOSTPREFDEC[64] = {
    0xf210, 0x69b6, 0x7a2c, 0x76ed, 0xa549, 0x5a96, 0x4517, 0xde2d,
    0xa62a, 0xd03a, 0x1b41, 0x00c8, 0x4d42, 0x184d, 0x0e10, 0x4317,
    0x25b7, 0x281b, 0x9c58, 0x4891, 0x179c, 0xadc9, 0xeea4, 0x4408,
    0x7eae, 0xa384, 0xdd59, 0xe901, 0xdee2, 0xb593, 0x92fb, 0xf4a4,
    0x1721, 0xcc59, 0x049b, 0x0249, 0x8dc4, 0x5d5d, 0x7b38, 0xd432,
    0x1963, 0xb366, 0xf184, 0x9f81, 0x4229, 0xeef6, 0x321d, 0x1308,
    0xd90a, 0x69e4, 0x7839, 0xc2e1, 0xc3ab, 0x0f54, 0x5dc4, 0xc058,
    0x7774, 0x8d48, 0xa891, 0x88d8, 0x02e8, 0xa382, 0x3791, 0xb387
};

const uint16_t DataSet_1_16u::outputs::MULV[64] = {
    0xb0f0, 0xb473, 0x7ee0, 0xeb6a, 0xb0e6, 0x19e6, 0xbc5c, 0xe2ac,
    0xcd1a, 0x40de, 0x669e, 0x8f50, 0x74ca, 0xb54e, 0x202c, 0x16e6,
    0xf4d7, 0x08a4, 0xe40a, 0x3489, 0xdb25, 0xbdae, 0xc718, 0xfaf8,
    0x9b4f, 0x8efc, 0xca82, 0x5f8e, 0x92be, 0xd720, 0x11b4, 0xe250,
    0x72be, 0x4b84, 0xc65c, 0x84b9, 0x7fb9, 0x76a5, 0x8f68, 0x79ff,
    0x70b8, 0x05ec, 0x3930, 0x0e24, 0x8475, 0x7c93, 0xb120, 0x82d0,
    0x777a, 0x7717, 0x7d2e, 0x8625, 0xc19c, 0x9964, 0xaddc, 0x53f9,
    0xbde1, 0xaa58, 0x5c4b, 0x9b53, 0x5408, 0x5095, 0x04a2, 0x0279
};

const uint16_t DataSet_1_16u::outputs::MMULV[64] = {
    0xf210, 0xb473, 0x7ee0, 0x76ed, 0xb0e6, 0x5a96, 0x4517, 0xe2ac,
    0xcd1a, 0xd03a, 0x1b41, 0x8f50, 0x4d42, 0xb54e, 0x202c, 0x4317,
    0x25b7, 0x08a4, 0xe40a, 0x4891, 0xdb25, 0xadc9, 0xeea4, 0xfaf8,
    0x9b4f, 0xa384, 0xdd59, 0x5f8e, 0xdee2, 0xd720, 0x11b4, 0xf4a4,
    0x1721, 0x4b84, 0xc65c, 0x0249, 0x7fb9, 0x5d5d, 0x7b38, 0x79ff,
    0x70b8, 0xb366, 0xf184, 0x0e24, 0x4229, 0x7c93, 0xb120, 0x1308,
    0xd90a, 0x7717, 0x7d2e, 0xc2e1, 0xc19c, 0x0f54, 0x5dc4, 0x53f9,
    0xbde1, 0x8d48, 0xa891, 0x9b53, 0x02e8, 0x5095, 0x04a2, 0xb387
};

const uint16_t DataSet_1_16u::outputs::MULS[64] = {
    0x9fe0, 0xaa52, 0x46e6, 0x7d66, 0x67ec, 0x9054, 0xcb92, 0xef24,
    0x0e6a, 0xce0c, 0x09be, 0x20ae, 0x95fc, 0x02e4, 0xd81e, 0x4f92,
    0x3252, 0xf6c8, 0xcd8e, 0x031e, 0x6806, 0x06ae, 0x8bb8, 0x6a2e,
    0x3e62, 0x59f8, 0x8b8e, 0x4e7c, 0xdabc, 0xb9d8, 0xd908, 0xffb8,
    0x09fe, 0xddcc, 0x5dc8, 0x7dae, 0x85b6, 0x4c86, 0x5790, 0xb45a,
    0xe638, 0x12b4, 0x3df8, 0x817c, 0xf5ee, 0xefd2, 0x4344, 0x1bf0,
    0xf06c, 0xd576, 0x7e0c, 0xa27e, 0xa3a8, 0x7658, 0x7578, 0x858e,
    0x1e56, 0xb770, 0x431e, 0x148e, 0x3430, 0xe9ba, 0x555c, 0x8ab2
};

const uint16_t DataSet_1_16u::outputs::MMULS[64] = {
    0xf210, 0xaa52, 0x46e6, 0x76ed, 0x67ec, 0x5a96, 0x4517, 0xef24,
    0x0e6a, 0xd03a, 0x1b41, 0x20ae, 0x4d42, 0x02e4, 0xd81e, 0x4317,
    0x25b7, 0xf6c8, 0xcd8e, 0x4891, 0x6806, 0xadc9, 0xeea4, 0x6a2e,
    0x3e62, 0xa384, 0xdd59, 0x4e7c, 0xdee2, 0xb9d8, 0xd908, 0xf4a4,
    0x1721, 0xddcc, 0x5dc8, 0x0249, 0x85b6, 0x5d5d, 0x7b38, 0xb45a,
    0xe638, 0xb366, 0xf184, 0x817c, 0x4229, 0xefd2, 0x4344, 0x1308,
    0xd90a, 0xd576, 0x7e0c, 0xc2e1, 0xa3a8, 0x0f54, 0x5dc4, 0x858e,
    0x1e56, 0x8d48, 0xa891, 0x148e, 0x02e8, 0xe9ba, 0x555c, 0xb387
};

const uint16_t DataSet_1_16u::outputs::DIVV[64] = {
    0x0002, 0x0000, 0x0003, 0x0002, 0x0001, 0x0000, 0x0000, 0x0001,
    0x0001, 0x0001, 0x0000, 0x0000, 0x0001, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0003, 0x0000,
    0x0000, 0x0002, 0x0001, 0x0001, 0x0000, 0x0000, 0x0019, 0x0001,
    0x0000, 0x0002, 0x0000, 0x0000, 0x0001, 0x0000, 0x0000, 0x0002,
    0x0000, 0x0001, 0x000d, 0x0000, 0x0000, 0x0001, 0x0000, 0x0000,
    0x0001, 0x0000, 0x0003, 0x0001, 0x0000, 0x0000, 0x0000, 0x0001,
    0x0001, 0x0000, 0x0000, 0x0001, 0x0000, 0x0002, 0x0000, 0x0006
};

const uint16_t DataSet_1_16u::outputs::MDIVV[64] = {
    0xf210, 0x0000, 0x0003, 0x76ed, 0x0001, 0x5a96, 0x4517, 0x0001,
    0x0001, 0xd03a, 0x1b41, 0x0000, 0x4d42, 0x0000, 0x0000, 0x4317,
    0x25b7, 0x0000, 0x0000, 0x4891, 0x0000, 0xadc9, 0xeea4, 0x0000,
    0x0000, 0xa384, 0xdd59, 0x0001, 0xdee2, 0x0000, 0x0019, 0xf4a4,
    0x1721, 0x0002, 0x0000, 0x0249, 0x0001, 0x5d5d, 0x7b38, 0x0002,
    0x0000, 0xb366, 0xf184, 0x0000, 0x4229, 0x0001, 0x0000, 0x1308,
    0xd90a, 0x0000, 0x0003, 0xc2e1, 0x0000, 0x0f54, 0x5dc4, 0x0001,
    0x0001, 0x8d48, 0xa891, 0x0001, 0x02e8, 0x0002, 0x0000, 0xb387
};

const uint16_t DataSet_1_16u::outputs::DIVS[64] = {
    0x0002, 0x0000, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0001,
    0x0001, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0001, 0x0000, 0x0000, 0x0001, 0x0002, 0x0000,
    0x0001, 0x0001, 0x0001, 0x0002, 0x0001, 0x0001, 0x0001, 0x0002,
    0x0000, 0x0001, 0x0000, 0x0000, 0x0001, 0x0000, 0x0001, 0x0001,
    0x0000, 0x0001, 0x0002, 0x0001, 0x0000, 0x0002, 0x0000, 0x0000,
    0x0001, 0x0000, 0x0001, 0x0001, 0x0001, 0x0000, 0x0000, 0x0001,
    0x0001, 0x0001, 0x0001, 0x0001, 0x0000, 0x0001, 0x0000, 0x0001
};

const uint16_t DataSet_1_16u::outputs::MDIVS[64] = {
    0xf210, 0x0000, 0x0001, 0x76ed, 0x0001, 0x5a96, 0x4517, 0x0001,
    0x0001, 0xd03a, 0x1b41, 0x0000, 0x4d42, 0x0000, 0x0000, 0x4317,
    0x25b7, 0x0000, 0x0001, 0x4891, 0x0000, 0xadc9, 0xeea4, 0x0000,
    0x0001, 0xa384, 0xdd59, 0x0002, 0xdee2, 0x0001, 0x0001, 0xf4a4,
    0x1721, 0x0001, 0x0000, 0x0249, 0x0001, 0x5d5d, 0x7b38, 0x0001,
    0x0000, 0xb366, 0xf184, 0x0001, 0x4229, 0x0002, 0x0000, 0x1308,
    0xd90a, 0x0000, 0x0001, 0xc2e1, 0x0001, 0x0f54, 0x5dc4, 0x0001,
    0x0001, 0x8d48, 0xa891, 0x0001, 0x02e8, 0x0001, 0x0000, 0xb387
};

const uint16_t DataSet_1_16u::outputs::RCP[64] = {
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000
};

const uint16_t DataSet_1_16u::outputs::MRCP[64] = {
    0xf210, 0x0000, 0x0000, 0x76ed, 0x0000, 0x5a96, 0x4517, 0x0000,
    0x0000, 0xd03a, 0x1b41, 0x0000, 0x4d42, 0x0000, 0x0000, 0x4317,
    0x25b7, 0x0000, 0x0000, 0x4891, 0x0000, 0xadc9, 0xeea4, 0x0000,
    0x0000, 0xa384, 0xdd59, 0x0000, 0xdee2, 0x0000, 0x0000, 0xf4a4,
    0x1721, 0x0000, 0x0000, 0x0249, 0x0000, 0x5d5d, 0x7b38, 0x0000,
    0x0000, 0xb366, 0xf184, 0x0000, 0x4229, 0x0000, 0x0000, 0x1308,
    0xd90a, 0x0000, 0x0000, 0xc2e1, 0x0000, 0x0f54, 0x5dc4, 0x0000,
    0x0000, 0x8d48, 0xa891, 0x0000, 0x02e8, 0x0000, 0x0000, 0xb387
};

const uint16_t DataSet_1_16u::outputs::RCPS[64] = {
    0x0000, 0x0001, 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0000,
    0x0000, 0x0000, 0x0004, 0x008e, 0x0001, 0x0004, 0x0007, 0x0001,
    0x0002, 0x0002, 0x0000, 0x0001, 0x0004, 0x0000, 0x0000, 0x0001,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0004, 0x0000, 0x0018, 0x0031, 0x0000, 0x0001, 0x0000, 0x0000,
    0x0004, 0x0000, 0x0000, 0x0000, 0x0001, 0x0000, 0x0002, 0x0005,
    0x0000, 0x0001, 0x0000, 0x0000, 0x0000, 0x0007, 0x0001, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0026, 0x0000, 0x0002, 0x0000
};

const uint16_t DataSet_1_16u::outputs::MRCPS[64] = {
    0xf210, 0x0001, 0x0000, 0x76ed, 0x0000, 0x5a96, 0x4517, 0x0000,
    0x0000, 0xd03a, 0x1b41, 0x008e, 0x4d42, 0x0004, 0x0007, 0x4317,
    0x25b7, 0x0002, 0x0000, 0x4891, 0x0004, 0xadc9, 0xeea4, 0x0001,
    0x0000, 0xa384, 0xdd59, 0x0000, 0xdee2, 0x0000, 0x0000, 0xf4a4,
    0x1721, 0x0000, 0x0018, 0x0249, 0x0000, 0x5d5d, 0x7b38, 0x0000,
    0x0004, 0xb366, 0xf184, 0x0000, 0x4229, 0x0000, 0x0002, 0x1308,
    0xd90a, 0x0001, 0x0000, 0xc2e1, 0x0000, 0x0f54, 0x5dc4, 0x0000,
    0x0000, 0x8d48, 0xa891, 0x0000, 0x02e8, 0x0000, 0x0002, 0xb387
};

const bool DataSet_1_16u::outputs::CMPEQV[64] = {
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false
};

const bool DataSet_1_16u::outputs::CMPEQS[64] = {
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false
};

const bool DataSet_1_16u::outputs::CMPNEV[64] = {
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true
};

const bool DataSet_1_16u::outputs::CMPNES[64] = {
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true
};

const bool DataSet_1_16u::outputs::CMPGTV[64] = {
    true,   false,  true,   true,   true,   false,  false,  true,
    true,   true,   false,  false,  true,   false,  false,  false,
    false,  false,  false,  false,  false,  false,  true,   false,
    false,  true,   true,   true,   false,  false,  true,   true,
    false,  true,   false,  false,  true,   false,  false,  true,
    false,  true,   true,   false,  false,  true,   false,  false,
    true,   false,  true,   true,   false,  false,  false,  true,
    true,   false,  false,  true,   false,  true,   false,  true
};

const bool DataSet_1_16u::outputs::CMPGTS[64] = {
    true,   false,  true,   true,   true,   false,  false,  true,
    true,   true,   false,  false,  false,  false,  false,  false,
    false,  false,  true,   false,  false,  true,   true,   false,
    true,   true,   true,   true,   true,   true,   true,   true,
    false,  true,   false,  false,  true,   false,  true,   true,
    false,  true,   true,   true,   false,  true,   false,  false,
    true,   false,  true,   true,   true,   false,  false,  true,
    true,   true,   true,   true,   false,  true,   false,  true
};

const bool DataSet_1_16u::outputs::CMPLTV[64] = {
    false,  true,   false,  false,  false,  true,   true,   false,
    false,  false,  true,   true,   false,  true,   true,   true,
    true,   true,   true,   true,   true,   true,   false,  true,
    true,   false,  false,  false,  true,   true,   false,  false,
    true,   false,  true,   true,   false,  true,   true,   false,
    true,   false,  false,  true,   true,   false,  true,   true,
    false,  true,   false,  false,  true,   true,   true,   false,
    false,  true,   true,   false,  true,   false,  true,   false
};

const bool DataSet_1_16u::outputs::CMPLTS[64] = {
    false,  true,   false,  false,  false,  true,   true,   false,
    false,  false,  true,   true,   true,   true,   true,   true,
    true,   true,   false,  true,   true,   false,  false,  true,
    false,  false,  false,  false,  false,  false,  false,  false,
    true,   false,  true,   true,   false,  true,   false,  false,
    true,   false,  false,  false,  true,   false,  true,   true,
    false,  true,   false,  false,  false,  true,   true,   false,
    false,  false,  false,  false,  true,   false,  true,   false
};

const bool DataSet_1_16u::outputs::CMPGEV[64] = {
    true,   false,  true,   true,   true,   false,  false,  true,
    true,   true,   false,  false,  true,   false,  false,  false,
    false,  false,  false,  false,  false,  false,  true,   false,
    false,  true,   true,   true,   false,  false,  true,   true,
    false,  true,   false,  false,  true,   false,  false,  true,
    false,  true,   true,   false,  false,  true,   false,  false,
    true,   false,  true,   true,   false,  false,  false,  true,
    true,   false,  false,  true,   false,  true,   false,  true
};

const bool DataSet_1_16u::outputs::CMPGES[64] = {
    true,   false,  true,   true,   true,   false,  false,  true,
    true,   true,   false,  false,  false,  false,  false,  false,
    false,  false,  true,   false,  false,  true,   true,   false,
    true,   true,   true,   true,   true,   true,   true,   true,
    false,  true,   false,  false,  true,   false,  true,   true,
    false,  true,   true,   true,   false,  true,   false,  false,
    true,   false,  true,   true,   true,   false,  false,  true,
    true,   true,   true,   true,   false,  true,   false,  true
};

const bool DataSet_1_16u::outputs::CMPLEV[64] = {
    false,  true,   false,  false,  false,  true,   true,   false,
    false,  false,  true,   true,   false,  true,   true,   true,
    true,   true,   true,   true,   true,   true,   false,  true,
    true,   false,  false,  false,  true,   true,   false,  false,
    true,   false,  true,   true,   false,  true,   true,   false,
    true,   false,  false,  true,   true,   false,  true,   true,
    false,  true,   false,  false,  true,   true,   true,   false,
    false,  true,   true,   false,  true,   false,  true,   false
};

const bool DataSet_1_16u::outputs::CMPLES[64] = {
    false,  true,   false,  false,  false,  true,   true,   false,
    false,  false,  true,   true,   true,   true,   true,   true,
    true,   true,   false,  true,   true,   false,  false,  true,
    false,  false,  false,  false,  false,  false,  false,  false,
    true,   false,  true,   true,   false,  true,   false,  false,
    true,   false,  false,  false,  true,   false,  true,   true,
    false,  true,   false,  false,  false,  true,   true,   false,
    false,  false,  false,  false,  true,   false,  true,   false
};


const bool  DataSet_1_16u::outputs::CMPEV = false;
const bool  DataSet_1_16u::outputs::CMPES = false;

const uint16_t DataSet_1_16u::outputs::HADD[64] = {
    0xf210, 0x5bc7, 0xd5f4, 0x4ce1, 0xf22b, 0x4cc1, 0x91d8, 0x7006,
    0x1631, 0xe66b, 0x01ac, 0x0275, 0x4fb7, 0x6805, 0x7616, 0xb92d,
    0xdee4, 0x0700, 0xa359, 0xebea, 0x0387, 0xb150, 0x9ff4, 0xe3fd,
    0x62ac, 0x0630, 0xe389, 0xcc8b, 0xab6d, 0x6101, 0xf3fd, 0xe8a1,
    0xffc2, 0xcc1c, 0xd0b8, 0xd301, 0x60c6, 0xbe23, 0x395b, 0x0d8e,
    0x26f2, 0xda58, 0xcbdc, 0x6b5e, 0xad87, 0x9c7e, 0xce9c, 0xe1a4,
    0xbaae, 0x2493, 0x9ccd, 0x5fae, 0x235a, 0x32ae, 0x9072, 0x50cb,
    0xc840, 0x5588, 0xfe19, 0x86f2, 0x89da, 0x2d5d, 0x64ef, 0x1876
};

const uint16_t DataSet_1_16u::outputs::MHADD[64] = {
    0x0000, 0x69b7, 0xe3e4, 0xe3e4, 0x892e, 0x892e, 0x892e, 0x675c,
    0x0d87, 0x0d87, 0x0d87, 0x0e50, 0x0e50, 0x269e, 0x34af, 0x34af,
    0x34af, 0x5ccb, 0xf924, 0xf924, 0x10c1, 0x10c1, 0x10c1, 0x54ca,
    0xd379, 0xd379, 0xd379, 0xbc7b, 0xbc7b, 0x720f, 0x050b, 0x050b,
    0x050b, 0xd165, 0xd601, 0xd601, 0x63c6, 0x63c6, 0x63c6, 0x37f9,
    0x515d, 0x515d, 0x515d, 0xf0df, 0xf0df, 0xdfd6, 0x11f4, 0x11f4,
    0x11f4, 0x7bd9, 0xf413, 0xf413, 0xb7bf, 0xb7bf, 0xb7bf, 0x7818,
    0xef8d, 0xef8d, 0xef8d, 0x7866, 0x7866, 0x1be9, 0x537b, 0x537b
};

const uint16_t DataSet_1_16u::outputs::HMUL[64] = {
    0xf210, 0x9970, 0x58b0, 0x3af0, 0xb960, 0x5e40, 0xb7c0, 0x8480,
    0x4180, 0xd700, 0x9700, 0x8f00, 0xde00, 0xa400, 0xe400, 0x7c00,
    0xa400, 0xf000, 0x7000, 0x7000, 0xb000, 0x3000, 0xc000, 0xc000,
    0x4000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000
};

const uint16_t DataSet_1_16u::outputs::MHMUL[64] = {
    0x0001, 0x69b7, 0xcb2b, 0xcb2b, 0x716e, 0x716e, 0x716e, 0xc5c4,
    0x4fec, 0x4fec, 0x4fec, 0xc04c, 0xc04c, 0xb728, 0x59a8, 0x59a8,
    0x59a8, 0x0e60, 0x7f60, 0x7f60, 0xbde0, 0xbde0, 0xbde0, 0x2ce0,
    0xed20, 0xed20, 0xed20, 0xfa40, 0xfa40, 0xed00, 0x4c00, 0x4c00,
    0x4c00, 0xb800, 0x2000, 0x2000, 0xa000, 0xa000, 0xa000, 0xe000,
    0x8000, 0x8000, 0x8000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000
};

const uint16_t DataSet_1_16u::outputs::FMULADDV[64] = {
    0x44e6, 0x93b6, 0x087f, 0xad8e, 0x05f2, 0xe553, 0x79ca, 0xbd7c,
    0xa8bf, 0x627b, 0x52ef, 0xabfd, 0x9e12, 0xc19d, 0x7328, 0xae5f,
    0x1e2f, 0xf597, 0xfb92, 0xc961, 0xaac5, 0x2fe9, 0xa257, 0x60a3,
    0x4382, 0x792f, 0x13ee, 0x4d49, 0xad23, 0x223d, 0x8375, 0x6add,
    0x53a7, 0x7770, 0x0a09, 0x5b67, 0xa271, 0x03e4, 0x9c87, 0x8521,
    0x24ea, 0x3b41, 0x1715, 0xf407, 0x6039, 0x51f4, 0xf559, 0x27ab,
    0x8a76, 0x3b55, 0x7ad3, 0x17f8, 0xbd97, 0xc5dc, 0xc967, 0x2b14,
    0xf9be, 0x7916, 0xa6cc, 0x108d, 0x2387, 0x1a8e, 0x6177, 0x3017
};

const uint16_t DataSet_1_16u::outputs::MFMULADDV[64] = {
    0xf210, 0x93b6, 0x087f, 0x76ed, 0x05f2, 0x5a96, 0x4517, 0xbd7c,
    0xa8bf, 0xd03a, 0x1b41, 0xabfd, 0x4d42, 0xc19d, 0x7328, 0x4317,
    0x25b7, 0xf597, 0xfb92, 0x4891, 0xaac5, 0xadc9, 0xeea4, 0x60a3,
    0x4382, 0xa384, 0xdd59, 0x4d49, 0xdee2, 0x223d, 0x8375, 0xf4a4,
    0x1721, 0x7770, 0x0a09, 0x0249, 0xa271, 0x5d5d, 0x7b38, 0x8521,
    0x24ea, 0xb366, 0xf184, 0xf407, 0x4229, 0x51f4, 0xf559, 0x1308,
    0xd90a, 0x3b55, 0x7ad3, 0xc2e1, 0xbd97, 0x0f54, 0x5dc4, 0x2b14,
    0xf9be, 0x8d48, 0xa891, 0x108d, 0x02e8, 0x1a8e, 0x6177, 0xb387
};

const uint16_t DataSet_1_16u::outputs::FMULSUBV[64] = {
    0x1cfa, 0xd530, 0xf541, 0x2946, 0x5bda, 0x4e79, 0xfeee, 0x07dc,
    0xf175, 0x1f41, 0x7a4d, 0x72a3, 0x4b82, 0xa8ff, 0xcd30, 0x7f6d,
    0xcb7f, 0x1bb1, 0xcc82, 0x9fb1, 0x0b85, 0x4b73, 0xebd9, 0x954d,
    0xf31c, 0xa4c9, 0x8116, 0x71d3, 0x7859, 0x8c03, 0x9ff3, 0x59c3,
    0x91d5, 0x1f98, 0x82af, 0xae0b, 0x5d01, 0xe966, 0x8249, 0x6edd,
    0xbc86, 0xd097, 0x5b4b, 0x2841, 0xa8b1, 0xa732, 0x6ce7, 0xddf5,
    0x647e, 0xb2d9, 0x7f89, 0xf452, 0xc5a1, 0x6cec, 0x9251, 0x7cde,
    0x8204, 0xdb9a, 0x11ca, 0x2619, 0x8489, 0x869c, 0xa7cd, 0xd4db
};

const uint16_t DataSet_1_16u::outputs::MFMULSUBV[64] = {
    0xf210, 0xd530, 0xf541, 0x76ed, 0x5bda, 0x5a96, 0x4517, 0x07dc,
    0xf175, 0xd03a, 0x1b41, 0x72a3, 0x4d42, 0xa8ff, 0xcd30, 0x4317,
    0x25b7, 0x1bb1, 0xcc82, 0x4891, 0x0b85, 0xadc9, 0xeea4, 0x954d,
    0xf31c, 0xa384, 0xdd59, 0x71d3, 0xdee2, 0x8c03, 0x9ff3, 0xf4a4,
    0x1721, 0x1f98, 0x82af, 0x0249, 0x5d01, 0x5d5d, 0x7b38, 0x6edd,
    0xbc86, 0xb366, 0xf184, 0x2841, 0x4229, 0xa732, 0x6ce7, 0x1308,
    0xd90a, 0xb2d9, 0x7f89, 0xc2e1, 0xc5a1, 0x0f54, 0x5dc4, 0x7cde,
    0x8204, 0x8d48, 0xa891, 0x2619, 0x02e8, 0x869c, 0xa7cd, 0xb387
};

const uint16_t DataSet_1_16u::outputs::FADDMULV[64] = {
    0x0d8a, 0x0994, 0x2c93, 0x78dc, 0x1b0c, 0x39e3, 0xf09a, 0x1a80,
    0xe2fd, 0x2e81, 0xc20f, 0x6c65, 0x34f8, 0x1ee1, 0xc90c, 0x3999,
    0xb440, 0x7391, 0xca18, 0x2870, 0x55c0, 0x4e9d, 0x1e86, 0x176b,
    0x3f30, 0x70f9, 0xf724, 0x9d53, 0x0bd5, 0x184c, 0x930f, 0x0838,
    0x9437, 0x9830, 0x39e9, 0x716c, 0x9a30, 0x603a, 0x809d, 0x2570,
    0xbee4, 0xeb78, 0x0690, 0xd6bc, 0xd0d8, 0x9dbc, 0x879e, 0x2996,
    0xa574, 0x7da0, 0x2c91, 0xc4d2, 0xac5b, 0x8238, 0x3959, 0x245e,
    0x19aa, 0xd40a, 0xfbac, 0xfba8, 0x5e23, 0x733a, 0x8b37, 0x66b4
};

const uint16_t DataSet_1_16u::outputs::MFADDMULV[64] = {
    0xf210, 0x0994, 0x2c93, 0x76ed, 0x1b0c, 0x5a96, 0x4517, 0x1a80,
    0xe2fd, 0xd03a, 0x1b41, 0x6c65, 0x4d42, 0x1ee1, 0xc90c, 0x4317,
    0x25b7, 0x7391, 0xca18, 0x4891, 0x55c0, 0xadc9, 0xeea4, 0x176b,
    0x3f30, 0xa384, 0xdd59, 0x9d53, 0xdee2, 0x184c, 0x930f, 0xf4a4,
    0x1721, 0x9830, 0x39e9, 0x0249, 0x9a30, 0x5d5d, 0x7b38, 0x2570,
    0xbee4, 0xb366, 0xf184, 0xd6bc, 0x4229, 0x9dbc, 0x879e, 0x1308,
    0xd90a, 0x7da0, 0x2c91, 0xc2e1, 0xac5b, 0x0f54, 0x5dc4, 0x245e,
    0x19aa, 0x8d48, 0xa891, 0xfba8, 0x02e8, 0x733a, 0x8b37, 0xb387
};

const uint16_t DataSet_1_16u::outputs::FSUBMULV[64] = {
    0x8936, 0x1e36, 0xc153, 0x2dcc, 0x87e4, 0xcdd9, 0x652a, 0x4840,
    0xe271, 0x2ca3, 0x5513, 0x9b45, 0x6428, 0x3143, 0xcc6c, 0x5425,
    0xd790, 0x5197, 0x5278, 0xf440, 0x1480, 0xd009, 0xee32, 0xe69b,
    0xea8a, 0x059f, 0x8df4, 0x7f99, 0xba7f, 0xc33c, 0x84e9, 0xb470,
    0x45db, 0x69c0, 0xa8ef, 0xb5d0, 0x8500, 0x058c, 0x06f3, 0x9a1c,
    0xcc2c, 0x7244, 0xf798, 0x9dd0, 0xa3f0, 0x7f72, 0xb9be, 0xa61a,
    0x0e3c, 0x754c, 0x7233, 0x5c14, 0x4eed, 0xbc88, 0xf17f, 0xec68,
    0x1458, 0xc2d6, 0xba76, 0x60ac, 0xb40d, 0x579c, 0xddbd, 0xa9f0
};

const uint16_t DataSet_1_16u::outputs::MFSUBMULV[64] = {
    0xf210, 0x1e36, 0xc153, 0x76ed, 0x87e4, 0x5a96, 0x4517, 0x4840,
    0xe271, 0xd03a, 0x1b41, 0x9b45, 0x4d42, 0x3143, 0xcc6c, 0x4317,
    0x25b7, 0x5197, 0x5278, 0x4891, 0x1480, 0xadc9, 0xeea4, 0xe69b,
    0xea8a, 0xa384, 0xdd59, 0x7f99, 0xdee2, 0xc33c, 0x84e9, 0xf4a4,
    0x1721, 0x69c0, 0xa8ef, 0x0249, 0x8500, 0x5d5d, 0x7b38, 0x9a1c,
    0xcc2c, 0xb366, 0xf184, 0x9dd0, 0x4229, 0x7f72, 0xb9be, 0x1308,
    0xd90a, 0x754c, 0x7233, 0xc2e1, 0x4eed, 0x0f54, 0x5dc4, 0xec68,
    0x1458, 0x8d48, 0xa891, 0x60ac, 0x02e8, 0x579c, 0xddbd, 0xb387
};

const uint16_t DataSet_1_16u::outputs::MAXV[64] = {
    0xf210, 0xfb25, 0x7a2d, 0x76ed, 0xa54a, 0x9df9, 0x9804, 0xde2e,
    0xa62b, 0xd03a, 0xf51e, 0x8cd0, 0x4d42, 0x4d81, 0x216c, 0x480a,
    0xa9e1, 0x464f, 0xbb1a, 0x6879, 0x6f29, 0xef3e, 0xeea4, 0x9138,
    0xd561, 0xa384, 0xdd59, 0xe902, 0xfdcf, 0xc628, 0x92fc, 0xf4a4,
    0x80fe, 0xcc5a, 0x3211, 0x6ef1, 0x8dc5, 0xa1e9, 0xc38b, 0xd433,
    0xd3de, 0xb366, 0xf184, 0xcb92, 0xb16d, 0xeef7, 0xfe70, 0xa29a,
    0xd90a, 0x9d4b, 0x783a, 0xc2e1, 0xff75, 0x8e75, 0x8d67, 0xc059,
    0x7775, 0xcfd3, 0xc51b, 0x88d9, 0x8075, 0xa383, 0xe649, 0xb387
};

const uint16_t DataSet_1_16u::outputs::MMAXV[64] = {
    0xf210, 0xfb25, 0x7a2d, 0x76ed, 0xa54a, 0x5a96, 0x4517, 0xde2e,
    0xa62b, 0xd03a, 0x1b41, 0x8cd0, 0x4d42, 0x4d81, 0x216c, 0x4317,
    0x25b7, 0x464f, 0xbb1a, 0x4891, 0x6f29, 0xadc9, 0xeea4, 0x9138,
    0xd561, 0xa384, 0xdd59, 0xe902, 0xdee2, 0xc628, 0x92fc, 0xf4a4,
    0x1721, 0xcc5a, 0x3211, 0x0249, 0x8dc5, 0x5d5d, 0x7b38, 0xd433,
    0xd3de, 0xb366, 0xf184, 0xcb92, 0x4229, 0xeef7, 0xfe70, 0x1308,
    0xd90a, 0x9d4b, 0x783a, 0xc2e1, 0xff75, 0x0f54, 0x5dc4, 0xc059,
    0x7775, 0x8d48, 0xa891, 0x88d9, 0x02e8, 0xa383, 0xe649, 0xb387
};

const uint16_t DataSet_1_16u::outputs::MAXS[64] = {
    0xf210, 0x703e, 0x7a2d, 0x76ed, 0xa54a, 0x703e, 0x703e, 0xde2e,
    0xa62b, 0xd03a, 0x703e, 0x703e, 0x703e, 0x703e, 0x703e, 0x703e,
    0x703e, 0x703e, 0x9c59, 0x703e, 0x703e, 0xadc9, 0xeea4, 0x703e,
    0x7eaf, 0xa384, 0xdd59, 0xe902, 0xdee2, 0xb594, 0x92fc, 0xf4a4,
    0x703e, 0xcc5a, 0x703e, 0x703e, 0x8dc5, 0x703e, 0x7b38, 0xd433,
    0x703e, 0xb366, 0xf184, 0x9f82, 0x703e, 0xeef7, 0x703e, 0x703e,
    0xd90a, 0x703e, 0x783a, 0xc2e1, 0xc3ac, 0x703e, 0x703e, 0xc059,
    0x7775, 0x8d48, 0xa891, 0x88d9, 0x703e, 0xa383, 0x703e, 0xb387
};

const uint16_t DataSet_1_16u::outputs::MMAXS[64] = {
    0xf210, 0x703e, 0x7a2d, 0x76ed, 0xa54a, 0x5a96, 0x4517, 0xde2e,
    0xa62b, 0xd03a, 0x1b41, 0x703e, 0x4d42, 0x703e, 0x703e, 0x4317,
    0x25b7, 0x703e, 0x9c59, 0x4891, 0x703e, 0xadc9, 0xeea4, 0x703e,
    0x7eaf, 0xa384, 0xdd59, 0xe902, 0xdee2, 0xb594, 0x92fc, 0xf4a4,
    0x1721, 0xcc5a, 0x703e, 0x0249, 0x8dc5, 0x5d5d, 0x7b38, 0xd433,
    0x703e, 0xb366, 0xf184, 0x9f82, 0x4229, 0xeef7, 0x703e, 0x1308,
    0xd90a, 0x703e, 0x783a, 0xc2e1, 0xc3ac, 0x0f54, 0x5dc4, 0xc059,
    0x7775, 0x8d48, 0xa891, 0x88d9, 0x02e8, 0xa383, 0x703e, 0xb387
};

const uint16_t DataSet_1_16u::outputs::MINV[64] = {
    0x642f, 0x69b7, 0x2660, 0x31d2, 0x8b37, 0x5a96, 0x4517, 0xbf1a,
    0x844e, 0x7cfb, 0x1b41, 0x00c9, 0x3145, 0x184e, 0x0e11, 0x4317,
    0x25b7, 0x281c, 0x9c59, 0x4891, 0x179d, 0xadc9, 0x4f56, 0x4409,
    0x7eaf, 0x409f, 0x8452, 0xe047, 0xdee2, 0xb594, 0x05d3, 0xf274,
    0x1721, 0x480a, 0x049c, 0x0249, 0x5d65, 0x5d5d, 0x7b38, 0x5705,
    0x1964, 0x8a32, 0x11cc, 0x9f82, 0x4229, 0x8c45, 0x321e, 0x1308,
    0x73d9, 0x69e5, 0x1f43, 0x6fc5, 0xc3ac, 0x0f54, 0x5dc4, 0xbca1,
    0x4b3d, 0x8d48, 0xa891, 0x4a0b, 0x02e8, 0x4807, 0x3792, 0x19ff
};

const uint16_t DataSet_1_16u::outputs::MMINV[64] = {
    0xf210, 0x69b7, 0x2660, 0x76ed, 0x8b37, 0x5a96, 0x4517, 0xbf1a,
    0x844e, 0xd03a, 0x1b41, 0x00c9, 0x4d42, 0x184e, 0x0e11, 0x4317,
    0x25b7, 0x281c, 0x9c59, 0x4891, 0x179d, 0xadc9, 0xeea4, 0x4409,
    0x7eaf, 0xa384, 0xdd59, 0xe047, 0xdee2, 0xb594, 0x05d3, 0xf4a4,
    0x1721, 0x480a, 0x049c, 0x0249, 0x5d65, 0x5d5d, 0x7b38, 0x5705,
    0x1964, 0xb366, 0xf184, 0x9f82, 0x4229, 0x8c45, 0x321e, 0x1308,
    0xd90a, 0x69e5, 0x1f43, 0xc2e1, 0xc3ac, 0x0f54, 0x5dc4, 0xbca1,
    0x4b3d, 0x8d48, 0xa891, 0x4a0b, 0x02e8, 0x4807, 0x3792, 0xb387
};

const uint16_t DataSet_1_16u::outputs::MINS[64] = {
    0x703e, 0x69b7, 0x703e, 0x703e, 0x703e, 0x5a96, 0x4517, 0x703e,
    0x703e, 0x703e, 0x1b41, 0x00c9, 0x4d42, 0x184e, 0x0e11, 0x4317,
    0x25b7, 0x281c, 0x703e, 0x4891, 0x179d, 0x703e, 0x703e, 0x4409,
    0x703e, 0x703e, 0x703e, 0x703e, 0x703e, 0x703e, 0x703e, 0x703e,
    0x1721, 0x703e, 0x049c, 0x0249, 0x703e, 0x5d5d, 0x703e, 0x703e,
    0x1964, 0x703e, 0x703e, 0x703e, 0x4229, 0x703e, 0x321e, 0x1308,
    0x703e, 0x69e5, 0x703e, 0x703e, 0x703e, 0x0f54, 0x5dc4, 0x703e,
    0x703e, 0x703e, 0x703e, 0x703e, 0x02e8, 0x703e, 0x3792, 0x703e
};

const uint16_t DataSet_1_16u::outputs::MMINS[64] = {
    0xf210, 0x69b7, 0x703e, 0x76ed, 0x703e, 0x5a96, 0x4517, 0x703e,
    0x703e, 0xd03a, 0x1b41, 0x00c9, 0x4d42, 0x184e, 0x0e11, 0x4317,
    0x25b7, 0x281c, 0x703e, 0x4891, 0x179d, 0xadc9, 0xeea4, 0x4409,
    0x703e, 0xa384, 0xdd59, 0x703e, 0xdee2, 0x703e, 0x703e, 0xf4a4,
    0x1721, 0x703e, 0x049c, 0x0249, 0x703e, 0x5d5d, 0x7b38, 0x703e,
    0x1964, 0xb366, 0xf184, 0x703e, 0x4229, 0x703e, 0x321e, 0x1308,
    0xd90a, 0x69e5, 0x703e, 0xc2e1, 0x703e, 0x0f54, 0x5dc4, 0x703e,
    0x703e, 0x8d48, 0xa891, 0x703e, 0x02e8, 0x703e, 0x3792, 0xb387
};

const uint16_t DataSet_1_16u::outputs::HMAX[64] = {
    0xf210, 0xf210, 0xf210, 0xf210, 0xf210, 0xf210, 0xf210, 0xf210,
    0xf210, 0xf210, 0xf210, 0xf210, 0xf210, 0xf210, 0xf210, 0xf210,
    0xf210, 0xf210, 0xf210, 0xf210, 0xf210, 0xf210, 0xf210, 0xf210,
    0xf210, 0xf210, 0xf210, 0xf210, 0xf210, 0xf210, 0xf210, 0xf4a4,
    0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4,
    0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4,
    0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4,
    0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4, 0xf4a4
};

const uint16_t DataSet_1_16u::outputs::MHMAX[64] = {
    0x0000, 0x69b7, 0x7a2d, 0x7a2d, 0xa54a, 0xa54a, 0xa54a, 0xde2e,
    0xde2e, 0xde2e, 0xde2e, 0xde2e, 0xde2e, 0xde2e, 0xde2e, 0xde2e,
    0xde2e, 0xde2e, 0xde2e, 0xde2e, 0xde2e, 0xde2e, 0xde2e, 0xde2e,
    0xde2e, 0xde2e, 0xde2e, 0xe902, 0xe902, 0xe902, 0xe902, 0xe902,
    0xe902, 0xe902, 0xe902, 0xe902, 0xe902, 0xe902, 0xe902, 0xe902,
    0xe902, 0xe902, 0xe902, 0xe902, 0xe902, 0xeef7, 0xeef7, 0xeef7,
    0xeef7, 0xeef7, 0xeef7, 0xeef7, 0xeef7, 0xeef7, 0xeef7, 0xeef7,
    0xeef7, 0xeef7, 0xeef7, 0xeef7, 0xeef7, 0xeef7, 0xeef7, 0xeef7
};

const uint16_t DataSet_1_16u::outputs::HMIN[64] = {
    0xf210, 0x69b7, 0x69b7, 0x69b7, 0x69b7, 0x5a96, 0x4517, 0x4517,
    0x4517, 0x4517, 0x1b41, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9,
    0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9,
    0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9,
    0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9,
    0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9,
    0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9,
    0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9
};

const uint16_t DataSet_1_16u::outputs::MHMIN[64] = {
    0xffff, 0x69b7, 0x69b7, 0x69b7, 0x69b7, 0x69b7, 0x69b7, 0x69b7,
    0x69b7, 0x69b7, 0x69b7, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9,
    0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9,
    0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9,
    0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9,
    0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9,
    0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9,
    0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9, 0x00c9
};

const uint16_t DataSet_1_16u::outputs::BANDV[64] = {
    0x6000, 0x6925, 0x2220, 0x30c0, 0x8102, 0x1890, 0x0004, 0x9e0a,
    0x840a, 0x503a, 0x1100, 0x00c0, 0x0140, 0x0800, 0x0000, 0x4002,
    0x21a1, 0x000c, 0x9818, 0x4811, 0x0709, 0xad08, 0x4e04, 0x0008,
    0x5421, 0x0084, 0x8450, 0xe002, 0xdcc2, 0x8400, 0x00d0, 0xf024,
    0x0020, 0x480a, 0x0010, 0x0241, 0x0d45, 0x0149, 0x4308, 0x5401,
    0x1144, 0x8222, 0x1184, 0x8b82, 0x0029, 0x8c45, 0x3210, 0x0208,
    0x5108, 0x0941, 0x1802, 0x42c1, 0xc324, 0x0e54, 0x0d44, 0x8001,
    0x4335, 0x8d40, 0x8011, 0x0809, 0x0060, 0x0003, 0x2600, 0x1187
};

const uint16_t DataSet_1_16u::outputs::MBANDV[64] = {
    0xf210, 0x6925, 0x2220, 0x76ed, 0x8102, 0x5a96, 0x4517, 0x9e0a,
    0x840a, 0xd03a, 0x1b41, 0x00c0, 0x4d42, 0x0800, 0x0000, 0x4317,
    0x25b7, 0x000c, 0x9818, 0x4891, 0x0709, 0xadc9, 0xeea4, 0x0008,
    0x5421, 0xa384, 0xdd59, 0xe002, 0xdee2, 0x8400, 0x00d0, 0xf4a4,
    0x1721, 0x480a, 0x0010, 0x0249, 0x0d45, 0x5d5d, 0x7b38, 0x5401,
    0x1144, 0xb366, 0xf184, 0x8b82, 0x4229, 0x8c45, 0x3210, 0x1308,
    0xd90a, 0x0941, 0x1802, 0xc2e1, 0xc324, 0x0f54, 0x5dc4, 0x8001,
    0x4335, 0x8d48, 0xa891, 0x0809, 0x02e8, 0x0003, 0x2600, 0xb387
};

const uint16_t DataSet_1_16u::outputs::BANDS[64] = {
    0x7010, 0x6036, 0x702c, 0x702c, 0x200a, 0x5016, 0x4016, 0x502e,
    0x202a, 0x503a, 0x1000, 0x0008, 0x4002, 0x100e, 0x0010, 0x4016,
    0x2036, 0x201c, 0x1018, 0x4010, 0x101c, 0x2008, 0x6024, 0x4008,
    0x702e, 0x2004, 0x5018, 0x6002, 0x5022, 0x3014, 0x103c, 0x7024,
    0x1020, 0x401a, 0x001c, 0x0008, 0x0004, 0x501c, 0x7038, 0x5032,
    0x1024, 0x3026, 0x7004, 0x1002, 0x4028, 0x6036, 0x301e, 0x1008,
    0x500a, 0x6024, 0x703a, 0x4020, 0x402c, 0x0014, 0x5004, 0x4018,
    0x7034, 0x0008, 0x2010, 0x0018, 0x0028, 0x2002, 0x3012, 0x3006
};

const uint16_t DataSet_1_16u::outputs::MBANDS[64] = {
    0xf210, 0x6036, 0x702c, 0x76ed, 0x200a, 0x5a96, 0x4517, 0x502e,
    0x202a, 0xd03a, 0x1b41, 0x0008, 0x4d42, 0x100e, 0x0010, 0x4317,
    0x25b7, 0x201c, 0x1018, 0x4891, 0x101c, 0xadc9, 0xeea4, 0x4008,
    0x702e, 0xa384, 0xdd59, 0x6002, 0xdee2, 0x3014, 0x103c, 0xf4a4,
    0x1721, 0x401a, 0x001c, 0x0249, 0x0004, 0x5d5d, 0x7b38, 0x5032,
    0x1024, 0xb366, 0xf184, 0x1002, 0x4229, 0x6036, 0x301e, 0x1308,
    0xd90a, 0x6024, 0x703a, 0xc2e1, 0x402c, 0x0f54, 0x5dc4, 0x4018,
    0x7034, 0x8d48, 0xa891, 0x0018, 0x02e8, 0x2002, 0x3012, 0xb387
};

const uint16_t DataSet_1_16u::outputs::BORV[64] = {
    0xf63f, 0xfbb7, 0x7e6d, 0x77ff, 0xaf7f, 0xdfff, 0xdd17, 0xff3e,
    0xa66f, 0xfcfb, 0xff5f, 0x8cd9, 0x7d47, 0x5dcf, 0x2f7d, 0x4b1f,
    0xadf7, 0x6e5f, 0xbf5b, 0x68f9, 0x7fbd, 0xefff, 0xeff6, 0xd539,
    0xffef, 0xe39f, 0xdd5b, 0xe947, 0xffef, 0xf7bc, 0x97ff, 0xf6f4,
    0x97ff, 0xcc5a, 0x369d, 0x6ef9, 0xdde5, 0xfdfd, 0xfbbb, 0xd737,
    0xdbfe, 0xbb76, 0xf1cc, 0xdf92, 0xf36d, 0xeef7, 0xfe7e, 0xb39a,
    0xfbdb, 0xfdef, 0x7f7b, 0xefe5, 0xfffd, 0x8f75, 0xdde7, 0xfcf9,
    0x7f7d, 0xcfdb, 0xed9b, 0xcadb, 0x82fd, 0xeb87, 0xf7db, 0xbbff
};

const uint16_t DataSet_1_16u::outputs::MBORV[64] = {
    0xf210, 0xfbb7, 0x7e6d, 0x76ed, 0xaf7f, 0x5a96, 0x4517, 0xff3e,
    0xa66f, 0xd03a, 0x1b41, 0x8cd9, 0x4d42, 0x5dcf, 0x2f7d, 0x4317,
    0x25b7, 0x6e5f, 0xbf5b, 0x4891, 0x7fbd, 0xadc9, 0xeea4, 0xd539,
    0xffef, 0xa384, 0xdd59, 0xe947, 0xdee2, 0xf7bc, 0x97ff, 0xf4a4,
    0x1721, 0xcc5a, 0x369d, 0x0249, 0xdde5, 0x5d5d, 0x7b38, 0xd737,
    0xdbfe, 0xb366, 0xf184, 0xdf92, 0x4229, 0xeef7, 0xfe7e, 0x1308,
    0xd90a, 0xfdef, 0x7f7b, 0xc2e1, 0xfffd, 0x0f54, 0x5dc4, 0xfcf9,
    0x7f7d, 0x8d48, 0xa891, 0xcadb, 0x02e8, 0xeb87, 0xf7db, 0xb387
};

const uint16_t DataSet_1_16u::outputs::BORS[64] = {
    0xf23e, 0x79bf, 0x7a3f, 0x76ff, 0xf57e, 0x7abe, 0x753f, 0xfe3e,
    0xf63f, 0xf03e, 0x7b7f, 0x70ff, 0x7d7e, 0x787e, 0x7e3f, 0x733f,
    0x75bf, 0x783e, 0xfc7f, 0x78bf, 0x77bf, 0xfdff, 0xfebe, 0x743f,
    0x7ebf, 0xf3be, 0xfd7f, 0xf93e, 0xfefe, 0xf5be, 0xf2fe, 0xf4be,
    0x773f, 0xfc7e, 0x74be, 0x727f, 0xfdff, 0x7d7f, 0x7b3e, 0xf43f,
    0x797e, 0xf37e, 0xf1be, 0xffbe, 0x723f, 0xfeff, 0x723e, 0x733e,
    0xf93e, 0x79ff, 0x783e, 0xf2ff, 0xf3be, 0x7f7e, 0x7dfe, 0xf07f,
    0x777f, 0xfd7e, 0xf8bf, 0xf8ff, 0x72fe, 0xf3bf, 0x77be, 0xf3bf
};

const uint16_t DataSet_1_16u::outputs::MBORS[64] = {
    0xf210, 0x79bf, 0x7a3f, 0x76ed, 0xf57e, 0x5a96, 0x4517, 0xfe3e,
    0xf63f, 0xd03a, 0x1b41, 0x70ff, 0x4d42, 0x787e, 0x7e3f, 0x4317,
    0x25b7, 0x783e, 0xfc7f, 0x4891, 0x77bf, 0xadc9, 0xeea4, 0x743f,
    0x7ebf, 0xa384, 0xdd59, 0xf93e, 0xdee2, 0xf5be, 0xf2fe, 0xf4a4,
    0x1721, 0xfc7e, 0x74be, 0x0249, 0xfdff, 0x5d5d, 0x7b38, 0xf43f,
    0x797e, 0xb366, 0xf184, 0xffbe, 0x4229, 0xfeff, 0x723e, 0x1308,
    0xd90a, 0x79ff, 0x783e, 0xc2e1, 0xf3be, 0x0f54, 0x5dc4, 0xf07f,
    0x777f, 0x8d48, 0xa891, 0xf8ff, 0x02e8, 0xf3bf, 0x77be, 0xb387
};

const uint16_t DataSet_1_16u::outputs::BXORV[64] = {
    0x963f, 0x9292, 0x5c4d, 0x473f, 0x2e7d, 0xc76f, 0xdd13, 0x6134,
    0x2265, 0xacc1, 0xee5f, 0x8c19, 0x7c07, 0x55cf, 0x2f7d, 0x0b1d,
    0x8c56, 0x6e53, 0x2743, 0x20e8, 0x78b4, 0x42f7, 0xa1f2, 0xd531,
    0xabce, 0xe31b, 0x590b, 0x0945, 0x232d, 0x73bc, 0x972f, 0x06d0,
    0x97df, 0x8450, 0x368d, 0x6cb8, 0xd0a0, 0xfcb4, 0xb8b3, 0x8336,
    0xcaba, 0x3954, 0xe048, 0x5410, 0xf344, 0x62b2, 0xcc6e, 0xb192,
    0xaad3, 0xf4ae, 0x6779, 0xad24, 0x3cd9, 0x8121, 0xd0a3, 0x7cf8,
    0x3c48, 0x429b, 0x6d8a, 0xc2d2, 0x829d, 0xeb84, 0xd1db, 0xaa78
};

const uint16_t DataSet_1_16u::outputs::MBXORV[64] = {
    0xf210, 0x9292, 0x5c4d, 0x76ed, 0x2e7d, 0x5a96, 0x4517, 0x6134,
    0x2265, 0xd03a, 0x1b41, 0x8c19, 0x4d42, 0x55cf, 0x2f7d, 0x4317,
    0x25b7, 0x6e53, 0x2743, 0x4891, 0x78b4, 0xadc9, 0xeea4, 0xd531,
    0xabce, 0xa384, 0xdd59, 0x0945, 0xdee2, 0x73bc, 0x972f, 0xf4a4,
    0x1721, 0x8450, 0x368d, 0x0249, 0xd0a0, 0x5d5d, 0x7b38, 0x8336,
    0xcaba, 0xb366, 0xf184, 0x5410, 0x4229, 0x62b2, 0xcc6e, 0x1308,
    0xd90a, 0xf4ae, 0x6779, 0xc2e1, 0x3cd9, 0x0f54, 0x5dc4, 0x7cf8,
    0x3c48, 0x8d48, 0xa891, 0xc2d2, 0x02e8, 0xeb84, 0xd1db, 0xb387
};

const uint16_t DataSet_1_16u::outputs::BXORS[64] = {
    0x822e, 0x1989, 0x0a13, 0x06d3, 0xd574, 0x2aa8, 0x3529, 0xae10,
    0xd615, 0xa004, 0x6b7f, 0x70f7, 0x3d7c, 0x6870, 0x7e2f, 0x3329,
    0x5589, 0x5822, 0xec67, 0x38af, 0x67a3, 0xddf7, 0x9e9a, 0x3437,
    0x0e91, 0xd3ba, 0xad67, 0x993c, 0xaedc, 0xc5aa, 0xe2c2, 0x849a,
    0x671f, 0xbc64, 0x74a2, 0x7277, 0xfdfb, 0x2d63, 0x0b06, 0xa40d,
    0x695a, 0xc358, 0x81ba, 0xefbc, 0x3217, 0x9ec9, 0x4220, 0x6336,
    0xa934, 0x19db, 0x0804, 0xb2df, 0xb392, 0x7f6a, 0x2dfa, 0xb067,
    0x074b, 0xfd76, 0xd8af, 0xf8e7, 0x72d6, 0xd3bd, 0x47ac, 0xc3b9
};

const uint16_t DataSet_1_16u::outputs::MBXORS[64] = {
    0xf210, 0x1989, 0x0a13, 0x76ed, 0xd574, 0x5a96, 0x4517, 0xae10,
    0xd615, 0xd03a, 0x1b41, 0x70f7, 0x4d42, 0x6870, 0x7e2f, 0x4317,
    0x25b7, 0x5822, 0xec67, 0x4891, 0x67a3, 0xadc9, 0xeea4, 0x3437,
    0x0e91, 0xa384, 0xdd59, 0x993c, 0xdee2, 0xc5aa, 0xe2c2, 0xf4a4,
    0x1721, 0xbc64, 0x74a2, 0x0249, 0xfdfb, 0x5d5d, 0x7b38, 0xa40d,
    0x695a, 0xb366, 0xf184, 0xefbc, 0x4229, 0x9ec9, 0x4220, 0x1308,
    0xd90a, 0x19db, 0x0804, 0xc2e1, 0xb392, 0x0f54, 0x5dc4, 0xb067,
    0x074b, 0x8d48, 0xa891, 0xf8e7, 0x02e8, 0xd3bd, 0x47ac, 0xb387
};

const uint16_t DataSet_1_16u::outputs::BNOT[64] = {
    0x0def, 0x9648, 0x85d2, 0x8912, 0x5ab5, 0xa569, 0xbae8, 0x21d1,
    0x59d4, 0x2fc5, 0xe4be, 0xff36, 0xb2bd, 0xe7b1, 0xf1ee, 0xbce8,
    0xda48, 0xd7e3, 0x63a6, 0xb76e, 0xe862, 0x5236, 0x115b, 0xbbf6,
    0x8150, 0x5c7b, 0x22a6, 0x16fd, 0x211d, 0x4a6b, 0x6d03, 0x0b5b,
    0xe8de, 0x33a5, 0xfb63, 0xfdb6, 0x723a, 0xa2a2, 0x84c7, 0x2bcc,
    0xe69b, 0x4c99, 0x0e7b, 0x607d, 0xbdd6, 0x1108, 0xcde1, 0xecf7,
    0x26f5, 0x961a, 0x87c5, 0x3d1e, 0x3c53, 0xf0ab, 0xa23b, 0x3fa6,
    0x888a, 0x72b7, 0x576e, 0x7726, 0xfd17, 0x5c7c, 0xc86d, 0x4c78
};

const uint16_t DataSet_1_16u::outputs::MBNOT[64] = {
    0xf210, 0x9648, 0x85d2, 0x76ed, 0x5ab5, 0x5a96, 0x4517, 0x21d1,
    0x59d4, 0xd03a, 0x1b41, 0xff36, 0x4d42, 0xe7b1, 0xf1ee, 0x4317,
    0x25b7, 0xd7e3, 0x63a6, 0x4891, 0xe862, 0xadc9, 0xeea4, 0xbbf6,
    0x8150, 0xa384, 0xdd59, 0x16fd, 0xdee2, 0x4a6b, 0x6d03, 0xf4a4,
    0x1721, 0x33a5, 0xfb63, 0x0249, 0x723a, 0x5d5d, 0x7b38, 0x2bcc,
    0xe69b, 0xb366, 0xf184, 0x607d, 0x4229, 0x1108, 0xcde1, 0x1308,
    0xd90a, 0x961a, 0x87c5, 0xc2e1, 0x3c53, 0x0f54, 0x5dc4, 0x3fa6,
    0x888a, 0x8d48, 0xa891, 0x7726, 0x02e8, 0x5c7c, 0xc86d, 0xb387
};

const uint16_t DataSet_1_16u::outputs::HBAND[64] = {
    0xf210, 0x6010, 0x6000, 0x6000, 0x2000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000
};

const uint16_t DataSet_1_16u::outputs::MHBAND[64] = {
    0xffff, 0x69b7, 0x6825, 0x6825, 0x2000, 0x2000, 0x2000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000
};

const uint16_t DataSet_1_16u::outputs::HBANDS[64] = {
    0x7010, 0x6010, 0x6000, 0x6000, 0x2000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000
};

const uint16_t DataSet_1_16u::outputs::MHBANDS[64] = {
    0x703e, 0x6036, 0x6024, 0x6024, 0x2000, 0x2000, 0x2000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000
};

const uint16_t DataSet_1_16u::outputs::HBOR[64] = {
    0xf210, 0xfbb7, 0xfbbf, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff
};

const uint16_t DataSet_1_16u::outputs::MHBOR[64] = {
    0x0000, 0x69b7, 0x7bbf, 0x7bbf, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff
};

const uint16_t DataSet_1_16u::outputs::HBORS[64] = {
    0xf23e, 0xfbbf, 0xfbbf, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff
};

const uint16_t DataSet_1_16u::outputs::MHBORS[64] = {
    0x703e, 0x79bf, 0x7bbf, 0x7bbf, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff
};

const uint16_t DataSet_1_16u::outputs::HBXOR[64] = {
    0xf210, 0x9ba7, 0xe18a, 0x9767, 0x322d, 0x68bb, 0x2dac, 0xf382,
    0x55a9, 0x8593, 0x9ed2, 0x9e1b, 0xd359, 0xcb17, 0xc506, 0x8611,
    0xa3a6, 0x8bba, 0x17e3, 0x5f72, 0x48ef, 0xe526, 0x0b82, 0x4f8b,
    0x3124, 0x92a0, 0x4ff9, 0xa6fb, 0x7819, 0xcd8d, 0x5f71, 0xabd5,
    0xbcf4, 0x70ae, 0x7432, 0x767b, 0xfbbe, 0xa6e3, 0xdddb, 0x09e8,
    0x108c, 0xa3ea, 0x526e, 0xcdec, 0x8fc5, 0x6132, 0x532c, 0x4024,
    0x992e, 0xf0cb, 0x88f1, 0x4a10, 0x89bc, 0x86e8, 0xdb2c, 0x1b75,
    0x6c00, 0xe148, 0x49d9, 0xc100, 0xc3e8, 0x606b, 0x57f9, 0xe47e
};

const uint16_t DataSet_1_16u::outputs::MHBXOR[64] = {
    0x0000, 0x69b7, 0x139a, 0x139a, 0xb6d0, 0xb6d0, 0xb6d0, 0x68fe,
    0xced5, 0xced5, 0xced5, 0xce1c, 0xce1c, 0xd652, 0xd843, 0xd843,
    0xd843, 0xf05f, 0x6c06, 0x6c06, 0x7b9b, 0x7b9b, 0x7b9b, 0x3f92,
    0x413d, 0x413d, 0x413d, 0xa83f, 0xa83f, 0x1dab, 0x8f57, 0x8f57,
    0x8f57, 0x430d, 0x4791, 0x4791, 0xca54, 0xca54, 0xca54, 0x1e67,
    0x0703, 0x0703, 0x0703, 0x9881, 0x9881, 0x7676, 0x4468, 0x4468,
    0x4468, 0x2d8d, 0x55b7, 0x55b7, 0x961b, 0x961b, 0x961b, 0x5642,
    0x2137, 0x2137, 0x2137, 0xa9ee, 0xa9ee, 0x0a6d, 0x3dff, 0x3dff
};

const uint16_t DataSet_1_16u::outputs::HBXORS[64] = {
    0x822e, 0xeb99, 0x91b4, 0xe759, 0x4213, 0x1885, 0x5d92, 0x83bc,
    0x2597, 0xf5ad, 0xeeec, 0xee25, 0xa367, 0xbb29, 0xb538, 0xf62f,
    0xd398, 0xfb84, 0x67dd, 0x2f4c, 0x38d1, 0x9518, 0x7bbc, 0x3fb5,
    0x411a, 0xe29e, 0x3fc7, 0xd6c5, 0x0827, 0xbdb3, 0x2f4f, 0xdbeb,
    0xccca, 0x0090, 0x040c, 0x0645, 0x8b80, 0xd6dd, 0xade5, 0x79d6,
    0x60b2, 0xd3d4, 0x2250, 0xbdd2, 0xfffb, 0x110c, 0x2312, 0x301a,
    0xe910, 0x80f5, 0xf8cf, 0x3a2e, 0xf982, 0xf6d6, 0xab12, 0x6b4b,
    0x1c3e, 0x9176, 0x39e7, 0xb13e, 0xb3d6, 0x1055, 0x27c7, 0x9440
};

const uint16_t DataSet_1_16u::outputs::MHBXORS[64] = {
    0x703e, 0x1989, 0x63a4, 0x63a4, 0xc6ee, 0xc6ee, 0xc6ee, 0x18c0,
    0xbeeb, 0xbeeb, 0xbeeb, 0xbe22, 0xbe22, 0xa66c, 0xa87d, 0xa87d,
    0xa87d, 0x8061, 0x1c38, 0x1c38, 0x0ba5, 0x0ba5, 0x0ba5, 0x4fac,
    0x3103, 0x3103, 0x3103, 0xd801, 0xd801, 0x6d95, 0xff69, 0xff69,
    0xff69, 0x3333, 0x37af, 0x37af, 0xba6a, 0xba6a, 0xba6a, 0x6e59,
    0x773d, 0x773d, 0x773d, 0xe8bf, 0xe8bf, 0x0648, 0x3456, 0x3456,
    0x3456, 0x5db3, 0x2589, 0x2589, 0xe625, 0xe625, 0xe625, 0x267c,
    0x5109, 0x5109, 0x5109, 0xd9d0, 0xd9d0, 0x7a53, 0x4dc1, 0x4dc1
};

const uint16_t DataSet_1_16u::outputs::LSHV[64] = {
    0x9080, 0x8000, 0x8b40, 0x4000, 0x2800, 0xb52c, 0x145c, 0xe000,
    0x2b00, 0x81d0, 0xda08, 0x2400, 0x6a10, 0xc000, 0x7088, 0xb800,
    0x4b6e, 0x0000, 0x38b2, 0x2200, 0x7400, 0x7240, 0x4800, 0x0240,
    0xaf00, 0x8400, 0xac80, 0x0400, 0x2000, 0xd650, 0x25f8, 0x0000,
    0x0800, 0xc5a0, 0x1270, 0x0924, 0x8a00, 0x7574, 0x8000, 0x1980,
    0x8000, 0x3660, 0x1000, 0x8000, 0x2000, 0xee00, 0xc000, 0x0800,
    0x2800, 0xa794, 0x4000, 0x8400, 0x0eb0, 0x8000, 0xdc40, 0x4000,
    0x4000, 0x9000, 0x5122, 0x3640, 0x1740, 0x3830, 0x8000, 0x9c38
};

const uint16_t DataSet_1_16u::outputs::MLSHV[64] = {
    0xf210, 0x8000, 0x8b40, 0x76ed, 0x2800, 0x5a96, 0x4517, 0xe000,
    0x2b00, 0xd03a, 0x1b41, 0x2400, 0x4d42, 0xc000, 0x7088, 0x4317,
    0x25b7, 0x0000, 0x38b2, 0x4891, 0x7400, 0xadc9, 0xeea4, 0x0240,
    0xaf00, 0xa384, 0xdd59, 0x0400, 0xdee2, 0xd650, 0x25f8, 0xf4a4,
    0x1721, 0xc5a0, 0x1270, 0x0249, 0x8a00, 0x5d5d, 0x7b38, 0x1980,
    0x8000, 0xb366, 0xf184, 0x8000, 0x4229, 0xee00, 0xc000, 0x1308,
    0xd90a, 0xa794, 0x4000, 0xc2e1, 0x0eb0, 0x0f54, 0x5dc4, 0x4000,
    0x4000, 0x8d48, 0xa891, 0x3640, 0x02e8, 0x3830, 0x8000, 0xb387
};

const uint16_t DataSet_1_16u::outputs::LSHS[64] = {
    0x2000, 0x6e00, 0x5a00, 0xda00, 0x9400, 0x2c00, 0x2e00, 0x5c00,
    0x5600, 0x7400, 0x8200, 0x9200, 0x8400, 0x9c00, 0x2200, 0x2e00,
    0x6e00, 0x3800, 0xb200, 0x2200, 0x3a00, 0x9200, 0x4800, 0x1200,
    0x5e00, 0x0800, 0xb200, 0x0400, 0xc400, 0x2800, 0xf800, 0x4800,
    0x4200, 0xb400, 0x3800, 0x9200, 0x8a00, 0xba00, 0x7000, 0x6600,
    0xc800, 0xcc00, 0x0800, 0x0400, 0x5200, 0xee00, 0x3c00, 0x1000,
    0x1400, 0xca00, 0x7400, 0xc200, 0x5800, 0xa800, 0x8800, 0xb200,
    0xea00, 0x9000, 0x2200, 0xb200, 0xd000, 0x0600, 0x2400, 0x0e00
};

const uint16_t DataSet_1_16u::outputs::MLSHS[64] = {
    0xf210, 0x6e00, 0x5a00, 0x76ed, 0x9400, 0x5a96, 0x4517, 0x5c00,
    0x5600, 0xd03a, 0x1b41, 0x9200, 0x4d42, 0x9c00, 0x2200, 0x4317,
    0x25b7, 0x3800, 0xb200, 0x4891, 0x3a00, 0xadc9, 0xeea4, 0x1200,
    0x5e00, 0xa384, 0xdd59, 0x0400, 0xdee2, 0x2800, 0xf800, 0xf4a4,
    0x1721, 0xb400, 0x3800, 0x0249, 0x8a00, 0x5d5d, 0x7b38, 0x6600,
    0xc800, 0xb366, 0xf184, 0x0400, 0x4229, 0xee00, 0x3c00, 0x1308,
    0xd90a, 0xca00, 0x7400, 0xc2e1, 0x5800, 0x0f54, 0x5dc4, 0xb200,
    0xea00, 0x8d48, 0xa891, 0xb200, 0x02e8, 0x0600, 0x2400, 0xb387
};

const uint16_t DataSet_1_16u::outputs::RSHV[64] = {
    0x1e42, 0x0000, 0x01e8, 0x0001, 0x0029, 0x2d4b, 0x1145, 0x000d,
    0x00a6, 0x1a07, 0x0368, 0x0000, 0x09a8, 0x0000, 0x01c2, 0x0008,
    0x12db, 0x0000, 0x4e2c, 0x0024, 0x0005, 0x02b7, 0x0077, 0x0110,
    0x007e, 0x00a3, 0x01ba, 0x0074, 0x000d, 0x2d65, 0x497e, 0x0003,
    0x0002, 0x0cc5, 0x0127, 0x0092, 0x0046, 0x1757, 0x0007, 0x01a8,
    0x0000, 0x0b36, 0x003c, 0x0002, 0x0002, 0x0077, 0x0001, 0x0013,
    0x0036, 0x1a79, 0x0003, 0x0030, 0x30eb, 0x0000, 0x05dc, 0x0003,
    0x0001, 0x0046, 0x5448, 0x0223, 0x005d, 0x0a38, 0x0000, 0x1670
};

const uint16_t DataSet_1_16u::outputs::MRSHV[64] = {
    0xf210, 0x0000, 0x01e8, 0x76ed, 0x0029, 0x5a96, 0x4517, 0x000d,
    0x00a6, 0xd03a, 0x1b41, 0x0000, 0x4d42, 0x0000, 0x01c2, 0x4317,
    0x25b7, 0x0000, 0x4e2c, 0x4891, 0x0005, 0xadc9, 0xeea4, 0x0110,
    0x007e, 0xa384, 0xdd59, 0x0074, 0xdee2, 0x2d65, 0x497e, 0xf4a4,
    0x1721, 0x0cc5, 0x0127, 0x0249, 0x0046, 0x5d5d, 0x7b38, 0x01a8,
    0x0000, 0xb366, 0xf184, 0x0002, 0x4229, 0x0077, 0x0001, 0x1308,
    0xd90a, 0x1a79, 0x0003, 0xc2e1, 0x30eb, 0x0f54, 0x5dc4, 0x0003,
    0x0001, 0x8d48, 0xa891, 0x0223, 0x02e8, 0x0a38, 0x0000, 0xb387
};

const uint16_t DataSet_1_16u::outputs::RSHS[64] = {
    0x0079, 0x0034, 0x003d, 0x003b, 0x0052, 0x002d, 0x0022, 0x006f,
    0x0053, 0x0068, 0x000d, 0x0000, 0x0026, 0x000c, 0x0007, 0x0021,
    0x0012, 0x0014, 0x004e, 0x0024, 0x000b, 0x0056, 0x0077, 0x0022,
    0x003f, 0x0051, 0x006e, 0x0074, 0x006f, 0x005a, 0x0049, 0x007a,
    0x000b, 0x0066, 0x0002, 0x0001, 0x0046, 0x002e, 0x003d, 0x006a,
    0x000c, 0x0059, 0x0078, 0x004f, 0x0021, 0x0077, 0x0019, 0x0009,
    0x006c, 0x0034, 0x003c, 0x0061, 0x0061, 0x0007, 0x002e, 0x0060,
    0x003b, 0x0046, 0x0054, 0x0044, 0x0001, 0x0051, 0x001b, 0x0059
};

const uint16_t DataSet_1_16u::outputs::MRSHS[64] = {
    0xf210, 0x0034, 0x003d, 0x76ed, 0x0052, 0x5a96, 0x4517, 0x006f,
    0x0053, 0xd03a, 0x1b41, 0x0000, 0x4d42, 0x000c, 0x0007, 0x4317,
    0x25b7, 0x0014, 0x004e, 0x4891, 0x000b, 0xadc9, 0xeea4, 0x0022,
    0x003f, 0xa384, 0xdd59, 0x0074, 0xdee2, 0x005a, 0x0049, 0xf4a4,
    0x1721, 0x0066, 0x0002, 0x0249, 0x0046, 0x5d5d, 0x7b38, 0x006a,
    0x000c, 0xb366, 0xf184, 0x004f, 0x4229, 0x0077, 0x0019, 0x1308,
    0xd90a, 0x0034, 0x003c, 0xc2e1, 0x0061, 0x0f54, 0x5dc4, 0x0060,
    0x003b, 0x8d48, 0xa891, 0x0044, 0x02e8, 0x0051, 0x001b, 0xb387
};

const uint16_t DataSet_1_16u::outputs::ROLV[64] = {
    0x9087, 0xb4db, 0x8b5e, 0x5dbb, 0x2a95, 0xb52c, 0x145d, 0xede2,
    0x2ba6, 0x81d6, 0xda08, 0x2403, 0x6a12, 0xc309, 0x7088, 0xba18,
    0x4b6e, 0x0a07, 0x38b3, 0x2291, 0x745e, 0x726b, 0x49dd, 0x0251,
    0xaf7e, 0x84a3, 0xacee, 0x05d2, 0x2dee, 0xd652, 0x25f9, 0x3d29,
    0x08b9, 0xc5ac, 0x1270, 0x0924, 0x8b1b, 0x7575, 0x87b3, 0x19ea,
    0x832c, 0x366b, 0x13c6, 0xa7e0, 0x2845, 0xefdd, 0xc643, 0x0813,
    0x2b64, 0xa795, 0x4f07, 0x870b, 0x0eb3, 0x81ea, 0xdc45, 0x7016,
    0x5ddd, 0x911a, 0x5123, 0x3662, 0x1740, 0x383a, 0x8de4, 0x9c3d
};

const uint16_t DataSet_1_16u::outputs::MROLV[64] = {
    0xf210, 0xb4db, 0x8b5e, 0x76ed, 0x2a95, 0x5a96, 0x4517, 0xede2,
    0x2ba6, 0xd03a, 0x1b41, 0x2403, 0x4d42, 0xc309, 0x7088, 0x4317,
    0x25b7, 0x0a07, 0x38b3, 0x4891, 0x745e, 0xadc9, 0xeea4, 0x0251,
    0xaf7e, 0xa384, 0xdd59, 0x05d2, 0xdee2, 0xd652, 0x25f9, 0xf4a4,
    0x1721, 0xc5ac, 0x1270, 0x0249, 0x8b1b, 0x5d5d, 0x7b38, 0x19ea,
    0x832c, 0xb366, 0xf184, 0xa7e0, 0x4229, 0xefdd, 0xc643, 0x1308,
    0xd90a, 0xa795, 0x4f07, 0xc2e1, 0x0eb3, 0x0f54, 0x5dc4, 0x7016,
    0x5ddd, 0x8d48, 0xa891, 0x3662, 0x02e8, 0x383a, 0x8de4, 0xb387
};

const uint16_t DataSet_1_16u::outputs::ROLS[64] = {
    0x21e4, 0x6ed3, 0x5af4, 0xdaed, 0x954a, 0x2cb5, 0x2e8a, 0x5dbc,
    0x574c, 0x75a0, 0x8236, 0x9201, 0x849a, 0x9c30, 0x221c, 0x2e86,
    0x6e4b, 0x3850, 0xb338, 0x2291, 0x3a2f, 0x935b, 0x49dd, 0x1288,
    0x5efd, 0x0947, 0xb3ba, 0x05d2, 0xc5bd, 0x296b, 0xf925, 0x49e9,
    0x422e, 0xb598, 0x3809, 0x9204, 0x8b1b, 0xbaba, 0x70f6, 0x67a8,
    0xc832, 0xcd66, 0x09e3, 0x053f, 0x5284, 0xefdd, 0x3c64, 0x1026,
    0x15b2, 0xcad3, 0x74f0, 0xc385, 0x5987, 0xa81e, 0x88bb, 0xb380,
    0xeaee, 0x911a, 0x2351, 0xb311, 0xd005, 0x0747, 0x246f, 0x0f67
};

const uint16_t DataSet_1_16u::outputs::MROLS[64] = {
    0xf210, 0x6ed3, 0x5af4, 0x76ed, 0x954a, 0x5a96, 0x4517, 0x5dbc,
    0x574c, 0xd03a, 0x1b41, 0x9201, 0x4d42, 0x9c30, 0x221c, 0x4317,
    0x25b7, 0x3850, 0xb338, 0x4891, 0x3a2f, 0xadc9, 0xeea4, 0x1288,
    0x5efd, 0xa384, 0xdd59, 0x05d2, 0xdee2, 0x296b, 0xf925, 0xf4a4,
    0x1721, 0xb598, 0x3809, 0x0249, 0x8b1b, 0x5d5d, 0x7b38, 0x67a8,
    0xc832, 0xb366, 0xf184, 0x053f, 0x4229, 0xefdd, 0x3c64, 0x1308,
    0xd90a, 0xcad3, 0x74f0, 0xc2e1, 0x5987, 0x0f54, 0x5dc4, 0xb380,
    0xeaee, 0x8d48, 0xa891, 0xb311, 0x02e8, 0x0747, 0x246f, 0xb387
};

const uint16_t DataSet_1_16u::outputs::RORV[64] = {
    0x1e42, 0xd36e, 0xb5e8, 0xdbb5, 0x52a9, 0x2d4b, 0xd145, 0xe2ed,
    0x2ba6, 0x5a07, 0x2368, 0x3240, 0x49a8, 0xc270, 0x21c2, 0x62e8,
    0x92db, 0xa070, 0xce2c, 0x48a4, 0xe745, 0x26b7, 0x5277, 0x2510,
    0xaf7e, 0x84a3, 0xb3ba, 0x8174, 0xee2d, 0x2d65, 0x497e, 0xd293,
    0xe422, 0xacc5, 0x0127, 0x4092, 0xe2c6, 0x5757, 0xb387, 0x67a8,
    0xcb20, 0x6b36, 0x613c, 0x7e0a, 0x114a, 0x7bf7, 0x90f1, 0x0813,
    0x42b6, 0x5a79, 0xc1d3, 0xb870, 0x30eb, 0x7aa0, 0x45dc, 0x0167,
    0xddd5, 0xa446, 0xd448, 0x6623, 0x005d, 0x3a38, 0xde48, 0xf670
};

const uint16_t DataSet_1_16u::outputs::MRORV[64] = {
    0xf210, 0xd36e, 0xb5e8, 0x76ed, 0x52a9, 0x5a96, 0x4517, 0xe2ed,
    0x2ba6, 0xd03a, 0x1b41, 0x3240, 0x4d42, 0xc270, 0x21c2, 0x4317,
    0x25b7, 0xa070, 0xce2c, 0x4891, 0xe745, 0xadc9, 0xeea4, 0x2510,
    0xaf7e, 0xa384, 0xdd59, 0x8174, 0xdee2, 0x2d65, 0x497e, 0xf4a4,
    0x1721, 0xacc5, 0x0127, 0x0249, 0xe2c6, 0x5d5d, 0x7b38, 0x67a8,
    0xcb20, 0xb366, 0xf184, 0x7e0a, 0x4229, 0x7bf7, 0x90f1, 0x1308,
    0xd90a, 0x5a79, 0xc1d3, 0xc2e1, 0x30eb, 0x0f54, 0x5dc4, 0x0167,
    0xddd5, 0x8d48, 0xa891, 0x6623, 0x02e8, 0x3a38, 0xde48, 0xb387
};

const uint16_t DataSet_1_16u::outputs::RORS[64] = {
    0x0879, 0xdbb4, 0x16bd, 0x76bb, 0xa552, 0x4b2d, 0x8ba2, 0x176f,
    0x15d3, 0x1d68, 0xa08d, 0x6480, 0xa126, 0x270c, 0x0887, 0x8ba1,
    0xdb92, 0x0e14, 0x2cce, 0x48a4, 0xce8b, 0xe4d6, 0x5277, 0x04a2,
    0x57bf, 0xc251, 0xacee, 0x8174, 0x716f, 0xca5a, 0x7e49, 0x527a,
    0x908b, 0x2d66, 0x4e02, 0x2481, 0xe2c6, 0xaeae, 0x9c3d, 0x19ea,
    0xb20c, 0xb359, 0xc278, 0xc14f, 0x14a1, 0x7bf7, 0x0f19, 0x8409,
    0x856c, 0xf2b4, 0x1d3c, 0x70e1, 0xd661, 0xaa07, 0xe22e, 0x2ce0,
    0xbabb, 0xa446, 0x48d4, 0x6cc4, 0x7401, 0xc1d1, 0xc91b, 0xc3d9
};

const uint16_t DataSet_1_16u::outputs::MRORS[64] = {
    0xf210, 0xdbb4, 0x16bd, 0x76ed, 0xa552, 0x5a96, 0x4517, 0x176f,
    0x15d3, 0xd03a, 0x1b41, 0x6480, 0x4d42, 0x270c, 0x0887, 0x4317,
    0x25b7, 0x0e14, 0x2cce, 0x4891, 0xce8b, 0xadc9, 0xeea4, 0x04a2,
    0x57bf, 0xa384, 0xdd59, 0x8174, 0xdee2, 0xca5a, 0x7e49, 0xf4a4,
    0x1721, 0x2d66, 0x4e02, 0x0249, 0xe2c6, 0x5d5d, 0x7b38, 0x19ea,
    0xb20c, 0xb366, 0xf184, 0xc14f, 0x4229, 0x7bf7, 0x0f19, 0x1308,
    0xd90a, 0xf2b4, 0x1d3c, 0xc2e1, 0xd661, 0x0f54, 0x5dc4, 0x2ce0,
    0xbabb, 0x8d48, 0xa891, 0x6cc4, 0x02e8, 0xc1d1, 0xc91b, 0xb387
};
/*
const int16_t DataSet_1_16u::outputs::UTOI[64] = {
    0xf210, 0x69b7, 0x7a2d, 0x76ed, 0xa54a, 0x5a96, 0x4517, 0xde2e,
    0xa62b, 0xd03a, 0x1b41, 0x00c9, 0x4d42, 0x184e, 0x0e11, 0x4317,
    0x25b7, 0x281c, 0x9c59, 0x4891, 0x179d, 0xadc9, 0xeea4, 0x4409,
    0x7eaf, 0xa384, 0xdd59, 0xe902, 0xdee2, 0xb594, 0x92fc, 0xf4a4,
    0x1721, 0xcc5a, 0x049c, 0x0249, 0x8dc5, 0x5d5d, 0x7b38, 0xd433,
    0x1964, 0xb366, 0xf184, 0x9f82, 0x4229, 0xeef7, 0x321e, 0x1308,
    0xd90a, 0x69e5, 0x783a, 0xc2e1, 0xc3ac, 0x0f54, 0x5dc4, 0xc059,
    0x7775, 0x8d48, 0xa891, 0x88d9, 0x02e8, 0xa383, 0x3792, 0xb387
};
*/

const int16_t DataSet_1_16u::outputs::UTOI[64] = {
    -3568,  27063,  31277,  30445,  -23222, 23190,  17687,  -8658,
    -22997, -12230, 6977,   201,    19778,  6222,   3601,   17175,
    9655,   10268,  -25511, 18577,  6045,   -21047, -4444,  17417,
    32431,  -23676, -8871,  -5886,  -8478,  -19052, -27908, -2908,
    5921,   -13222, 1180,   585,    -29243, 23901,  31544,  -11213,
    6500,   -19610, -3708,  -24702, 16937,  -4361,  12830,  4872,
    -9974,  27109,  30778,  -15647, -15444, 3924,   24004,  -16295,
    30581,  -29368, -22383, -30503, 744,    -23677, 14226,  -19577
};





const int16_t DataSet_1_16i::inputs::inputA[64] = {
    11410,  -30737, 1891,   -27538, 15774,  -31569, -18871, 23582,
    -7666,  -10559, -5498,  11945,  9682,   -29406, -2020,  23203,
    13653,  -16286, -17559, -32069, -21940, 4552,   -23502, 2675,
    -7754,  1169,   26002,  -4574,  7522,   -3612,  -7286,  7417,
    23067,  -15428, 1020,   -10491, 27755,  28077,  -26090, -21890,
    5233,   17615,  -6561,  -6107,  15073,  14944,  16256,  19940,
    -26909, 29802,  17682,  18148,  -16451, 16384,  4073,   -10887,
    -10946, -17841, -13193, 6887,   30870,  -13707, 19001,  -20674
};

const int16_t DataSet_1_16i::inputs::inputB[64] = {
    6160,   -17357, 5265,   -7340,  -26278, -11194, -27064, 23474,
    -27932, 28303,  -3511,  -28986, -31275, 28980,  15091,  -14271,
    -13798, 1335,   -22090, -15080, -31205, -24339, -28877, -2754,
    14106,  -29733, 30657,  -2597,  -14211, -11101, 22960,  12243,
    -18765, -8196,  18475,  5734,   -9602,  -23575, -30397, 24913,
    28185,  7633,   -22685, 1819,   -30338, -20903, 1973,   -17631,
    -5881,  1212,   -13367, 19758,  -19533, 28602,  4149,   -11921,
    -7949,  7422,   -11752, 21986,  15778,  27027,  13153,  -29983
};

const int16_t DataSet_1_16i::inputs::inputC[64] = {
    -32070, 74,     18974,  19343,  30212,  32470,  -27880, 4950,
    -18177, 12707,  -7496,  -4227,  2477,   -29067, 6559,   -4071,
    22210,  21386,  -24976, -14163, 16283,  24340,  -7345,  -2724,
    -30140, 17116,  28143,  -25111, -24828, -863,   361,    16691,
    -22306, -15943, -8402,  -24151, 5821,   17840,  -32422, -8364,
    -15100, -31048, -27810, 20769,  -32700, 5569,   -20762, -32305,
    23703,  -25720, -5181,  30479,  1208,   -21851, -4541,  1881,
    -29434, 1388,   -10529, -28891, -20402, 10430,  13980,  -24684
};

const uint16_t DataSet_1_16i::inputs::inputShiftA[64] = {
    3,  11, 13, 9,  3,  13, 9,  3,
    13, 15, 10, 3,  9,  6,  5,  15,
    9,  11, 15, 15, 5,  11, 2,  9,
    6,  15, 2,  10, 10, 10, 12, 7,
    11, 13, 14, 11, 8,  1,  4,  13,
    10, 8,  14, 2,  15, 11, 9,  15,
    7,  10, 13, 5,  8,  12, 13, 8,
    4,  8,  11, 2,  7,  4,  13, 3
};

const int16_t DataSet_1_16i::inputs::scalarA = 14;
const uint16_t DataSet_1_16i::inputs::inputShiftScalarA = 3;

const bool    DataSet_1_16i::inputs::maskA[64] = {
    false,                              // 1
    true,                               // 2
    true,   false,                      // 4
    true,   false,  false,  true,       // 8
    true,   false,  false,  true,
    false,  true,   true,   false,      // 16
    false,  true,   true,   false,
    true,   false,  false,  true,
    true,   false,  false,  true,
    false,  true,   true,   false,      // 32
    false,  true,   true,   false,
    true,   false,  false,  true,
    true,   false,  false,  true,
    false,  true,   true,   false,
    false,  true,   true,   false,
    true,   false,  false,  true,
    true,   false,  false,  true,
    false,  true,   true,   false,      // 64
};

const bool  DataSet_1_16i::outputs::CMPEV = false;
const bool  DataSet_1_16i::outputs::CMPES = false;

const int16_t DataSet_1_16i::outputs::ADDV[64] = {
    17570,  17442,  7156,   30658,  -10504, 22773,  19601,  -18480,
    29938,  17744,  -9009,  -17041, -21593, -426,   13071,  8932,
    -145,   -14951, 25887,  18387,  12391,  -19787, 13157,  -79,
    6352,   -28564, -8877,  -7171,  -6689,  -14713, 15674,  19660,
    4302,   -23624, 19495,  -4757,  18153,  4502,   9049,   3023,
    -32118, 25248,  -29246, -4288,  -15265, -5959,  18229,  2309,
    32746,  31014,  4315,   -27630, 29552,  -20550, 8222,   -22808,
    -18895, -10419, -24945, 28873,  -18888, 13320,  32154,  14879
};

const int16_t DataSet_1_16i::outputs::MADDV[64] = {
    11410,  17442,  7156,   -27538, -10504, -31569, -18871, -18480,
    29938,  -10559, -5498,  -17041, 9682,   -426,   13071,  23203,
    13653,  -14951, 25887,  -32069, 12391,  4552,   -23502, -79,
    6352,   1169,   26002,  -7171,  7522,   -14713, 15674,  7417,
    23067,  -23624, 19495,  -10491, 18153,  28077,  -26090, 3023,
    -32118, 17615,  -6561,  -4288,  15073,  -5959,  18229,  19940,
    -26909, 31014,  4315,   18148,  29552,  16384,  4073,   -22808,
    -18895, -17841, -13193, 28873,  30870,  13320,  32154,  -20674
};

const int16_t DataSet_1_16i::outputs::ADDS[64] = {
    11424,  -30723, 1905,   -27524, 15788,  -31555, -18857, 23596,
    -7652,  -10545, -5484,  11959,  9696,   -29392, -2006,  23217,
    13667,  -16272, -17545, -32055, -21926, 4566,   -23488, 2689,
    -7740,  1183,   26016,  -4560,  7536,   -3598,  -7272,  7431,
    23081,  -15414, 1034,   -10477, 27769,  28091,  -26076, -21876,
    5247,   17629,  -6547,  -6093,  15087,  14958,  16270,  19954,
    -26895, 29816,  17696,  18162,  -16437, 16398,  4087,   -10873,
    -10932, -17827, -13179, 6901,   30884,  -13693, 19015,  -20660
};

const int16_t DataSet_1_16i::outputs::MADDS[64] = {
    11410,  -30723, 1905,   -27538, 15788,  -31569, -18871, 23596,
    -7652,  -10559, -5498,  11959,  9682,   -29392, -2006,  23203,
    13653,  -16272, -17545, -32069, -21926, 4552,   -23502, 2689,
    -7740,  1169,   26002,  -4560,  7522,   -3598,  -7272,  7417,
    23067,  -15414, 1034,   -10491, 27769,  28077,  -26090, -21876,
    5247,   17615,  -6561,  -6093,  15073,  14958,  16270,  19940,
    -26909, 29816,  17696,  18148,  -16437, 16384,  4073,   -10873,
    -10932, -17841, -13193, 6901,   30870,  -13693, 19015,  -20674
};

const int16_t DataSet_1_16i::outputs::POSTPREFINC[64] = {
    11411,  (int16_t)-30736, 1892,   -27537, 15775,  -31568, -18870, 23583,
    -7665,  -10558, -5497,  11946,  9683,   -29405, -2019,  23204,
    13654,  -16285, -17558, -32068, -21939, 4553,   -23501, 2676,
    -7753,  1170,   26003,  -4573,  7523,   -3611,  -7285,  7418,
    23068,  -15427, 1021,   -10490, 27756,  28078,  -26089, -21889,
    5234,   17616,  -6560,  -6106,  15074,  14945,  16257,  19941,
    -26908, 29803,  17683,  18149,  -16450, 16385,  4074,   -10886,
    -10945, -17840, -13192, 6888,   30871,  -13706, 19002,  -20673
};

const int16_t DataSet_1_16i::outputs::MPOSTPREFINC[64] = {
    11410,  -30736, 1892,   -27538, 15775,  -31569, -18871, 23583,
    -7665,  -10559, -5498,  11946,  9682,   -29405, -2019,  23203,
    13653,  -16285, -17558, -32069, -21939, 4552,   -23502, 2676,
    -7753,  1169,   26002,  -4573,  7522,   -3611,  -7285,  7417,
    23067,  -15427, 1021,   -10491, 27756,  28077,  -26090, -21889,
    5234,   17615,  -6561,  -6106,  15073,  14945,  16257,  19940,
    -26909, 29803,  17683,  18148,  -16450, 16384,  4073,   -10886,
    -10945, -17841, -13193, 6888,   30870,  -13706, 19002,  -20674
};

const int16_t DataSet_1_16i::outputs::SUBV[64] = {
    5250,   -13380, -3374,  -20198, -23484, -20375, 8193,   108,
    20266,  26674,  -1987,  -24605, -24579, 7150,   -17111, -28062,
    27451,  -17621, 4531,   -16989, 9265,   28891,  5375,   5429,
    -21860, 30902,  -4655,  -1977,  21733,  7489,   -30246, -4826,
    -23704, -7232,  -17455, -16225, -28179, -13884, 4307,   18733,
    -22952, 9982,   16124,  -7926,  -20125, -29689, 14283,  -27965,
    -21028, 28590,  31049,  -1610,  3082,   -12218, -76,    1034,
    -2997,  -25263, -1441,  -15099, 15092,  24802,  5848,   9309
};

const int16_t DataSet_1_16i::outputs::MSUBV[64] = {
    11410,  -13380, -3374,  -27538, -23484, -31569, -18871, 108,
    20266,  -10559, -5498,  -24605, 9682,   7150,   -17111, 23203,
    13653,  -17621, 4531,   -32069, 9265,   4552,   -23502, 5429,
    -21860, 1169,   26002,  -1977,  7522,   7489,   -30246, 7417,
    23067,  -7232,  -17455, -10491, -28179, 28077,  -26090, 18733,
    -22952, 17615,  -6561,  -7926,  15073,  -29689, 14283,  19940,
    -26909, 28590,  31049,  18148,  3082,   16384,  4073,   1034,
    -2997,  -17841, -13193, -15099, 30870,  24802,  5848,   -20674
};

const int16_t DataSet_1_16i::outputs::SUBS[64] = {
    11396,  -30751, 1877,   -27552, 15760,  -31583, -18885, 23568,
    -7680,  -10573, -5512,  11931,  9668,   -29420, -2034,  23189,
    13639,  -16300, -17573, -32083, -21954, 4538,   -23516, 2661,
    -7768,  1155,   25988,  -4588,  7508,   -3626,  -7300,  7403,
    23053,  -15442, 1006,   -10505, 27741,  28063,  -26104, -21904,
    5219,   17601,  -6575,  -6121,  15059,  14930,  16242,  19926,
    -26923, 29788,  17668,  18134,  -16465, 16370,  4059,   -10901,
    -10960, -17855, -13207, 6873,   30856,  -13721, 18987,  -20688
};

const int16_t DataSet_1_16i::outputs::MSUBS[64] = {
    11410,  -30751, 1877,   -27538, 15760,  -31569, -18871, 23568,
    -7680,  -10559, -5498,  11931,  9682,   -29420, -2034,  23203,
    13653,  -16300, -17573, -32069, -21954, 4552,   -23502, 2661,
    -7768,  1169,   26002,  -4588,  7522,   -3626,  -7300,  7417,
    23067,  -15442, 1006,   -10491, 27741,  28077,  -26090, -21904,
    5219,   17615,  -6561,  -6121,  15073,  14930,  16242,  19940,
    -26909, 29788,  17668,  18148,  -16465, 16384,  4073,   -10901,
    -10960, -17841, -13193, 6873,   30870,  -13721, 18987,  -20674
};

const int16_t DataSet_1_16i::outputs::SUBFROMV[64] = {
    (int16_t)-5250,   (int16_t)13380,   (int16_t)3374,    (int16_t)20198,   (int16_t)23484,   (int16_t)20375,   (int16_t)-8193,   (int16_t)-108,
    (int16_t)-20266,  (int16_t)-26674,  (int16_t)1987,    (int16_t)24605,   (int16_t)24579,   (int16_t)-7150,   (int16_t)17111,   (int16_t)28062,
    (int16_t)-27451,  (int16_t)17621,   (int16_t)-4531,   (int16_t)16989,   (int16_t)-9265,   (int16_t)-28891,  (int16_t)-5375,   (int16_t)-5429,
    (int16_t)21860,   (int16_t)-30902,  (int16_t)4655,    (int16_t)1977,    (int16_t)-21733,  (int16_t)-7489,   (int16_t)30246,   (int16_t)4826,
    (int16_t)23704,   (int16_t)7232,    (int16_t)17455,   (int16_t)16225,   (int16_t)28179,   (int16_t)13884,   (int16_t)-4307,   (int16_t)-18733,
    (int16_t)22952,   (int16_t)-9982,   (int16_t)-16124,  (int16_t)7926,    (int16_t)20125,   (int16_t)29689,   (int16_t)-14283,  (int16_t)27965,
    (int16_t)21028,   (int16_t)-28590,  (int16_t)-31049,  (int16_t)1610,    (int16_t)-3082,   (int16_t)12218,   (int16_t)76,      (int16_t)-1034,
    (int16_t)2997,    (int16_t)25263,   (int16_t)1441,    (int16_t)15099,   (int16_t)-15092,  (int16_t)-24802,  (int16_t)-5848,   (int16_t)-9309
};

const int16_t DataSet_1_16i::outputs::MSUBFROMV[64] = {
    (int16_t)6160,    (int16_t)13380,   (int16_t)3374,    (int16_t)-7340,   (int16_t)23484,   (int16_t)-11194,  (int16_t)-27064,  (int16_t)-108,
    (int16_t)-20266,  (int16_t)28303,   (int16_t)-3511,   (int16_t)24605,   (int16_t)-31275,  (int16_t)-7150,   (int16_t)17111,   (int16_t)-14271,
    (int16_t)-13798,  (int16_t)17621,   (int16_t)-4531,   (int16_t)-15080,  (int16_t)-9265,   (int16_t)-24339,  (int16_t)-28877,  (int16_t)-5429,
    (int16_t)21860,   (int16_t)-29733,  (int16_t)30657,   (int16_t)1977,    (int16_t)-14211,  (int16_t)-7489,   (int16_t)30246,   (int16_t)12243,
    (int16_t)-18765,  (int16_t)7232,    (int16_t)17455,   (int16_t)5734,    (int16_t)28179,   (int16_t)-23575,  (int16_t)-30397,  (int16_t)-18733,
    (int16_t)22952,   (int16_t)7633,    (int16_t)-22685,  (int16_t)7926,    (int16_t)-30338,  (int16_t)29689,   (int16_t)-14283,  (int16_t)-17631,
    (int16_t)-5881,   (int16_t)-28590,  (int16_t)-31049,  (int16_t)19758,   (int16_t)-3082,   (int16_t)28602,   (int16_t)4149,    (int16_t)-1034,
    (int16_t)2997,    (int16_t)7422,    (int16_t)-11752,  (int16_t)15099,   (int16_t)15778,   (int16_t)-24802,  (int16_t)-5848,   (int16_t)-29983
};

const int16_t DataSet_1_16i::outputs::SUBFROMS[64] = {
    (int16_t)-11396,  (int16_t)30751,   (int16_t)-1877,   (int16_t)27552,   (int16_t)-15760,  (int16_t)31583,   (int16_t)18885,   (int16_t)-23568,
    (int16_t)7680,    (int16_t)10573,   (int16_t)5512,    (int16_t)-11931,  (int16_t)-9668,   (int16_t)29420,   (int16_t)2034,    (int16_t)-23189,
    (int16_t)-13639,  (int16_t)16300,   (int16_t)17573,   (int16_t)32083,   (int16_t)21954,   (int16_t)-4538,   (int16_t)23516,   (int16_t)-2661,
    (int16_t)7768,    (int16_t)-1155,   (int16_t)-25988,  (int16_t)4588,    (int16_t)-7508,   (int16_t)3626,    (int16_t)7300,    (int16_t)-7403,
    (int16_t)-23053,  (int16_t)15442,   (int16_t)-1006,   (int16_t)10505,   (int16_t)-27741,  (int16_t)-28063,  (int16_t)26104,   (int16_t)21904,
    (int16_t)-5219,   (int16_t)-17601,  (int16_t)6575,    (int16_t)6121,    (int16_t)-15059,  (int16_t)-14930,  (int16_t)-16242,  (int16_t)-19926,
    (int16_t)26923,   (int16_t)-29788,  (int16_t)-17668,  (int16_t)-18134,  (int16_t)16465,   (int16_t)-16370,  (int16_t)-4059,   (int16_t)10901,
    (int16_t)10960,   (int16_t)17855,   (int16_t)13207,   (int16_t)-6873,   (int16_t)-30856,  (int16_t)13721,   (int16_t)-18987,  (int16_t)20688
};

const int16_t DataSet_1_16i::outputs::MSUBFROMS[64] = {
    (int16_t)14,      (int16_t)30751,   (int16_t)-1877,   (int16_t)14,      (int16_t)-15760,  (int16_t)14,      (int16_t)14,      (int16_t)-23568,
    (int16_t)7680,    (int16_t)14,      (int16_t)14,      (int16_t)-11931,  (int16_t)14,      (int16_t)29420,   (int16_t)2034,    (int16_t)14,
    (int16_t)14,      (int16_t)16300,   (int16_t)17573,   (int16_t)14,      (int16_t)21954,   (int16_t)14,      (int16_t)14,      (int16_t)-2661,
    (int16_t)7768,    (int16_t)14,      (int16_t)14,      (int16_t)4588,    (int16_t)14,      (int16_t)3626,    (int16_t)7300,    (int16_t)14,
    (int16_t)14,      (int16_t)15442,   (int16_t)-1006,   (int16_t)14,      (int16_t)-27741,  (int16_t)14,      (int16_t)14,      (int16_t)21904,
    (int16_t)-5219,   (int16_t)14,      (int16_t)14,      (int16_t)6121,    (int16_t)14,      (int16_t)-14930,  (int16_t)-16242,  (int16_t)14,
    (int16_t)14,      (int16_t)-29788,  (int16_t)-17668,  (int16_t)14,      (int16_t)16465,   (int16_t)14,      (int16_t)14,      (int16_t)10901,
    (int16_t)10960,   (int16_t)14,      (int16_t)14,      (int16_t)-6873,   (int16_t)14,      (int16_t)13721,   (int16_t)-18987,  (int16_t)14
};

const int16_t DataSet_1_16i::outputs::POSTPREFDEC[64] = {
    (int16_t)11409,   (int16_t)-30738,  (int16_t)1890,    (int16_t)-27539,  (int16_t)15773,   (int16_t)-31570,  (int16_t)-18872,  (int16_t)23581,
    (int16_t)-7667,   (int16_t)-10560,  (int16_t)-5499,   (int16_t)11944,   (int16_t)9681,    (int16_t)-29407,  (int16_t)-2021,   (int16_t)23202,
    (int16_t)13652,   (int16_t)-16287,  (int16_t)-17560,  (int16_t)-32070,  (int16_t)-21941,  (int16_t)4551,    (int16_t)-23503,  (int16_t)2674,
    (int16_t)-7755,   (int16_t)1168,    (int16_t)26001,   (int16_t)-4575,   (int16_t)7521,    (int16_t)-3613,   (int16_t)-7287,   (int16_t)7416,
    (int16_t)23066,   (int16_t)-15429,  (int16_t)1019,    (int16_t)-10492,  (int16_t)27754,   (int16_t)28076,   (int16_t)-26091,  (int16_t)-21891,
    (int16_t)5232,    (int16_t)17614,   (int16_t)-6562,   (int16_t)-6108,   (int16_t)15072,   (int16_t)14943,   (int16_t)16255,   (int16_t)19939,
    (int16_t)-26910,  (int16_t)29801,   (int16_t)17681,   (int16_t)18147,   (int16_t)-16452,  (int16_t)16383,   (int16_t)4072,    (int16_t)-10888,
    (int16_t)-10947,  (int16_t)-17842,  (int16_t)-13194,  (int16_t)6886,    (int16_t)30869,   (int16_t)-13708,  (int16_t)19000,   (int16_t)-20675
};

const int16_t DataSet_1_16i::outputs::MPOSTPREFDEC[64] = {
    (int16_t)11410,   (int16_t)-30738,  (int16_t)1890,    (int16_t)-27538,  (int16_t)15773,   (int16_t)-31569,  (int16_t)-18871,  (int16_t)23581,
    (int16_t)-7667,   (int16_t)-10559,  (int16_t)-5498,   (int16_t)11944,   (int16_t)9682,    (int16_t)-29407,  (int16_t)-2021,   (int16_t)23203,
    (int16_t)13653,   (int16_t)-16287,  (int16_t)-17560,  (int16_t)-32069,  (int16_t)-21941,  (int16_t)4552,    (int16_t)-23502,  (int16_t)2674,
    (int16_t)-7755,   (int16_t)1169,    (int16_t)26002,   (int16_t)-4575,   (int16_t)7522,    (int16_t)-3613,   (int16_t)-7287,   (int16_t)7417,
    (int16_t)23067,   (int16_t)-15429,  (int16_t)1019,    (int16_t)-10491,  (int16_t)27754,   (int16_t)28077,   (int16_t)-26090,  (int16_t)-21891,
    (int16_t)5232,    (int16_t)17615,   (int16_t)-6561,   (int16_t)-6108,   (int16_t)15073,   (int16_t)14943,   (int16_t)16255,   (int16_t)19940,
    (int16_t)-26909,  (int16_t)29801,   (int16_t)17681,   (int16_t)18148,   (int16_t)-16452,  (int16_t)16384,   (int16_t)4073,    (int16_t)-10888,
    (int16_t)-10947,  (int16_t)-17841,  (int16_t)-13193,  (int16_t)6886,    (int16_t)30870,   (int16_t)-13708,  (int16_t)19000,   (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::MULV[64] = {
    (int16_t)31008,   (int16_t)-26467,  (int16_t)-5357,   (int16_t)15896,   (int16_t)6028,    (int16_t)13274,   (int16_t)2696,    (int16_t)-18724,
    (int16_t)20600,   (int16_t)-7217,   (int16_t)-29642,  (int16_t)-11082,  (int16_t)-28230,  (int16_t)-21272,  (int16_t)-9580,   (int16_t)23395,
    (int16_t)31906,   (int16_t)16142,   (int16_t)-29274,  (int16_t)10376,   (int16_t)-16892,  (int16_t)30248,   (int16_t)-23562,  (int16_t)-26918,
    (int16_t)1660,    (int16_t)-23797,  (int16_t)28946,   (int16_t)16662,   (int16_t)-5926,   (int16_t)-11220,  (int16_t)26848,   (int16_t)-26565,
    (int16_t)13025,   (int16_t)28944,   (int16_t)-29868,  (int16_t)6654,    (int16_t)31402,   (int16_t)-1675,   (int16_t)6594,    (int16_t)-20514,
    (int16_t)-29431,  (int16_t)-24577,  (int16_t)4029,    (int16_t)32487,   (int16_t)25534,   (int16_t)-29856,  (int16_t)25984,   (int16_t)-27036,
    (int16_t)-17611,  (int16_t)9688,    (int16_t)-32478,  (int16_t)20728,   (int16_t)14375,   (int16_t)-32768,  (int16_t)-9411,   (int16_t)22647,
    (int16_t)-22054,  (int16_t)32354,   (int16_t)-14040,  (int16_t)29422,   (int16_t)3308,    (int16_t)15919,   (int16_t)31385,   (int16_t)29054
};

const int16_t DataSet_1_16i::outputs::MMULV[64] = {
    (int16_t)11410,   (int16_t)-26467,  (int16_t)-5357,   (int16_t)-27538,  (int16_t)6028,    (int16_t)-31569,  (int16_t)-18871,  (int16_t)-18724,
    (int16_t)20600,   (int16_t)-10559,  (int16_t)-5498,   (int16_t)-11082,  (int16_t)9682,    (int16_t)-21272,  (int16_t)-9580,   (int16_t)23203,
    (int16_t)13653,   (int16_t)16142,   (int16_t)-29274,  (int16_t)-32069,  (int16_t)-16892,  (int16_t)4552,    (int16_t)-23502,  (int16_t)-26918,
    (int16_t)1660,    (int16_t)1169,    (int16_t)26002,   (int16_t)16662,   (int16_t)7522,    (int16_t)-11220,  (int16_t)26848,   (int16_t)7417,
    (int16_t)23067,   (int16_t)28944,   (int16_t)-29868,  (int16_t)-10491,  (int16_t)31402,   (int16_t)28077,   (int16_t)-26090,  (int16_t)-20514,
    (int16_t)-29431,  (int16_t)17615,   (int16_t)-6561,   (int16_t)32487,   (int16_t)15073,   (int16_t)-29856,  (int16_t)25984,   (int16_t)19940,
    (int16_t)-26909,  (int16_t)9688,    (int16_t)-32478,  (int16_t)18148,   (int16_t)14375,   (int16_t)16384,   (int16_t)4073,    (int16_t)22647,
    (int16_t)-22054,  (int16_t)-17841,  (int16_t)-13193,  (int16_t)29422,   (int16_t)30870,   (int16_t)15919,   (int16_t)31385,   (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::MULS[64] = {
    (int16_t)28668,   (int16_t)28434,   (int16_t)26474,   (int16_t)7684,    (int16_t)24228,   (int16_t)16786,   (int16_t)-2050,   (int16_t)2468,
    (int16_t)23748,   (int16_t)-16754,  (int16_t)-11436,  (int16_t)-29378,  (int16_t)4476,    (int16_t)-18468,  (int16_t)-28280,  (int16_t)-2838,
    (int16_t)-5466,   (int16_t)-31396,  (int16_t)16318,   (int16_t)9786,    (int16_t)20520,   (int16_t)-1808,   (int16_t)-1348,   (int16_t)-28086,
    (int16_t)22516,   (int16_t)16366,   (int16_t)-29188,  (int16_t)1500,    (int16_t)-25764,  (int16_t)14968,   (int16_t)29068,   (int16_t)-27234,
    (int16_t)-4742,   (int16_t)-19384,  (int16_t)14280,   (int16_t)-15802,  (int16_t)-4646,   (int16_t)-138,    (int16_t)27956,   (int16_t)21220,
    (int16_t)7726,    (int16_t)-15534,  (int16_t)-26318,  (int16_t)-19962,  (int16_t)14414,   (int16_t)12608,   (int16_t)30976,   (int16_t)17016,
    (int16_t)16490,   (int16_t)24012,   (int16_t)-14596,  (int16_t)-8072,   (int16_t)31830,   (int16_t)-32768,  (int16_t)-8514,   (int16_t)-21346,
    (int16_t)-22172,  (int16_t)12370,   (int16_t)11906,   (int16_t)30882,   (int16_t)-26572,  (int16_t)4710,    (int16_t)3870,    (int16_t)-27292
};

const int16_t DataSet_1_16i::outputs::MMULS[64] = {
    (int16_t)11410,   (int16_t)28434,   (int16_t)26474,   (int16_t)-27538,  (int16_t)24228,   (int16_t)-31569,  (int16_t)-18871,  (int16_t)2468,
    (int16_t)23748,   (int16_t)-10559,  (int16_t)-5498,   (int16_t)-29378,  (int16_t)9682,    (int16_t)-18468,  (int16_t)-28280,  (int16_t)23203,
    (int16_t)13653,   (int16_t)-31396,  (int16_t)16318,   (int16_t)-32069,  (int16_t)20520,   (int16_t)4552,    (int16_t)-23502,  (int16_t)-28086,
    (int16_t)22516,   (int16_t)1169,    (int16_t)26002,   (int16_t)1500,    (int16_t)7522,    (int16_t)14968,   (int16_t)29068,   (int16_t)7417,
    (int16_t)23067,   (int16_t)-19384,  (int16_t)14280,   (int16_t)-10491,  (int16_t)-4646,   (int16_t)28077,   (int16_t)-26090,  (int16_t)21220,
    (int16_t)7726,    (int16_t)17615,   (int16_t)-6561,   (int16_t)-19962,  (int16_t)15073,   (int16_t)12608,   (int16_t)30976,   (int16_t)19940,
    (int16_t)-26909,  (int16_t)24012,   (int16_t)-14596,  (int16_t)18148,   (int16_t)31830,   (int16_t)16384,   (int16_t)4073,    (int16_t)-21346,
    (int16_t)-22172,  (int16_t)-17841,  (int16_t)-13193,  (int16_t)30882,   (int16_t)30870,   (int16_t)4710,    (int16_t)3870,    (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::DIVV[64] = {
    (int16_t)1,       (int16_t)1,       (int16_t)0,       (int16_t)3,       (int16_t)0,       (int16_t)2,       (int16_t)0,       (int16_t)1,
    (int16_t)0,       (int16_t)0,       (int16_t)1,       (int16_t)0,       (int16_t)0,       (int16_t)-1,      (int16_t)0,       (int16_t)-1,
    (int16_t)0,       (int16_t)-12,     (int16_t)0,       (int16_t)2,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)1,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)-1,      (int16_t)1,       (int16_t)0,       (int16_t)-1,      (int16_t)-2,      (int16_t)-1,      (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)2,       (int16_t)0,       (int16_t)-3,      (int16_t)0,       (int16_t)0,       (int16_t)8,       (int16_t)-1,
    (int16_t)4,       (int16_t)24,      (int16_t)-1,      (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)1,       (int16_t)-2,      (int16_t)1,       (int16_t)0,       (int16_t)1,       (int16_t)0,       (int16_t)1,       (int16_t)0
};

const int16_t DataSet_1_16i::outputs::MDIVV[64] = {
    (int16_t)11410,   (int16_t)1,       (int16_t)0,       (int16_t)-27538,  (int16_t)0,       (int16_t)-31569,  (int16_t)-18871,  (int16_t)1,
    (int16_t)0,       (int16_t)-10559,  (int16_t)-5498,   (int16_t)0,       (int16_t)9682,    (int16_t)-1,      (int16_t)0,       (int16_t)23203,
    (int16_t)13653,   (int16_t)-12,     (int16_t)0,       (int16_t)-32069,  (int16_t)0,       (int16_t)4552,    (int16_t)-23502,  (int16_t)0,
    (int16_t)0,       (int16_t)1169,    (int16_t)26002,   (int16_t)1,       (int16_t)7522,    (int16_t)0,       (int16_t)0,       (int16_t)7417,
    (int16_t)23067,   (int16_t)1,       (int16_t)0,       (int16_t)-10491,  (int16_t)-2,      (int16_t)28077,   (int16_t)-26090,  (int16_t)0,
    (int16_t)0,       (int16_t)17615,   (int16_t)-6561,   (int16_t)-3,      (int16_t)15073,   (int16_t)0,       (int16_t)8,       (int16_t)19940,
    (int16_t)-26909,  (int16_t)24,      (int16_t)-1,      (int16_t)18148,   (int16_t)0,       (int16_t)16384,   (int16_t)4073,    (int16_t)0,
    (int16_t)1,       (int16_t)-17841,  (int16_t)-13193,  (int16_t)0,       (int16_t)30870,   (int16_t)0,       (int16_t)1,       (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::DIVS[64] = {
    (int16_t)815,     (int16_t)-2195,   (int16_t)135,     (int16_t)-1967,   (int16_t)1126,    (int16_t)-2254,   (int16_t)-1347,   (int16_t)1684,
    (int16_t)-547,    (int16_t)-754,    (int16_t)-392,    (int16_t)853,     (int16_t)691,     (int16_t)-2100,   (int16_t)-144,    (int16_t)1657,
    (int16_t)975,     (int16_t)-1163,   (int16_t)-1254,   (int16_t)-2290,   (int16_t)-1567,   (int16_t)325,     (int16_t)-1678,   (int16_t)191,
    (int16_t)-553,    (int16_t)83,      (int16_t)1857,    (int16_t)-326,    (int16_t)537,     (int16_t)-258,    (int16_t)-520,    (int16_t)529,
    (int16_t)1647,    (int16_t)-1102,   (int16_t)72,      (int16_t)-749,    (int16_t)1982,    (int16_t)2005,    (int16_t)-1863,   (int16_t)-1563,
    (int16_t)373,     (int16_t)1258,    (int16_t)-468,    (int16_t)-436,    (int16_t)1076,    (int16_t)1067,    (int16_t)1161,    (int16_t)1424,
    (int16_t)-1922,   (int16_t)2128,    (int16_t)1263,    (int16_t)1296,    (int16_t)-1175,   (int16_t)1170,    (int16_t)290,     (int16_t)-777,
    (int16_t)-781,    (int16_t)-1274,   (int16_t)-942,    (int16_t)491,     (int16_t)2205,    (int16_t)-979,    (int16_t)1357,    (int16_t)-1476
};

const int16_t DataSet_1_16i::outputs::MDIVS[64] = {
    (int16_t)11410,   (int16_t)-2195,   (int16_t)135,     (int16_t)-27538,  (int16_t)1126,    (int16_t)-31569,  (int16_t)-18871,  (int16_t)1684,
    (int16_t)-547,    (int16_t)-10559,  (int16_t)-5498,   (int16_t)853,     (int16_t)9682,    (int16_t)-2100,   (int16_t)-144,    (int16_t)23203,
    (int16_t)13653,   (int16_t)-1163,   (int16_t)-1254,   (int16_t)-32069,  (int16_t)-1567,   (int16_t)4552,    (int16_t)-23502,  (int16_t)191,
    (int16_t)-553,    (int16_t)1169,    (int16_t)26002,   (int16_t)-326,    (int16_t)7522,    (int16_t)-258,    (int16_t)-520,    (int16_t)7417,
    (int16_t)23067,   (int16_t)-1102,   (int16_t)72,      (int16_t)-10491,  (int16_t)1982,    (int16_t)28077,   (int16_t)-26090,  (int16_t)-1563,
    (int16_t)373,     (int16_t)17615,   (int16_t)-6561,   (int16_t)-436,    (int16_t)15073,   (int16_t)1067,    (int16_t)1161,    (int16_t)19940,
    (int16_t)-26909,  (int16_t)2128,    (int16_t)1263,    (int16_t)18148,   (int16_t)-1175,   (int16_t)16384,   (int16_t)4073,    (int16_t)-777,
    (int16_t)-781,    (int16_t)-17841,  (int16_t)-13193,  (int16_t)491,     (int16_t)30870,   (int16_t)-979,    (int16_t)1357,    (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::RCP[64] = {
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0
};

const int16_t DataSet_1_16i::outputs::MRCP[64] = {
    (int16_t)11410,   (int16_t)0,       (int16_t)0,       (int16_t)-27538,  (int16_t)0,       (int16_t)-31569,  (int16_t)-18871,  (int16_t)0,
    (int16_t)0,       (int16_t)-10559,  (int16_t)-5498,   (int16_t)0,       (int16_t)9682,    (int16_t)0,       (int16_t)0,       (int16_t)23203,
    (int16_t)13653,   (int16_t)0,       (int16_t)0,       (int16_t)-32069,  (int16_t)0,       (int16_t)4552,    (int16_t)-23502,  (int16_t)0,
    (int16_t)0,       (int16_t)1169,    (int16_t)26002,   (int16_t)0,       (int16_t)7522,    (int16_t)0,       (int16_t)0,       (int16_t)7417,
    (int16_t)23067,   (int16_t)0,       (int16_t)0,       (int16_t)-10491,  (int16_t)0,       (int16_t)28077,   (int16_t)-26090,  (int16_t)0,
    (int16_t)0,       (int16_t)17615,   (int16_t)-6561,   (int16_t)0,       (int16_t)15073,   (int16_t)0,       (int16_t)0,       (int16_t)19940,
    (int16_t)-26909,  (int16_t)0,       (int16_t)0,       (int16_t)18148,   (int16_t)0,       (int16_t)16384,   (int16_t)4073,    (int16_t)0,
    (int16_t)0,       (int16_t)-17841,  (int16_t)-13193,  (int16_t)0,       (int16_t)30870,   (int16_t)0,       (int16_t)0,       (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::RCPS[64] = {
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0
};

const int16_t DataSet_1_16i::outputs::MRCPS[64] = {
    (int16_t)11410,   (int16_t)0,       (int16_t)0,       (int16_t)-27538,  (int16_t)0,       (int16_t)-31569,  (int16_t)-18871,  (int16_t)0,
    (int16_t)0,       (int16_t)-10559,  (int16_t)-5498,   (int16_t)0,       (int16_t)9682,    (int16_t)0,       (int16_t)0,       (int16_t)23203,
    (int16_t)13653,   (int16_t)0,       (int16_t)0,       (int16_t)-32069,  (int16_t)0,       (int16_t)4552,    (int16_t)-23502,  (int16_t)0,
    (int16_t)0,       (int16_t)1169,    (int16_t)26002,   (int16_t)0,       (int16_t)7522,    (int16_t)0,       (int16_t)0,       (int16_t)7417,
    (int16_t)23067,   (int16_t)0,       (int16_t)0,       (int16_t)-10491,  (int16_t)0,       (int16_t)28077,   (int16_t)-26090,  (int16_t)0,
    (int16_t)0,       (int16_t)17615,   (int16_t)-6561,   (int16_t)0,       (int16_t)15073,   (int16_t)0,       (int16_t)0,       (int16_t)19940,
    (int16_t)-26909,  (int16_t)0,       (int16_t)0,       (int16_t)18148,   (int16_t)0,       (int16_t)16384,   (int16_t)4073,    (int16_t)0,
    (int16_t)0,       (int16_t)-17841,  (int16_t)-13193,  (int16_t)0,       (int16_t)30870,   (int16_t)0,       (int16_t)0,       (int16_t)-20674
};

const bool DataSet_1_16i::outputs::CMPEQV[64] = {
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false
};

const bool DataSet_1_16i::outputs::CMPEQS[64] = {
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false,
    false,  false,  false,  false,  false,  false,  false,  false
};

const bool DataSet_1_16i::outputs::CMPNEV[64] = {
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true
};

const bool DataSet_1_16i::outputs::CMPNES[64] = {
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true,
    true,   true,   true,   true,   true,   true,   true,   true
};

const bool DataSet_1_16i::outputs::CMPGTV[64] = {
    true,   false,  false,  false,  true,   false,  true,   true,
    true,   false,  false,  true,   true,   false,  false,  true,
    true,   false,  true,   false,  true,   true,   true,   true,
    false,  true,   false,  false,  true,   true,   false,  false,
    true,   false,  false,  false,  true,   true,   true,   false,
    false,  true,   true,   false,  true,   true,   true,   true,
    false,  true,   true,   false,  true,   false,  false,  true,
    false,  false,  false,  false,  true,   false,  true,   true
};

const bool DataSet_1_16i::outputs::CMPGTS[64] = {
    true,   false,  true,   false,  true,   false,  false,  true,
    false,  false,  false,  true,   true,   false,  false,  true,
    true,   false,  false,  false,  false,  true,   false,  true,
    false,  true,   true,   false,  true,   false,  false,  true,
    true,   false,  true,   false,  true,   true,   false,  false,
    true,   true,   false,  false,  true,   true,   true,   true,
    false,  true,   true,   true,   false,  true,   true,   false,
    false,  false,  false,  true,   true,   false,  true,   false
};

const bool DataSet_1_16i::outputs::CMPLTV[64] = {
    false,  true,   true,   true,   false,  true,   false,  false,
    false,  true,   true,   false,  false,  true,   true,   false,
    false,  true,   false,  true,   false,  false,  false,  false,
    true,   false,  true,   true,   false,  false,  true,   true,
    false,  true,   true,   true,   false,  false,  false,  true,
    true,   false,  false,  true,   false,  false,  false,  false,
    true,   false,  false,  true,   false,  true,   true,   false,
    true,   true,   true,   true,   false,  true,   false,  false
};

const bool DataSet_1_16i::outputs::CMPLTS[64] = {
    false,  true,   false,  true,   false,  true,   true,   false,
    true,   true,   true,   false,  false,  true,   true,   false,
    false,  true,   true,   true,   true,   false,  true,   false,
    true,   false,  false,  true,   false,  true,   true,   false,
    false,  true,   false,  true,   false,  false,  true,   true,
    false,  false,  true,   true,   false,  false,  false,  false,
    true,   false,  false,  false,  true,   false,  false,  true,
    true,   true,   true,   false,  false,  true,   false,  true
};

const bool DataSet_1_16i::outputs::CMPGEV[64] = {
    true,   false,  false,  false,  true,   false,  true,   true,
    true,   false,  false,  true,   true,   false,  false,  true,
    true,   false,  true,   false,  true,   true,   true,   true,
    false,  true,   false,  false,  true,   true,   false,  false,
    true,   false,  false,  false,  true,   true,   true,   false,
    false,  true,   true,   false,  true,   true,   true,   true,
    false,  true,   true,   false,  true,   false,  false,  true,
    false,  false,  false,  false,  true,   false,  true,   true
};

const bool DataSet_1_16i::outputs::CMPGES[64] = {
    true,   false,  true,   false,  true,   false,  false,  true,
    false,  false,  false,  true,   true,   false,  false,  true,
    true,   false,  false,  false,  false,  true,   false,  true,
    false,  true,   true,   false,  true,   false,  false,  true,
    true,   false,  true,   false,  true,   true,   false,  false,
    true,   true,   false,  false,  true,   true,   true,   true,
    false,  true,   true,   true,   false,  true,   true,   false,
    false,  false,  false,  true,   true,   false,  true,   false
};

const bool DataSet_1_16i::outputs::CMPLEV[64] = {
    false,  true,   true,   true,   false,  true,   false,  false,
    false,  true,   true,   false,  false,  true,   true,   false,
    false,  true,   false,  true,   false,  false,  false,  false,
    true,   false,  true,   true,   false,  false,  true,   true,
    false,  true,   true,   true,   false,  false,  false,  true,
    true,   false,  false,  true,   false,  false,  false,  false,
    true,   false,  false,  true,   false,  true,   true,   false,
    true,   true,   true,   true,   false,  true,   false,  false
};

const bool DataSet_1_16i::outputs::CMPLES[64] = {
    false,  true,   false,  true,   false,  true,   true,   false,
    true,   true,   true,   false,  false,  true,   true,   false,
    false,  true,   true,   true,   true,   false,  true,   false,
    true,   false,  false,  true,   false,  true,   true,   false,
    false,  true,   false,  true,   false,  false,  true,   true,
    false,  false,  true,   true,   false,  false,  false,  false,
    true,   false,  false,  false,  true,   false,  false,  true,
    true,   true,   true,   false,  false,  true,   false,  true
};

const int16_t DataSet_1_16i::outputs::HADD[64] = {
    (int16_t)11410,   (int16_t)-19327,  (int16_t)-17436,  (int16_t)20562,   (int16_t)-29200,  (int16_t)4767,    (int16_t)-14104,  (int16_t)9478,
    (int16_t)1812,    (int16_t)-8747,   (int16_t)-14245,  (int16_t)-2300,   (int16_t)7382,    (int16_t)-22024,  (int16_t)-24044,  (int16_t)-841,
    (int16_t)12812,   (int16_t)-3474,   (int16_t)-21033,  (int16_t)12434,   (int16_t)-9506,   (int16_t)-4954,   (int16_t)-28456,  (int16_t)-25781,
    (int16_t)32001,   (int16_t)-32366,  (int16_t)-6364,   (int16_t)-10938,  (int16_t)-3416,   (int16_t)-7028,   (int16_t)-14314,  (int16_t)-6897,
    (int16_t)16170,   (int16_t)742,     (int16_t)1762,    (int16_t)-8729,   (int16_t)19026,   (int16_t)-18433,  (int16_t)21013,   (int16_t)-877,
    (int16_t)4356,    (int16_t)21971,   (int16_t)15410,   (int16_t)9303,    (int16_t)24376,   (int16_t)-26216,  (int16_t)-9960,   (int16_t)9980,
    (int16_t)-16929,  (int16_t)12873,   (int16_t)30555,   (int16_t)-16833,  (int16_t)32252,   (int16_t)-16900,  (int16_t)-12827,  (int16_t)-23714,
    (int16_t)30876,   (int16_t)13035,   (int16_t)-158,    (int16_t)6729,    (int16_t)-27937,  (int16_t)23892,   (int16_t)-22643,  (int16_t)22219
};

const int16_t DataSet_1_16i::outputs::MHADD[64] = {
    (int16_t)0,       (int16_t)-30737,  (int16_t)-28846,  (int16_t)-28846,  (int16_t)-13072,  (int16_t)-13072,  (int16_t)-13072,  (int16_t)10510,
    (int16_t)2844,    (int16_t)2844,    (int16_t)2844,    (int16_t)14789,   (int16_t)14789,   (int16_t)-14617,  (int16_t)-16637,  (int16_t)-16637,
    (int16_t)-16637,  (int16_t)32613,   (int16_t)15054,   (int16_t)15054,   (int16_t)-6886,   (int16_t)-6886,   (int16_t)-6886,   (int16_t)-4211,
    (int16_t)-11965,  (int16_t)-11965,  (int16_t)-11965,  (int16_t)-16539,  (int16_t)-16539,  (int16_t)-20151,  (int16_t)-27437,  (int16_t)-27437,
    (int16_t)-27437,  (int16_t)22671,   (int16_t)23691,   (int16_t)23691,   (int16_t)-14090,  (int16_t)-14090,  (int16_t)-14090,  (int16_t)29556,
    (int16_t)-30747,  (int16_t)-30747,  (int16_t)-30747,  (int16_t)28682,   (int16_t)28682,   (int16_t)-21910,  (int16_t)-5654,   (int16_t)-5654,
    (int16_t)-5654,   (int16_t)24148,   (int16_t)-23706,  (int16_t)-23706,  (int16_t)25379,   (int16_t)25379,   (int16_t)25379,   (int16_t)14492,
    (int16_t)3546,    (int16_t)3546,    (int16_t)3546,    (int16_t)10433,   (int16_t)10433,   (int16_t)-3274,   (int16_t)15727,   (int16_t)15727
};

const int16_t DataSet_1_16i::outputs::HMUL[64] = {
    (int16_t)11410,   (int16_t)-26034,  (int16_t)-12758,  (int16_t)-8692,   (int16_t)-6296,   (int16_t)-12264,  (int16_t)26328,   (int16_t)-21168,
    (int16_t)6752,    (int16_t)8800,    (int16_t)-16832,  (int16_t)6208,    (int16_t)9344,    (int16_t)22784,   (int16_t)-17408,  (int16_t)-19456,
    (int16_t)-15360,  (int16_t)2048,    (int16_t)18432,   (int16_t)-26624,  (int16_t)8192,    (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0
};

const int16_t DataSet_1_16i::outputs::MHMUL[64] = {
    (int16_t)1,       (int16_t)-30737,  (int16_t)6765,    (int16_t)6765,    (int16_t)18502,   (int16_t)18502,   (int16_t)18502,   (int16_t)-24524,
    (int16_t)-21800,  (int16_t)-21800,  (int16_t)-21800,  (int16_t)-26472,  (int16_t)-26472,  (int16_t)-976,    (int16_t)5440,    (int16_t)5440,
    (int16_t)5440,    (int16_t)8832,    (int16_t)-22912,  (int16_t)-22912,  (int16_t)28160,   (int16_t)28160,   (int16_t)28160,   (int16_t)27136,
    (int16_t)23552,   (int16_t)23552,   (int16_t)23552,   (int16_t)14336,   (int16_t)14336,   (int16_t)-8192,   (int16_t)-16384,  (int16_t)-16384,
    (int16_t)-16384,  (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0
};

const int16_t DataSet_1_16i::outputs::FMULADDV[64] = {
    (int16_t)-1062,   (int16_t)-26393,  (int16_t)13617,   (int16_t)-30297,  (int16_t)-29296,  (int16_t)-19792,  (int16_t)-25184,  (int16_t)-13774,
    (int16_t)2423,    (int16_t)5490,    (int16_t)28398,   (int16_t)-15309,  (int16_t)-25753,  (int16_t)15197,   (int16_t)-3021,   (int16_t)19324,
    (int16_t)-11420,  (int16_t)-28008,  (int16_t)11286,   (int16_t)-3787,   (int16_t)-609,    (int16_t)-10948,  (int16_t)-30907,  (int16_t)-29642,
    (int16_t)-28480,  (int16_t)-6681,   (int16_t)-8447,   (int16_t)-8449,   (int16_t)-30754,  (int16_t)-12083,  (int16_t)27209,   (int16_t)-9874,
    (int16_t)-9281,   (int16_t)13001,   (int16_t)27266,   (int16_t)-17497,  (int16_t)-28313,  (int16_t)16165,   (int16_t)-25828,  (int16_t)-28878,
    (int16_t)21005,   (int16_t)9911,    (int16_t)-23781,  (int16_t)-12280,  (int16_t)-7166,   (int16_t)-24287,  (int16_t)5222,    (int16_t)6195,
    (int16_t)6092,    (int16_t)-16032,  (int16_t)27877,   (int16_t)-14329,  (int16_t)15583,   (int16_t)10917,   (int16_t)-13952,  (int16_t)24528,
    (int16_t)14048,   (int16_t)-31794,  (int16_t)-24569,  (int16_t)531,     (int16_t)-17094,  (int16_t)26349,   (int16_t)-20171,  (int16_t)4370
};

const int16_t DataSet_1_16i::outputs::MFMULADDV[64] = {
    (int16_t)11410,   (int16_t)-26393,  (int16_t)13617,   (int16_t)-27538,  (int16_t)-29296,  (int16_t)-31569,  (int16_t)-18871,  (int16_t)-13774,
    (int16_t)2423,    (int16_t)-10559,  (int16_t)-5498,   (int16_t)-15309,  (int16_t)9682,    (int16_t)15197,   (int16_t)-3021,   (int16_t)23203,
    (int16_t)13653,   (int16_t)-28008,  (int16_t)11286,   (int16_t)-32069,  (int16_t)-609,    (int16_t)4552,    (int16_t)-23502,  (int16_t)-29642,
    (int16_t)-28480,  (int16_t)1169,    (int16_t)26002,   (int16_t)-8449,   (int16_t)7522,    (int16_t)-12083,  (int16_t)27209,   (int16_t)7417,
    (int16_t)23067,   (int16_t)13001,   (int16_t)27266,   (int16_t)-10491,  (int16_t)-28313,  (int16_t)28077,   (int16_t)-26090,  (int16_t)-28878,
    (int16_t)21005,   (int16_t)17615,   (int16_t)-6561,   (int16_t)-12280,  (int16_t)15073,   (int16_t)-24287,  (int16_t)5222,    (int16_t)19940,
    (int16_t)-26909,  (int16_t)-16032,  (int16_t)27877,   (int16_t)18148,   (int16_t)15583,   (int16_t)16384,   (int16_t)4073,    (int16_t)24528,
    (int16_t)14048,   (int16_t)-17841,  (int16_t)-13193,  (int16_t)531,     (int16_t)30870,   (int16_t)26349,   (int16_t)-20171,  (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::FMULSUBV[64] = {
    (int16_t)-2458,   (int16_t)-26541,  (int16_t)-24331,  (int16_t)-3447,   (int16_t)-24184,  (int16_t)-19196,  (int16_t)30576,   (int16_t)-23674,
    (int16_t)-26759,  (int16_t)-19924,  (int16_t)-22146,  (int16_t)-6855,   (int16_t)-30707,  (int16_t)7795,    (int16_t)-16139,  (int16_t)27466,
    (int16_t)9696,    (int16_t)-5244,   (int16_t)-4298,   (int16_t)24539,   (int16_t)32361,   (int16_t)5908,    (int16_t)-16217,  (int16_t)-24194,
    (int16_t)31800,   (int16_t)24623,   (int16_t)803,     (int16_t)-23763,  (int16_t)18902,   (int16_t)-10357,  (int16_t)26487,   (int16_t)22280,
    (int16_t)-30205,  (int16_t)-20649,  (int16_t)-21466,  (int16_t)30805,   (int16_t)25581,   (int16_t)-19515,  (int16_t)-26520,  (int16_t)-12150,
    (int16_t)-14331,  (int16_t)6471,    (int16_t)31839,   (int16_t)11718,   (int16_t)-7302,   (int16_t)30111,   (int16_t)-18790,  (int16_t)5269,
    (int16_t)24222,   (int16_t)-30128,  (int16_t)-27297,  (int16_t)-9751,   (int16_t)13167,   (int16_t)-10917,  (int16_t)-4870,   (int16_t)20766,
    (int16_t)7380,    (int16_t)30966,   (int16_t)-3511,   (int16_t)-7223,   (int16_t)23710,   (int16_t)5489,    (int16_t)17405,   (int16_t)-11798
};

const int16_t DataSet_1_16i::outputs::MFMULSUBV[64] = {
    (int16_t)11410,   (int16_t)-26541,  (int16_t)-24331,  (int16_t)-27538,  (int16_t)-24184,  (int16_t)-31569,  (int16_t)-18871,  (int16_t)-23674,
    (int16_t)-26759,  (int16_t)-10559,  (int16_t)-5498,   (int16_t)-6855,   (int16_t)9682,    (int16_t)7795,    (int16_t)-16139,  (int16_t)23203,
    (int16_t)13653,   (int16_t)-5244,   (int16_t)-4298,   (int16_t)-32069,  (int16_t)32361,   (int16_t)4552,    (int16_t)-23502,  (int16_t)-24194,
    (int16_t)31800,   (int16_t)1169,    (int16_t)26002,   (int16_t)-23763,  (int16_t)7522,    (int16_t)-10357,  (int16_t)26487,   (int16_t)7417,
    (int16_t)23067,   (int16_t)-20649,  (int16_t)-21466,  (int16_t)-10491,  (int16_t)25581,   (int16_t)28077,   (int16_t)-26090,  (int16_t)-12150,
    (int16_t)-14331,  (int16_t)17615,   (int16_t)-6561,   (int16_t)11718,   (int16_t)15073,   (int16_t)30111,   (int16_t)-18790,  (int16_t)19940,
    (int16_t)-26909,  (int16_t)-30128,  (int16_t)-27297,  (int16_t)18148,   (int16_t)13167,   (int16_t)16384,   (int16_t)4073,    (int16_t)20766,
    (int16_t)7380,    (int16_t)-17841,  (int16_t)-13193,  (int16_t)-7223,   (int16_t)30870,   (int16_t)5489,    (int16_t)17405,   (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::FADDMULV[64] = {
    (int16_t)8628,    (int16_t)-20012,  (int16_t)-12648,  (int16_t)-17570,  (int16_t)-21536,  (int16_t)-3378,   (int16_t)28824,   (int16_t)12256,
    (int16_t)27918,   (int16_t)29168,   (int16_t)29384,   (int16_t)8243,    (int16_t)-8485,   (int16_t)-3762,   (int16_t)11601,   (int16_t)10308,
    (int16_t)-9186,   (int16_t)8058,    (int16_t)24464,   (int16_t)24983,   (int16_t)-22691,  (int16_t)8484,    (int16_t)27435,   (int16_t)18588,
    (int16_t)-18624,  (int16_t)-2864,   (int16_t)-2179,   (int16_t)-21947,  (int16_t)6268,    (int16_t)-16665,  (int16_t)22218,   (int16_t)6308,
    (int16_t)-15708,  (int16_t)2040,    (int16_t)-22526,  (int16_t)1699,    (int16_t)24581,   (int16_t)-31456,  (int16_t)17994,   (int16_t)12524,
    (int16_t)15400,   (int16_t)-23808,  (int16_t)29500,   (int16_t)5952,    (int16_t)-22212,  (int16_t)-24455,  (int16_t)-98,     (int16_t)-12277,
    (int16_t)-29946,  (int16_t)24112,   (int16_t)-8239,   (int16_t)2830,    (int16_t)-18304,  (int16_t)-14622,  (int16_t)19418,   (int16_t)24232,
    (int16_t)16934,   (int16_t)21884,   (int16_t)-22383,  (int16_t)-27635,  (int16_t)1296,    (int16_t)-8720,   (int16_t)1496,    (int16_t)-9492
};

const int16_t DataSet_1_16i::outputs::MFADDMULV[64] = {
    (int16_t)11410,   (int16_t)-20012,  (int16_t)-12648,  (int16_t)-27538,  (int16_t)-21536,  (int16_t)-31569,  (int16_t)-18871,  (int16_t)12256,
    (int16_t)27918,   (int16_t)-10559,  (int16_t)-5498,   (int16_t)8243,    (int16_t)9682,    (int16_t)-3762,   (int16_t)11601,   (int16_t)23203,
    (int16_t)13653,   (int16_t)8058,    (int16_t)24464,   (int16_t)-32069,  (int16_t)-22691,  (int16_t)4552,    (int16_t)-23502,  (int16_t)18588,
    (int16_t)-18624,  (int16_t)1169,    (int16_t)26002,   (int16_t)-21947,  (int16_t)7522,    (int16_t)-16665,  (int16_t)22218,   (int16_t)7417,
    (int16_t)23067,   (int16_t)2040,    (int16_t)-22526,  (int16_t)-10491,  (int16_t)24581,   (int16_t)28077,   (int16_t)-26090,  (int16_t)12524,
    (int16_t)15400,   (int16_t)17615,   (int16_t)-6561,   (int16_t)5952,    (int16_t)15073,   (int16_t)-24455,  (int16_t)-98,     (int16_t)19940,
    (int16_t)-26909,  (int16_t)24112,   (int16_t)-8239,   (int16_t)18148,   (int16_t)-18304,  (int16_t)16384,   (int16_t)4073,    (int16_t)24232,
    (int16_t)16934,   (int16_t)-17841,  (int16_t)-13193,  (int16_t)-27635,  (int16_t)30870,   (int16_t)-8720,   (int16_t)1496,    (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::FSUBMULV[64] = {
    (int16_t)-5516,   (int16_t)-7080,   (int16_t)10396,   (int16_t)-29818,  (int16_t)-5872,   (int16_t)9670,    (int16_t)-27880,  (int16_t)10312,
    (int16_t)2774,    (int16_t)-5674,   (int16_t)17880,   (int16_t)-297,    (int16_t)761,     (int16_t)-14394,  (int16_t)32119,   (int16_t)11154,
    (int16_t)5302,    (int16_t)-10706,  (int16_t)14416,   (int16_t)32551,   (int16_t)-1877,   (int16_t)5660,    (int16_t)-26703,  (int16_t)22540,
    (int16_t)26992,   (int16_t)-22424,  (int16_t)799,     (int16_t)-31841,  (int16_t)-29036,  (int16_t)25057,   (int16_t)25706,   (int16_t)-7022,
    (int16_t)-3024,   (int16_t)21952,   (int16_t)-12658,  (int16_t)10231,   (int16_t)6649,    (int16_t)-30016,  (int16_t)15662,   (int16_t)13764,
    (int16_t)20832,   (int16_t)-1392,   (int16_t)-11128,  (int16_t)11338,   (int16_t)-25012,  (int16_t)9287,    (int16_t)6754,    (int16_t)-4435,
    (int16_t)-25404,  (int16_t)-20880,  (int16_t)26011,   (int16_t)15274,   (int16_t)-12496,  (int16_t)-18146,  (int16_t)17436,   (int16_t)-21126,
    (int16_t)2242,    (int16_t)-3284,   (int16_t)-32063,  (int16_t)17593,   (int16_t)-18856,  (int16_t)14268,   (int16_t)31648,   (int16_t)-14140
};

const int16_t DataSet_1_16i::outputs::MFSUBMULV[64] = {
    (int16_t)11410,   (int16_t)-7080,   (int16_t)10396,   (int16_t)-27538,  (int16_t)-5872,   (int16_t)-31569,  (int16_t)-18871,  (int16_t)10312,
    (int16_t)2774,    (int16_t)-10559,  (int16_t)-5498,   (int16_t)-297,    (int16_t)9682,    (int16_t)-14394,  (int16_t)32119,   (int16_t)23203,
    (int16_t)13653,   (int16_t)-10706,  (int16_t)14416,   (int16_t)-32069,  (int16_t)-1877,   (int16_t)4552,    (int16_t)-23502,  (int16_t)22540,
    (int16_t)26992,   (int16_t)1169,    (int16_t)26002,   (int16_t)-31841,  (int16_t)7522,    (int16_t)25057,   (int16_t)25706,   (int16_t)7417,
    (int16_t)23067,   (int16_t)21952,   (int16_t)-12658,  (int16_t)-10491,  (int16_t)6649,    (int16_t)28077,   (int16_t)-26090,  (int16_t)13764,
    (int16_t)20832,   (int16_t)17615,   (int16_t)-6561,   (int16_t)11338,   (int16_t)15073,   (int16_t)9287,    (int16_t)6754,    (int16_t)19940,
    (int16_t)-26909,  (int16_t)-20880,  (int16_t)26011,   (int16_t)18148,   (int16_t)-12496,  (int16_t)16384,   (int16_t)4073,    (int16_t)-21126,
    (int16_t)2242,    (int16_t)-17841,  (int16_t)-13193,  (int16_t)17593,   (int16_t)30870,   (int16_t)14268,   (int16_t)31648,   (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::MAXV[64] = {
    (int16_t)11410,   (int16_t)-17357,  (int16_t)5265,    (int16_t)-7340,   (int16_t)15774,   (int16_t)-11194,  (int16_t)-18871,  (int16_t)23582,
    (int16_t)-7666,   (int16_t)28303,   (int16_t)-3511,   (int16_t)11945,   (int16_t)9682,    (int16_t)28980,   (int16_t)15091,   (int16_t)23203,
    (int16_t)13653,   (int16_t)1335,    (int16_t)-17559,  (int16_t)-15080,  (int16_t)-21940,  (int16_t)4552,    (int16_t)-23502,  (int16_t)2675,
    (int16_t)14106,   (int16_t)1169,    (int16_t)30657,   (int16_t)-2597,   (int16_t)7522,    (int16_t)-3612,   (int16_t)22960,   (int16_t)12243,
    (int16_t)23067,   (int16_t)-8196,   (int16_t)18475,   (int16_t)5734,    (int16_t)27755,   (int16_t)28077,   (int16_t)-26090,  (int16_t)24913,
    (int16_t)28185,   (int16_t)17615,   (int16_t)-6561,   (int16_t)1819,    (int16_t)15073,   (int16_t)14944,   (int16_t)16256,   (int16_t)19940,
    (int16_t)-5881,   (int16_t)29802,   (int16_t)17682,   (int16_t)19758,   (int16_t)-16451,  (int16_t)28602,   (int16_t)4149,    (int16_t)-10887,
    (int16_t)-7949,   (int16_t)7422,    (int16_t)-11752,  (int16_t)21986,   (int16_t)30870,   (int16_t)27027,   (int16_t)19001,   (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::MMAXV[64] = {
    (int16_t)11410,   (int16_t)-17357,  (int16_t)5265,    (int16_t)-27538,  (int16_t)15774,   (int16_t)-31569,  (int16_t)-18871,  (int16_t)23582,
    (int16_t)-7666,   (int16_t)-10559,  (int16_t)-5498,   (int16_t)11945,   (int16_t)9682,    (int16_t)28980,   (int16_t)15091,   (int16_t)23203,
    (int16_t)13653,   (int16_t)1335,    (int16_t)-17559,  (int16_t)-32069,  (int16_t)-21940,  (int16_t)4552,    (int16_t)-23502,  (int16_t)2675,
    (int16_t)14106,   (int16_t)1169,    (int16_t)26002,   (int16_t)-2597,   (int16_t)7522,    (int16_t)-3612,   (int16_t)22960,   (int16_t)7417,
    (int16_t)23067,   (int16_t)-8196,   (int16_t)18475,   (int16_t)-10491,  (int16_t)27755,   (int16_t)28077,   (int16_t)-26090,  (int16_t)24913,
    (int16_t)28185,   (int16_t)17615,   (int16_t)-6561,   (int16_t)1819,    (int16_t)15073,   (int16_t)14944,   (int16_t)16256,   (int16_t)19940,
    (int16_t)-26909,  (int16_t)29802,   (int16_t)17682,   (int16_t)18148,   (int16_t)-16451,  (int16_t)16384,   (int16_t)4073,    (int16_t)-10887,
    (int16_t)-7949,   (int16_t)-17841,  (int16_t)-13193,  (int16_t)21986,   (int16_t)30870,   (int16_t)27027,   (int16_t)19001,   (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::MAXS[64] = {
    (int16_t)11410,   (int16_t)14,      (int16_t)1891,    (int16_t)14,      (int16_t)15774,   (int16_t)14,      (int16_t)14,      (int16_t)23582,
    (int16_t)14,      (int16_t)14,      (int16_t)14,      (int16_t)11945,   (int16_t)9682,    (int16_t)14,      (int16_t)14,      (int16_t)23203,
    (int16_t)13653,   (int16_t)14,      (int16_t)14,      (int16_t)14,      (int16_t)14,      (int16_t)4552,    (int16_t)14,      (int16_t)2675,
    (int16_t)14,      (int16_t)1169,    (int16_t)26002,   (int16_t)14,      (int16_t)7522,    (int16_t)14,      (int16_t)14,      (int16_t)7417,
    (int16_t)23067,   (int16_t)14,      (int16_t)1020,    (int16_t)14,      (int16_t)27755,   (int16_t)28077,   (int16_t)14,      (int16_t)14,
    (int16_t)5233,    (int16_t)17615,   (int16_t)14,      (int16_t)14,      (int16_t)15073,   (int16_t)14944,   (int16_t)16256,   (int16_t)19940,
    (int16_t)14,      (int16_t)29802,   (int16_t)17682,   (int16_t)18148,   (int16_t)14,      (int16_t)16384,   (int16_t)4073,    (int16_t)14,
    (int16_t)14,      (int16_t)14,      (int16_t)14,      (int16_t)6887,    (int16_t)30870,   (int16_t)14,      (int16_t)19001,   (int16_t)14
};

const int16_t DataSet_1_16i::outputs::MMAXS[64] = {
    (int16_t)11410,   (int16_t)14,      (int16_t)1891,    (int16_t)-27538,  (int16_t)15774,   (int16_t)-31569,  (int16_t)-18871,  (int16_t)23582,
    (int16_t)14,      (int16_t)-10559,  (int16_t)-5498,   (int16_t)11945,   (int16_t)9682,    (int16_t)14,      (int16_t)14,      (int16_t)23203,
    (int16_t)13653,   (int16_t)14,      (int16_t)14,      (int16_t)-32069,  (int16_t)14,      (int16_t)4552,    (int16_t)-23502,  (int16_t)2675,
    (int16_t)14,      (int16_t)1169,    (int16_t)26002,   (int16_t)14,      (int16_t)7522,    (int16_t)14,      (int16_t)14,      (int16_t)7417,
    (int16_t)23067,   (int16_t)14,      (int16_t)1020,    (int16_t)-10491,  (int16_t)27755,   (int16_t)28077,   (int16_t)-26090,  (int16_t)14,
    (int16_t)5233,    (int16_t)17615,   (int16_t)-6561,   (int16_t)14,      (int16_t)15073,   (int16_t)14944,   (int16_t)16256,   (int16_t)19940,
    (int16_t)-26909,  (int16_t)29802,   (int16_t)17682,   (int16_t)18148,   (int16_t)14,      (int16_t)16384,   (int16_t)4073,    (int16_t)14,
    (int16_t)14,      (int16_t)-17841,  (int16_t)-13193,  (int16_t)6887,    (int16_t)30870,   (int16_t)14,      (int16_t)19001,   (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::MINV[64] = {
    (int16_t)6160,    (int16_t)-30737,  (int16_t)1891,    (int16_t)-27538,  (int16_t)-26278,  (int16_t)-31569,  (int16_t)-27064,  (int16_t)23474,
    (int16_t)-27932,  (int16_t)-10559,  (int16_t)-5498,   (int16_t)-28986,  (int16_t)-31275,  (int16_t)-29406,  (int16_t)-2020,   (int16_t)-14271,
    (int16_t)-13798,  (int16_t)-16286,  (int16_t)-22090,  (int16_t)-32069,  (int16_t)-31205,  (int16_t)-24339,  (int16_t)-28877,  (int16_t)-2754,
    (int16_t)-7754,   (int16_t)-29733,  (int16_t)26002,   (int16_t)-4574,   (int16_t)-14211,  (int16_t)-11101,  (int16_t)-7286,   (int16_t)7417,
    (int16_t)-18765,  (int16_t)-15428,  (int16_t)1020,    (int16_t)-10491,  (int16_t)-9602,   (int16_t)-23575,  (int16_t)-30397,  (int16_t)-21890,
    (int16_t)5233,    (int16_t)7633,    (int16_t)-22685,  (int16_t)-6107,   (int16_t)-30338,  (int16_t)-20903,  (int16_t)1973,    (int16_t)-17631,
    (int16_t)-26909,  (int16_t)1212,    (int16_t)-13367,  (int16_t)18148,   (int16_t)-19533,  (int16_t)16384,   (int16_t)4073,    (int16_t)-11921,
    (int16_t)-10946,  (int16_t)-17841,  (int16_t)-13193,  (int16_t)6887,    (int16_t)15778,   (int16_t)-13707,  (int16_t)13153,   (int16_t)-29983
};

const int16_t DataSet_1_16i::outputs::MMINV[64] = {
    (int16_t)11410,   (int16_t)-30737,  (int16_t)1891,    (int16_t)-27538,  (int16_t)-26278,  (int16_t)-31569,  (int16_t)-18871,  (int16_t)23474,
    (int16_t)-27932,  (int16_t)-10559,  (int16_t)-5498,   (int16_t)-28986,  (int16_t)9682,    (int16_t)-29406,  (int16_t)-2020,   (int16_t)23203,
    (int16_t)13653,   (int16_t)-16286,  (int16_t)-22090,  (int16_t)-32069,  (int16_t)-31205,  (int16_t)4552,    (int16_t)-23502,  (int16_t)-2754,
    (int16_t)-7754,   (int16_t)1169,    (int16_t)26002,   (int16_t)-4574,   (int16_t)7522,    (int16_t)-11101,  (int16_t)-7286,   (int16_t)7417,
    (int16_t)23067,   (int16_t)-15428,  (int16_t)1020,    (int16_t)-10491,  (int16_t)-9602,   (int16_t)28077,   (int16_t)-26090,  (int16_t)-21890,
    (int16_t)5233,    (int16_t)17615,   (int16_t)-6561,   (int16_t)-6107,   (int16_t)15073,   (int16_t)-20903,  (int16_t)1973,    (int16_t)19940,
    (int16_t)-26909,  (int16_t)1212,    (int16_t)-13367,  (int16_t)18148,   (int16_t)-19533,  (int16_t)16384,   (int16_t)4073,    (int16_t)-11921,
    (int16_t)-10946,  (int16_t)-17841,  (int16_t)-13193,  (int16_t)6887,    (int16_t)30870,   (int16_t)-13707,  (int16_t)13153,   (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::MINS[64] = {
    (int16_t)14,      (int16_t)-30737,  (int16_t)14,      (int16_t)-27538,  (int16_t)14,      (int16_t)-31569,  (int16_t)-18871,  (int16_t)14,
    (int16_t)-7666,   (int16_t)-10559,  (int16_t)-5498,   (int16_t)14,      (int16_t)14,      (int16_t)-29406,  (int16_t)-2020,   (int16_t)14,
    (int16_t)14,      (int16_t)-16286,  (int16_t)-17559,  (int16_t)-32069,  (int16_t)-21940,  (int16_t)14,      (int16_t)-23502,  (int16_t)14,
    (int16_t)-7754,   (int16_t)14,      (int16_t)14,      (int16_t)-4574,   (int16_t)14,      (int16_t)-3612,   (int16_t)-7286,   (int16_t)14,
    (int16_t)14,      (int16_t)-15428,  (int16_t)14,      (int16_t)-10491,  (int16_t)14,      (int16_t)14,      (int16_t)-26090,  (int16_t)-21890,
    (int16_t)14,      (int16_t)14,      (int16_t)-6561,   (int16_t)-6107,   (int16_t)14,      (int16_t)14,      (int16_t)14,      (int16_t)14,
    (int16_t)-26909,  (int16_t)14,      (int16_t)14,      (int16_t)14,      (int16_t)-16451,  (int16_t)14,      (int16_t)14,      (int16_t)-10887,
    (int16_t)-10946,  (int16_t)-17841,  (int16_t)-13193,  (int16_t)14,      (int16_t)14,      (int16_t)-13707,  (int16_t)14,      (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::MMINS[64] = {
    (int16_t)11410,   (int16_t)-30737,  (int16_t)14,      (int16_t)-27538,  (int16_t)14,      (int16_t)-31569,  (int16_t)-18871,  (int16_t)14,
    (int16_t)-7666,   (int16_t)-10559,  (int16_t)-5498,   (int16_t)14,      (int16_t)9682,    (int16_t)-29406,  (int16_t)-2020,   (int16_t)23203,
    (int16_t)13653,   (int16_t)-16286,  (int16_t)-17559,  (int16_t)-32069,  (int16_t)-21940,  (int16_t)4552,    (int16_t)-23502,  (int16_t)14,
    (int16_t)-7754,   (int16_t)1169,    (int16_t)26002,   (int16_t)-4574,   (int16_t)7522,    (int16_t)-3612,   (int16_t)-7286,   (int16_t)7417,
    (int16_t)23067,   (int16_t)-15428,  (int16_t)14,      (int16_t)-10491,  (int16_t)14,      (int16_t)28077,   (int16_t)-26090,  (int16_t)-21890,
    (int16_t)14,      (int16_t)17615,   (int16_t)-6561,   (int16_t)-6107,   (int16_t)15073,   (int16_t)14,      (int16_t)14,      (int16_t)19940,
    (int16_t)-26909,  (int16_t)14,      (int16_t)14,      (int16_t)18148,   (int16_t)-16451,  (int16_t)16384,   (int16_t)4073,    (int16_t)-10887,
    (int16_t)-10946,  (int16_t)-17841,  (int16_t)-13193,  (int16_t)14,      (int16_t)30870,   (int16_t)-13707,  (int16_t)14,      (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::HMAX[64] = {
    (int16_t)11410,   (int16_t)11410,   (int16_t)11410,   (int16_t)11410,   (int16_t)15774,   (int16_t)15774,   (int16_t)15774,   (int16_t)23582,
    (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,
    (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,
    (int16_t)23582,   (int16_t)23582,   (int16_t)26002,   (int16_t)26002,   (int16_t)26002,   (int16_t)26002,   (int16_t)26002,   (int16_t)26002,
    (int16_t)26002,   (int16_t)26002,   (int16_t)26002,   (int16_t)26002,   (int16_t)27755,   (int16_t)28077,   (int16_t)28077,   (int16_t)28077,
    (int16_t)28077,   (int16_t)28077,   (int16_t)28077,   (int16_t)28077,   (int16_t)28077,   (int16_t)28077,   (int16_t)28077,   (int16_t)28077,
    (int16_t)28077,   (int16_t)29802,   (int16_t)29802,   (int16_t)29802,   (int16_t)29802,   (int16_t)29802,   (int16_t)29802,   (int16_t)29802,
    (int16_t)29802,   (int16_t)29802,   (int16_t)29802,   (int16_t)29802,   (int16_t)30870,   (int16_t)30870,   (int16_t)30870,   (int16_t)30870
};

const int16_t DataSet_1_16i::outputs::MHMAX[64] = {
    (int16_t)0,       (int16_t)0,       (int16_t)1891,    (int16_t)1891,    (int16_t)15774,   (int16_t)15774,   (int16_t)15774,   (int16_t)23582,
    (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,
    (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,
    (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,
    (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)23582,   (int16_t)27755,   (int16_t)27755,   (int16_t)27755,   (int16_t)27755,
    (int16_t)27755,   (int16_t)27755,   (int16_t)27755,   (int16_t)27755,   (int16_t)27755,   (int16_t)27755,   (int16_t)27755,   (int16_t)27755,
    (int16_t)27755,   (int16_t)29802,   (int16_t)29802,   (int16_t)29802,   (int16_t)29802,   (int16_t)29802,   (int16_t)29802,   (int16_t)29802,
    (int16_t)29802,   (int16_t)29802,   (int16_t)29802,   (int16_t)29802,   (int16_t)29802,   (int16_t)29802,   (int16_t)29802,   (int16_t)29802
};

const int16_t DataSet_1_16i::outputs::HMIN[64] = {
    (int16_t)11410,   (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-31569,  (int16_t)-31569,  (int16_t)-31569,
    (int16_t)-31569,  (int16_t)-31569,  (int16_t)-31569,  (int16_t)-31569,  (int16_t)-31569,  (int16_t)-31569,  (int16_t)-31569,  (int16_t)-31569,
    (int16_t)-31569,  (int16_t)-31569,  (int16_t)-31569,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,
    (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,
    (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,
    (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,
    (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,
    (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069,  (int16_t)-32069
};

const int16_t DataSet_1_16i::outputs::MHMIN[64] = {
    (int16_t)32767,   (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,
    (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,
    (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,
    (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,
    (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,
    (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,
    (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,
    (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737
};

const int16_t DataSet_1_16i::outputs::BANDV[64] = {
    (int16_t)2064,    (int16_t)-31709,  (int16_t)1025,    (int16_t)-32700,  (int16_t)6426,    (int16_t)-31738,  (int16_t)-27064,  (int16_t)22546,
    (int16_t)-32252,  (int16_t)18049,   (int16_t)-7680,   (int16_t)3712,    (int16_t)1488,    (int16_t)288,     (int16_t)14352,   (int16_t)18433,
    (int16_t)16,      (int16_t)34,      (int16_t)-22240,  (int16_t)-32744,  (int16_t)-32248,  (int16_t)200,     (int16_t)-31694,  (int16_t)50,
    (int16_t)8466,    (int16_t)145,     (int16_t)25984,   (int16_t)-7166,   (int16_t)2144,    (int16_t)-12128,  (int16_t)16768,   (int16_t)3281,
    (int16_t)4627,    (int16_t)-15428,  (int16_t)40,      (int16_t)5636,    (int16_t)18538,   (int16_t)8617,    (int16_t)-30718,  (int16_t)8272,
    (int16_t)1041,    (int16_t)1217,    (int16_t)-22973,  (int16_t)1,       (int16_t)2144,    (int16_t)10816,   (int16_t)1920,    (int16_t)2336,
    (int16_t)-32765,  (int16_t)1064,    (int16_t)16640,   (int16_t)17444,   (int16_t)-19535,  (int16_t)16384,   (int16_t)33,      (int16_t)-11927,
    (int16_t)-16334,  (int16_t)6222,    (int16_t)-16368,  (int16_t)4322,    (int16_t)14466,   (int16_t)18449,   (int16_t)545,     (int16_t)-30176
};

const int16_t DataSet_1_16i::outputs::MBANDV[64] = {
    (int16_t)11410,   (int16_t)-31709,  (int16_t)1025,    (int16_t)-27538,  (int16_t)6426,    (int16_t)-31569,  (int16_t)-18871,  (int16_t)22546,
    (int16_t)-32252,  (int16_t)-10559,  (int16_t)-5498,   (int16_t)3712,    (int16_t)9682,    (int16_t)288,     (int16_t)14352,   (int16_t)23203,
    (int16_t)13653,   (int16_t)34,      (int16_t)-22240,  (int16_t)-32069,  (int16_t)-32248,  (int16_t)4552,    (int16_t)-23502,  (int16_t)50,
    (int16_t)8466,    (int16_t)1169,    (int16_t)26002,   (int16_t)-7166,   (int16_t)7522,    (int16_t)-12128,  (int16_t)16768,   (int16_t)7417,
    (int16_t)23067,   (int16_t)-15428,  (int16_t)40,      (int16_t)-10491,  (int16_t)18538,   (int16_t)28077,   (int16_t)-26090,  (int16_t)8272,
    (int16_t)1041,    (int16_t)17615,   (int16_t)-6561,   (int16_t)1,       (int16_t)15073,   (int16_t)10816,   (int16_t)1920,    (int16_t)19940,
    (int16_t)-26909,  (int16_t)1064,    (int16_t)16640,   (int16_t)18148,   (int16_t)-19535,  (int16_t)16384,   (int16_t)4073,    (int16_t)-11927,
    (int16_t)-16334,  (int16_t)-17841,  (int16_t)-13193,  (int16_t)4322,    (int16_t)30870,   (int16_t)18449,   (int16_t)545,     (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::BANDS[64] = {
    (int16_t)2,       (int16_t)14,      (int16_t)2,       (int16_t)14,      (int16_t)14,      (int16_t)14,      (int16_t)8,       (int16_t)14,
    (int16_t)14,      (int16_t)0,       (int16_t)6,       (int16_t)8,       (int16_t)2,       (int16_t)2,       (int16_t)12,      (int16_t)2,
    (int16_t)4,       (int16_t)2,       (int16_t)8,       (int16_t)10,      (int16_t)12,      (int16_t)8,       (int16_t)2,       (int16_t)2,
    (int16_t)6,       (int16_t)0,       (int16_t)2,       (int16_t)2,       (int16_t)2,       (int16_t)4,       (int16_t)10,      (int16_t)8,
    (int16_t)10,      (int16_t)12,      (int16_t)12,      (int16_t)4,       (int16_t)10,      (int16_t)12,      (int16_t)6,       (int16_t)14,
    (int16_t)0,       (int16_t)14,      (int16_t)14,      (int16_t)4,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)4,
    (int16_t)2,       (int16_t)10,      (int16_t)2,       (int16_t)4,       (int16_t)12,      (int16_t)0,       (int16_t)8,       (int16_t)8,
    (int16_t)14,      (int16_t)14,      (int16_t)6,       (int16_t)6,       (int16_t)6,       (int16_t)4,       (int16_t)8,       (int16_t)14
};

const int16_t DataSet_1_16i::outputs::MBANDS[64] = {
    (int16_t)11410,   (int16_t)14,      (int16_t)2,       (int16_t)-27538,  (int16_t)14,      (int16_t)-31569,  (int16_t)-18871,  (int16_t)14,
    (int16_t)14,      (int16_t)-10559,  (int16_t)-5498,   (int16_t)8,       (int16_t)9682,    (int16_t)2,       (int16_t)12,      (int16_t)23203,
    (int16_t)13653,   (int16_t)2,       (int16_t)8,       (int16_t)-32069,  (int16_t)12,      (int16_t)4552,    (int16_t)-23502,  (int16_t)2,
    (int16_t)6,       (int16_t)1169,    (int16_t)26002,   (int16_t)2,       (int16_t)7522,    (int16_t)4,       (int16_t)10,      (int16_t)7417,
    (int16_t)23067,   (int16_t)12,      (int16_t)12,      (int16_t)-10491,  (int16_t)10,      (int16_t)28077,   (int16_t)-26090,  (int16_t)14,
    (int16_t)0,       (int16_t)17615,   (int16_t)-6561,   (int16_t)4,       (int16_t)15073,   (int16_t)0,       (int16_t)0,       (int16_t)19940,
    (int16_t)-26909,  (int16_t)10,      (int16_t)2,       (int16_t)18148,   (int16_t)12,      (int16_t)16384,   (int16_t)4073,    (int16_t)8,
    (int16_t)14,      (int16_t)-17841,  (int16_t)-13193,  (int16_t)6,       (int16_t)30870,   (int16_t)4,       (int16_t)8,       (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::BORV[64] = {
    (int16_t)15506,   (int16_t)-16385,  (int16_t)6131,    (int16_t)-2178,   (int16_t)-16930,  (int16_t)-11025,  (int16_t)-18871,  (int16_t)24510,
    (int16_t)-3346,   (int16_t)-305,    (int16_t)-1329,   (int16_t)-20753,  (int16_t)-23081,  (int16_t)-714,    (int16_t)-1281,   (int16_t)-9501,
    (int16_t)-161,    (int16_t)-14985,  (int16_t)-17409,  (int16_t)-14405,  (int16_t)-20897,  (int16_t)-19987,  (int16_t)-20685,  (int16_t)-129,
    (int16_t)-2114,   (int16_t)-28709,  (int16_t)30675,   (int16_t)-5,      (int16_t)-8833,   (int16_t)-2585,   (int16_t)-1094,   (int16_t)16379,
    (int16_t)-325,    (int16_t)-8196,   (int16_t)19455,   (int16_t)-10393,  (int16_t)-385,    (int16_t)-4115,   (int16_t)-25769,  (int16_t)-5249,
    (int16_t)32377,   (int16_t)24031,   (int16_t)-6273,   (int16_t)-4289,   (int16_t)-17409,  (int16_t)-16775,  (int16_t)16309,   (int16_t)-27,
    (int16_t)-25,     (int16_t)29950,   (int16_t)-12325,  (int16_t)20462,   (int16_t)-16449,  (int16_t)28602,   (int16_t)8189,    (int16_t)-10881,
    (int16_t)-2561,   (int16_t)-16641,  (int16_t)-8577,   (int16_t)24551,   (int16_t)32182,   (int16_t)-5129,   (int16_t)31609,   (int16_t)-20481
};

const int16_t DataSet_1_16i::outputs::MBORV[64] = {
    (int16_t)11410,   (int16_t)-16385,  (int16_t)6131,    (int16_t)-27538,  (int16_t)-16930,  (int16_t)-31569,  (int16_t)-18871,  (int16_t)24510,
    (int16_t)-3346,   (int16_t)-10559,  (int16_t)-5498,   (int16_t)-20753,  (int16_t)9682,    (int16_t)-714,    (int16_t)-1281,   (int16_t)23203,
    (int16_t)13653,   (int16_t)-14985,  (int16_t)-17409,  (int16_t)-32069,  (int16_t)-20897,  (int16_t)4552,    (int16_t)-23502,  (int16_t)-129,
    (int16_t)-2114,   (int16_t)1169,    (int16_t)26002,   (int16_t)-5,      (int16_t)7522,    (int16_t)-2585,   (int16_t)-1094,   (int16_t)7417,
    (int16_t)23067,   (int16_t)-8196,   (int16_t)19455,   (int16_t)-10491,  (int16_t)-385,    (int16_t)28077,   (int16_t)-26090,  (int16_t)-5249,
    (int16_t)32377,   (int16_t)17615,   (int16_t)-6561,   (int16_t)-4289,   (int16_t)15073,   (int16_t)-16775,  (int16_t)16309,   (int16_t)19940,
    (int16_t)-26909,  (int16_t)29950,   (int16_t)-12325,  (int16_t)18148,   (int16_t)-16449,  (int16_t)16384,   (int16_t)4073,    (int16_t)-10881,
    (int16_t)-2561,   (int16_t)-17841,  (int16_t)-13193,  (int16_t)24551,   (int16_t)30870,   (int16_t)-5129,   (int16_t)31609,   (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::BORS[64] = {
    (int16_t)11422,   (int16_t)-30737,  (int16_t)1903,    (int16_t)-27538,  (int16_t)15774,   (int16_t)-31569,  (int16_t)-18865,  (int16_t)23582,
    (int16_t)-7666,   (int16_t)-10545,  (int16_t)-5490,   (int16_t)11951,   (int16_t)9694,    (int16_t)-29394,  (int16_t)-2018,   (int16_t)23215,
    (int16_t)13663,   (int16_t)-16274,  (int16_t)-17553,  (int16_t)-32065,  (int16_t)-21938,  (int16_t)4558,    (int16_t)-23490,  (int16_t)2687,
    (int16_t)-7746,   (int16_t)1183,    (int16_t)26014,   (int16_t)-4562,   (int16_t)7534,    (int16_t)-3602,   (int16_t)-7282,   (int16_t)7423,
    (int16_t)23071,   (int16_t)-15426,  (int16_t)1022,    (int16_t)-10481,  (int16_t)27759,   (int16_t)28079,   (int16_t)-26082,  (int16_t)-21890,
    (int16_t)5247,    (int16_t)17615,   (int16_t)-6561,   (int16_t)-6097,   (int16_t)15087,   (int16_t)14958,   (int16_t)16270,   (int16_t)19950,
    (int16_t)-26897,  (int16_t)29806,   (int16_t)17694,   (int16_t)18158,   (int16_t)-16449,  (int16_t)16398,   (int16_t)4079,    (int16_t)-10881,
    (int16_t)-10946,  (int16_t)-17841,  (int16_t)-13185,  (int16_t)6895,    (int16_t)30878,   (int16_t)-13697,  (int16_t)19007,   (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::MBORS[64] = {
    (int16_t)11410,   (int16_t)-30737,  (int16_t)1903,    (int16_t)-27538,  (int16_t)15774,   (int16_t)-31569,  (int16_t)-18871,  (int16_t)23582,
    (int16_t)-7666,   (int16_t)-10559,  (int16_t)-5498,   (int16_t)11951,   (int16_t)9682,    (int16_t)-29394,  (int16_t)-2018,   (int16_t)23203,
    (int16_t)13653,   (int16_t)-16274,  (int16_t)-17553,  (int16_t)-32069,  (int16_t)-21938,  (int16_t)4552,    (int16_t)-23502,  (int16_t)2687,
    (int16_t)-7746,   (int16_t)1169,    (int16_t)26002,   (int16_t)-4562,   (int16_t)7522,    (int16_t)-3602,   (int16_t)-7282,   (int16_t)7417,
    (int16_t)23067,   (int16_t)-15426,  (int16_t)1022,    (int16_t)-10491,  (int16_t)27759,   (int16_t)28077,   (int16_t)-26090,  (int16_t)-21890,
    (int16_t)5247,    (int16_t)17615,   (int16_t)-6561,   (int16_t)-6097,   (int16_t)15073,   (int16_t)14958,   (int16_t)16270,   (int16_t)19940,
    (int16_t)-26909,  (int16_t)29806,   (int16_t)17694,   (int16_t)18148,   (int16_t)-16449,  (int16_t)16384,   (int16_t)4073,    (int16_t)-10881,
    (int16_t)-10946,  (int16_t)-17841,  (int16_t)-13193,  (int16_t)6895,    (int16_t)30870,   (int16_t)-13697,  (int16_t)19007,   (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::BXORV[64] = {
    (int16_t)13442,   (int16_t)15324,   (int16_t)5106,    (int16_t)30522,   (int16_t)-23356,  (int16_t)20713,   (int16_t)8193,    (int16_t)1964,
    (int16_t)28906,   (int16_t)-18354,  (int16_t)6351,    (int16_t)-24465,  (int16_t)-24569,  (int16_t)-1002,   (int16_t)-15633,  (int16_t)-27934,
    (int16_t)-177,    (int16_t)-15019,  (int16_t)4831,    (int16_t)18339,   (int16_t)11351,   (int16_t)-20187,  (int16_t)11009,   (int16_t)-179,
    (int16_t)-10580,  (int16_t)-28854,  (int16_t)4691,    (int16_t)7161,    (int16_t)-10977,  (int16_t)9543,    (int16_t)-17862,  (int16_t)13098,
    (int16_t)-4952,   (int16_t)7232,    (int16_t)19415,   (int16_t)-16029,  (int16_t)-18923,  (int16_t)-12732,  (int16_t)4949,    (int16_t)-13521,
    (int16_t)31336,   (int16_t)22814,   (int16_t)16700,   (int16_t)-4290,   (int16_t)-19553,  (int16_t)-27591,  (int16_t)14389,   (int16_t)-2363,
    (int16_t)32740,   (int16_t)28886,   (int16_t)-28965,  (int16_t)3018,    (int16_t)3086,    (int16_t)12218,   (int16_t)8156,    (int16_t)1046,
    (int16_t)13773,   (int16_t)-22863,  (int16_t)7791,    (int16_t)20229,   (int16_t)17716,   (int16_t)-23578,  (int16_t)31064,   (int16_t)9695
};

const int16_t DataSet_1_16i::outputs::MBXORV[64] = {
    (int16_t)11410,   (int16_t)15324,   (int16_t)5106,    (int16_t)-27538,  (int16_t)-23356,  (int16_t)-31569,  (int16_t)-18871,  (int16_t)1964,
    (int16_t)28906,   (int16_t)-10559,  (int16_t)-5498,   (int16_t)-24465,  (int16_t)9682,    (int16_t)-1002,   (int16_t)-15633,  (int16_t)23203,
    (int16_t)13653,   (int16_t)-15019,  (int16_t)4831,    (int16_t)-32069,  (int16_t)11351,   (int16_t)4552,    (int16_t)-23502,  (int16_t)-179,
    (int16_t)-10580,  (int16_t)1169,    (int16_t)26002,   (int16_t)7161,    (int16_t)7522,    (int16_t)9543,    (int16_t)-17862,  (int16_t)7417,
    (int16_t)23067,   (int16_t)7232,    (int16_t)19415,   (int16_t)-10491,  (int16_t)-18923,  (int16_t)28077,   (int16_t)-26090,  (int16_t)-13521,
    (int16_t)31336,   (int16_t)17615,   (int16_t)-6561,   (int16_t)-4290,   (int16_t)15073,   (int16_t)-27591,  (int16_t)14389,   (int16_t)19940,
    (int16_t)-26909,  (int16_t)28886,   (int16_t)-28965,  (int16_t)18148,   (int16_t)3086,    (int16_t)16384,   (int16_t)4073,    (int16_t)1046,
    (int16_t)13773,   (int16_t)-17841,  (int16_t)-13193,  (int16_t)20229,   (int16_t)30870,   (int16_t)-23578,  (int16_t)31064,   (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::BXORS[64] = {
    (int16_t)11420,   (int16_t)-30751,  (int16_t)1901,    (int16_t)-27552,  (int16_t)15760,   (int16_t)-31583,  (int16_t)-18873,  (int16_t)23568,
    (int16_t)-7680,   (int16_t)-10545,  (int16_t)-5496,   (int16_t)11943,   (int16_t)9692,    (int16_t)-29396,  (int16_t)-2030,   (int16_t)23213,
    (int16_t)13659,   (int16_t)-16276,  (int16_t)-17561,  (int16_t)-32075,  (int16_t)-21950,  (int16_t)4550,    (int16_t)-23492,  (int16_t)2685,
    (int16_t)-7752,   (int16_t)1183,    (int16_t)26012,   (int16_t)-4564,   (int16_t)7532,    (int16_t)-3606,   (int16_t)-7292,   (int16_t)7415,
    (int16_t)23061,   (int16_t)-15438,  (int16_t)1010,    (int16_t)-10485,  (int16_t)27749,   (int16_t)28067,   (int16_t)-26088,  (int16_t)-21904,
    (int16_t)5247,    (int16_t)17601,   (int16_t)-6575,   (int16_t)-6101,   (int16_t)15087,   (int16_t)14958,   (int16_t)16270,   (int16_t)19946,
    (int16_t)-26899,  (int16_t)29796,   (int16_t)17692,   (int16_t)18154,   (int16_t)-16461,  (int16_t)16398,   (int16_t)4071,    (int16_t)-10889,
    (int16_t)-10960,  (int16_t)-17855,  (int16_t)-13191,  (int16_t)6889,    (int16_t)30872,   (int16_t)-13701,  (int16_t)18999,   (int16_t)-20688
};

const int16_t DataSet_1_16i::outputs::MBXORS[64] = {
    (int16_t)11410,   (int16_t)-30751,  (int16_t)1901,    (int16_t)-27538,  (int16_t)15760,   (int16_t)-31569,  (int16_t)-18871,  (int16_t)23568,
    (int16_t)-7680,   (int16_t)-10559,  (int16_t)-5498,   (int16_t)11943,   (int16_t)9682,    (int16_t)-29396,  (int16_t)-2030,   (int16_t)23203,
    (int16_t)13653,   (int16_t)-16276,  (int16_t)-17561,  (int16_t)-32069,  (int16_t)-21950,  (int16_t)4552,    (int16_t)-23502,  (int16_t)2685,
    (int16_t)-7752,   (int16_t)1169,    (int16_t)26002,   (int16_t)-4564,   (int16_t)7522,    (int16_t)-3606,   (int16_t)-7292,   (int16_t)7417,
    (int16_t)23067,   (int16_t)-15438,  (int16_t)1010,    (int16_t)-10491,  (int16_t)27749,   (int16_t)28077,   (int16_t)-26090,  (int16_t)-21904,
    (int16_t)5247,    (int16_t)17615,   (int16_t)-6561,   (int16_t)-6101,   (int16_t)15073,   (int16_t)14958,   (int16_t)16270,   (int16_t)19940,
    (int16_t)-26909,  (int16_t)29796,   (int16_t)17692,   (int16_t)18148,   (int16_t)-16461,  (int16_t)16384,   (int16_t)4073,    (int16_t)-10889,
    (int16_t)-10960,  (int16_t)-17841,  (int16_t)-13193,  (int16_t)6889,    (int16_t)30870,   (int16_t)-13701,  (int16_t)18999,   (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::BNOT[64] = {
    (int16_t)-11411,  (int16_t)30736,   (int16_t)-1892,   (int16_t)27537,   (int16_t)-15775,  (int16_t)31568,   (int16_t)18870,   (int16_t)-23583,
    (int16_t)7665,    (int16_t)10558,   (int16_t)5497,    (int16_t)-11946,  (int16_t)-9683,   (int16_t)29405,   (int16_t)2019,    (int16_t)-23204,
    (int16_t)-13654,  (int16_t)16285,   (int16_t)17558,   (int16_t)32068,   (int16_t)21939,   (int16_t)-4553,   (int16_t)23501,   (int16_t)-2676,
    (int16_t)7753,    (int16_t)-1170,   (int16_t)-26003,  (int16_t)4573,    (int16_t)-7523,   (int16_t)3611,    (int16_t)7285,    (int16_t)-7418,
    (int16_t)-23068,  (int16_t)15427,   (int16_t)-1021,   (int16_t)10490,   (int16_t)-27756,  (int16_t)-28078,  (int16_t)26089,   (int16_t)21889,
    (int16_t)-5234,   (int16_t)-17616,  (int16_t)6560,    (int16_t)6106,    (int16_t)-15074,  (int16_t)-14945,  (int16_t)-16257,  (int16_t)-19941,
    (int16_t)26908,   (int16_t)-29803,  (int16_t)-17683,  (int16_t)-18149,  (int16_t)16450,   (int16_t)-16385,  (int16_t)-4074,   (int16_t)10886,
    (int16_t)10945,   (int16_t)17840,   (int16_t)13192,   (int16_t)-6888,   (int16_t)-30871,  (int16_t)13706,   (int16_t)-19002,  (int16_t)20673
};

const int16_t DataSet_1_16i::outputs::MBNOT[64] = {
    (int16_t)11410,   (int16_t)30736,   (int16_t)-1892,   (int16_t)-27538,  (int16_t)-15775,  (int16_t)-31569,  (int16_t)-18871,  (int16_t)-23583,
    (int16_t)7665,    (int16_t)-10559,  (int16_t)-5498,   (int16_t)-11946,  (int16_t)9682,    (int16_t)29405,   (int16_t)2019,    (int16_t)23203,
    (int16_t)13653,   (int16_t)16285,   (int16_t)17558,   (int16_t)-32069,  (int16_t)21939,   (int16_t)4552,    (int16_t)-23502,  (int16_t)-2676,
    (int16_t)7753,    (int16_t)1169,    (int16_t)26002,   (int16_t)4573,    (int16_t)7522,    (int16_t)3611,    (int16_t)7285,    (int16_t)7417,
    (int16_t)23067,   (int16_t)15427,   (int16_t)-1021,   (int16_t)-10491,  (int16_t)-27756,  (int16_t)28077,   (int16_t)-26090,  (int16_t)21889,
    (int16_t)-5234,   (int16_t)17615,   (int16_t)-6561,   (int16_t)6106,    (int16_t)15073,   (int16_t)-14945,  (int16_t)-16257,  (int16_t)19940,
    (int16_t)-26909,  (int16_t)-29803,  (int16_t)-17683,  (int16_t)18148,   (int16_t)16450,   (int16_t)16384,   (int16_t)4073,    (int16_t)10886,
    (int16_t)10945,   (int16_t)-17841,  (int16_t)-13193,  (int16_t)-6888,   (int16_t)30870,   (int16_t)13706,   (int16_t)-19002,  (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::HBAND[64] = {
    (int16_t)11410,   (int16_t)1154,    (int16_t)1026,    (int16_t)1026,    (int16_t)1026,    (int16_t)1026,    (int16_t)1024,    (int16_t)1024,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0
};

const int16_t DataSet_1_16i::outputs::MHBAND[64] = {
    (int16_t)-1,    (int16_t)-30737,        (int16_t)1891,  (int16_t)1891,  (int16_t)1282,  (int16_t)1282,  (int16_t)1282,  (int16_t)1026,
    (int16_t)2,     (int16_t)2,     (int16_t)2,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,
    (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,
    (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,
    (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,
    (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,
    (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,
    (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0,     (int16_t)0
};

const int16_t DataSet_1_16i::outputs::HBANDS[64] = {
    (int16_t)2,       (int16_t)2,       (int16_t)2,       (int16_t)2,       (int16_t)2,       (int16_t)2,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0
};

const int16_t DataSet_1_16i::outputs::MHBANDS[64] = {
    (int16_t)14,      (int16_t)14,      (int16_t)2,       (int16_t)2,       (int16_t)2,       (int16_t)2,       (int16_t)2,       (int16_t)2,
    (int16_t)2,       (int16_t)2,       (int16_t)2,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0,       (int16_t)0
};

const int16_t DataSet_1_16i::outputs::HBOR[64] = {
    (int16_t)11410,   (int16_t)-20481,  (int16_t)-20481,  (int16_t)-16385,  (int16_t)-16385,  (int16_t)-16385,  (int16_t)-16385,  (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1
};

const int16_t DataSet_1_16i::outputs::MHBOR[64] = {
    (int16_t)0,       (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-16385,  (int16_t)-16385,  (int16_t)-16385,  (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1
};

const int16_t DataSet_1_16i::outputs::HBORS[64] = {
    (int16_t)11422,   (int16_t)-20481,  (int16_t)-20481,  (int16_t)-16385,  (int16_t)-16385,  (int16_t)-16385,  (int16_t)-16385,  (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1
};

const int16_t DataSet_1_16i::outputs::MHBORS[64] = {
    (int16_t)14,      (int16_t)-30737,  (int16_t)-30737,  (int16_t)-30737,  (int16_t)-16385,  (int16_t)-16385,  (int16_t)-16385,  (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,
    (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1,      (int16_t)-1
};

const int16_t DataSet_1_16i::outputs::HBXOR[64] = {
    (int16_t)11410,   (int16_t)-21635,  (int16_t)-21474,  (int16_t)14448,   (int16_t)1518,    (int16_t)-32447,  (int16_t)14088,   (int16_t)27414,
    (int16_t)-30440,  (int16_t)24537,   (int16_t)-19105,  (int16_t)-25610,  (int16_t)-16860,  (int16_t)13062,   (int16_t)-13542,  (int16_t)-28231,
    (int16_t)-23316,  (int16_t)25742,   (int16_t)-8217,   (int16_t)23900,   (int16_t)-2288,   (int16_t)-6440,   (int16_t)17130,   (int16_t)18585,
    (int16_t)-22225,  (int16_t)-21058,  (int16_t)-14292,  (int16_t)9742,    (int16_t)15212,   (int16_t)-13688,  (int16_t)10498,   (int16_t)13819,
    (int16_t)28640,   (int16_t)-21412,  (int16_t)-20576,  (int16_t)30885,   (int16_t)5326,    (int16_t)31075,   (int16_t)-7307,   (int16_t)18699,
    (int16_t)23930,   (int16_t)6581,    (int16_t)-22,     (int16_t)6095,    (int16_t)11566,   (int16_t)5966,    (int16_t)10446,   (int16_t)25898,
    (int16_t)-3127,   (int16_t)-30813,  (int16_t)-15695,  (int16_t)-31659,  (int16_t)15336,   (int16_t)31720,   (int16_t)29697,   (int16_t)-24200,
    (int16_t)29766,   (int16_t)-12791,  (int16_t)638,     (int16_t)6297,    (int16_t)24591,   (int16_t)-21894,  (int16_t)-8125,   (int16_t)20349
};

const int16_t DataSet_1_16i::outputs::MHBXOR[64] = {
    (int16_t)0,       (int16_t)-30737,  (int16_t)-32628,  (int16_t)-32628,  (int16_t)-17134,  (int16_t)-17134,  (int16_t)-17134,  (int16_t)-7924,
    (int16_t)770,     (int16_t)770,     (int16_t)770,     (int16_t)11691,   (int16_t)11691,   (int16_t)-24439,  (int16_t)22677,   (int16_t)22677,
    (int16_t)22677,   (int16_t)-26377,  (int16_t)9118,    (int16_t)9118,    (int16_t)-30254,  (int16_t)-30254,  (int16_t)-30254,  (int16_t)-31839,
    (int16_t)25111,   (int16_t)25111,   (int16_t)25111,   (int16_t)-29643,  (int16_t)-29643,  (int16_t)32209,   (int16_t)-24997,  (int16_t)-24997,
    (int16_t)-24997,  (int16_t)24039,   (int16_t)24091,   (int16_t)24091,   (int16_t)12912,   (int16_t)12912,   (int16_t)12912,   (int16_t)-26610,
    (int16_t)-29569,  (int16_t)-29569,  (int16_t)-29569,  (int16_t)25690,   (int16_t)25690,   (int16_t)24122,   (int16_t)25018,   (int16_t)25018,
    (int16_t)25018,   (int16_t)5584,    (int16_t)20674,   (int16_t)20674,   (int16_t)-4225,   (int16_t)-4225,   (int16_t)-4225,   (int16_t)14854,
    (int16_t)-4296,   (int16_t)-4296,   (int16_t)-4296,   (int16_t)-2593,   (int16_t)-2593,   (int16_t)16298,   (int16_t)30099,   (int16_t)30099
};

const int16_t DataSet_1_16i::outputs::HBXORS[64] = {
    (int16_t)11420,   (int16_t)-21645,  (int16_t)-21488,  (int16_t)14462,   (int16_t)1504,    (int16_t)-32433,  (int16_t)14086,   (int16_t)27416,
    (int16_t)-30442,  (int16_t)24535,   (int16_t)-19119,  (int16_t)-25608,  (int16_t)-16854,  (int16_t)13064,   (int16_t)-13548,  (int16_t)-28233,
    (int16_t)-23326,  (int16_t)25728,   (int16_t)-8215,   (int16_t)23890,   (int16_t)-2274,   (int16_t)-6442,   (int16_t)17124,   (int16_t)18583,
    (int16_t)-22239,  (int16_t)-21072,  (int16_t)-14302,  (int16_t)9728,    (int16_t)15202,   (int16_t)-13690,  (int16_t)10508,   (int16_t)13813,
    (int16_t)28654,   (int16_t)-21422,  (int16_t)-20562,  (int16_t)30891,   (int16_t)5312,    (int16_t)31085,   (int16_t)-7301,   (int16_t)18693,
    (int16_t)23924,   (int16_t)6587,    (int16_t)-28,     (int16_t)6081,    (int16_t)11552,   (int16_t)5952,    (int16_t)10432,   (int16_t)25892,
    (int16_t)-3129,   (int16_t)-30803,  (int16_t)-15681,  (int16_t)-31653,  (int16_t)15334,   (int16_t)31718,   (int16_t)29711,   (int16_t)-24202,
    (int16_t)29768,   (int16_t)-12793,  (int16_t)624,     (int16_t)6295,    (int16_t)24577,   (int16_t)-21900,  (int16_t)-8115,   (int16_t)20339
};

const int16_t DataSet_1_16i::outputs::MHBXORS[64] = {
    (int16_t)14,      (int16_t)-30751,  (int16_t)-32638,  (int16_t)-32638,  (int16_t)-17124,  (int16_t)-17124,  (int16_t)-17124,  (int16_t)-7934,
    (int16_t)780,     (int16_t)780,     (int16_t)780,     (int16_t)11685,   (int16_t)11685,   (int16_t)-24441,  (int16_t)22683,   (int16_t)22683,
    (int16_t)22683,   (int16_t)-26375,  (int16_t)9104,    (int16_t)9104,    (int16_t)-30244,  (int16_t)-30244,  (int16_t)-30244,  (int16_t)-31825,
    (int16_t)25113,   (int16_t)25113,   (int16_t)25113,   (int16_t)-29637,  (int16_t)-29637,  (int16_t)32223,   (int16_t)-25003,  (int16_t)-25003,
    (int16_t)-25003,  (int16_t)24041,   (int16_t)24085,   (int16_t)24085,   (int16_t)12926,   (int16_t)12926,   (int16_t)12926,   (int16_t)-26624,
    (int16_t)-29583,  (int16_t)-29583,  (int16_t)-29583,  (int16_t)25684,   (int16_t)25684,   (int16_t)24116,   (int16_t)25012,   (int16_t)25012,
    (int16_t)25012,   (int16_t)5598,    (int16_t)20684,   (int16_t)20684,   (int16_t)-4239,   (int16_t)-4239,   (int16_t)-4239,   (int16_t)14856,
    (int16_t)-4298,   (int16_t)-4298,   (int16_t)-4298,   (int16_t)-2607,   (int16_t)-2607,   (int16_t)16292,   (int16_t)30109,   (int16_t)30109
};

const int16_t DataSet_1_16i::outputs::LSHV[64] = {
    (int16_t)25744,   (int16_t)30720,   (int16_t)24576,   (int16_t)-9216,   (int16_t)-4880,   (int16_t)-8192,   (int16_t)-28160,  (int16_t)-7952,
    (int16_t)-16384,  (int16_t)-32768,  (int16_t)6144,    (int16_t)30024,   (int16_t)-23552,  (int16_t)18560,   (int16_t)896,     (int16_t)-32768,
    (int16_t)-22016,  (int16_t)4096,    (int16_t)-32768,  (int16_t)-32768,  (int16_t)18816,   (int16_t)16384,   (int16_t)-28472,  (int16_t)-6656,
    (int16_t)28032,   (int16_t)-32768,  (int16_t)-27064,  (int16_t)-30720,  (int16_t)-30720,  (int16_t)-28672,  (int16_t)-24576,  (int16_t)31872,
    (int16_t)-10240,  (int16_t)-32768,  (int16_t)0,       (int16_t)10240,   (int16_t)27392,   (int16_t)-9382,   (int16_t)-24224,  (int16_t)-16384,
    (int16_t)-15360,  (int16_t)-12544,  (int16_t)-16384,  (int16_t)-24428,  (int16_t)-32768,  (int16_t)0,       (int16_t)0,       (int16_t)0,
    (int16_t)29056,   (int16_t)-22528,  (int16_t)16384,   (int16_t)-9088,   (int16_t)-17152,  (int16_t)0,       (int16_t)8192,    (int16_t)30976,
    (int16_t)21472,   (int16_t)20224,   (int16_t)-18432,  (int16_t)27548,   (int16_t)19200,   (int16_t)-22704,  (int16_t)8192,    (int16_t)31216
};

const int16_t DataSet_1_16i::outputs::MLSHV[64] = {
    (int16_t)11410,   (int16_t)30720,   (int16_t)24576,   (int16_t)-27538,  (int16_t)-4880,   (int16_t)-31569,  (int16_t)-18871,  (int16_t)-7952,
    (int16_t)-16384,  (int16_t)-10559,  (int16_t)-5498,   (int16_t)30024,   (int16_t)9682,    (int16_t)18560,   (int16_t)896,     (int16_t)23203,
    (int16_t)13653,   (int16_t)4096,    (int16_t)-32768,  (int16_t)-32069,  (int16_t)18816,   (int16_t)4552,    (int16_t)-23502,  (int16_t)-6656,
    (int16_t)28032,   (int16_t)1169,    (int16_t)26002,   (int16_t)-30720,  (int16_t)7522,    (int16_t)-28672,  (int16_t)-24576,  (int16_t)7417,
    (int16_t)23067,   (int16_t)-32768,  (int16_t)0,       (int16_t)-10491,  (int16_t)27392,   (int16_t)28077,   (int16_t)-26090,  (int16_t)-16384,
    (int16_t)-15360,  (int16_t)17615,   (int16_t)-6561,   (int16_t)-24428,  (int16_t)15073,   (int16_t)0,       (int16_t)0,       (int16_t)19940,
    (int16_t)-26909,  (int16_t)-22528,  (int16_t)16384,   (int16_t)18148,   (int16_t)-17152,  (int16_t)16384,   (int16_t)4073,    (int16_t)30976,
    (int16_t)21472,   (int16_t)-17841,  (int16_t)-13193,  (int16_t)27548,   (int16_t)30870,   (int16_t)-22704,  (int16_t)8192,    (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::LSHS[64] = {
    (int16_t)25744, (int16_t)16248, (int16_t)15128, (int16_t)-23696,        (int16_t)-4880, (int16_t)9592,  (int16_t)-19896,        (int16_t)-7952,
    (int16_t)4208,  (int16_t)-18936,        (int16_t)21552, (int16_t)30024, (int16_t)11920, (int16_t)26896, (int16_t)-16160,        (int16_t)-10984,
    (int16_t)-21848,        (int16_t)784,   (int16_t)-9400, (int16_t)5592,  (int16_t)21088, (int16_t)-29120,        (int16_t)8592,  (int16_t)21400,
    (int16_t)3504,  (int16_t)9352,  (int16_t)11408, (int16_t)28944, (int16_t)-5360, (int16_t)-28896,        (int16_t)7248,  (int16_t)-6200,
    (int16_t)-12072,        (int16_t)7648,  (int16_t)8160,  (int16_t)-18392,        (int16_t)25432, (int16_t)28008, (int16_t)-12112,        (int16_t)21488,
    (int16_t)-23672,        (int16_t)9848,  (int16_t)13048, (int16_t)16680, (int16_t)-10488,        (int16_t)-11520,        (int16_t)-1024, (int16_t)28448,
    (int16_t)-18664,        (int16_t)-23728,        (int16_t)10384, (int16_t)14112, (int16_t)-536,  (int16_t)0,     (int16_t)32584, (int16_t)-21560,
    (int16_t)-22032,        (int16_t)-11656,        (int16_t)25528, (int16_t)-10440,        (int16_t)-15184,        (int16_t)21416, (int16_t)20936, (int16_t)31216
};

const int16_t DataSet_1_16i::outputs::MLSHS[64] = {
    (int16_t)11410, (int16_t)16248, (int16_t)15128, (int16_t)-27538,        (int16_t)-4880, (int16_t)-31569,        (int16_t)-18871,        (int16_t)-7952,
    (int16_t)4208,  (int16_t)-10559,        (int16_t)-5498, (int16_t)30024, (int16_t)9682,  (int16_t)26896, (int16_t)-16160,        (int16_t)23203,
    (int16_t)13653, (int16_t)784,   (int16_t)-9400, (int16_t)-32069,        (int16_t)21088, (int16_t)4552,  (int16_t)-23502,        (int16_t)21400,
    (int16_t)3504,  (int16_t)1169,  (int16_t)26002, (int16_t)28944, (int16_t)7522,  (int16_t)-28896,        (int16_t)7248,  (int16_t)7417,
    (int16_t)23067, (int16_t)7648,  (int16_t)8160,  (int16_t)-10491,        (int16_t)25432, (int16_t)28077, (int16_t)-26090,        (int16_t)21488,
    (int16_t)-23672,        (int16_t)17615, (int16_t)-6561, (int16_t)16680, (int16_t)15073, (int16_t)-11520,        (int16_t)-1024, (int16_t)19940,
    (int16_t)-26909,        (int16_t)-23728,        (int16_t)10384, (int16_t)18148, (int16_t)-536,  (int16_t)16384, (int16_t)4073,  (int16_t)-21560,
    (int16_t)-22032,        (int16_t)-17841,        (int16_t)-13193,        (int16_t)-10440,        (int16_t)30870, (int16_t)21416, (int16_t)20936, (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::RSHV[64] = {
    (int16_t)1426,  (int16_t)-16,   (int16_t)0,     (int16_t)-54,   (int16_t)1971,  (int16_t)-4,    (int16_t)-37,   (int16_t)2947,
    (int16_t)-1,    (int16_t)-1,    (int16_t)-6,    (int16_t)1493,  (int16_t)18,    (int16_t)-460,  (int16_t)-64,   (int16_t)0,
    (int16_t)26,    (int16_t)-8,    (int16_t)-1,    (int16_t)-1,    (int16_t)-686,  (int16_t)2,     (int16_t)-5876, (int16_t)5,
    (int16_t)-122,  (int16_t)0,     (int16_t)6500,  (int16_t)-5,    (int16_t)7,     (int16_t)-4,    (int16_t)-2,    (int16_t)57,
    (int16_t)11,    (int16_t)-2,    (int16_t)0,     (int16_t)-6,    (int16_t)108,   (int16_t)14038, (int16_t)-1631, (int16_t)-3,
    (int16_t)5,     (int16_t)68,    (int16_t)-1,    (int16_t)-1527, (int16_t)0,     (int16_t)7,     (int16_t)31,    (int16_t)0,
    (int16_t)-211,  (int16_t)29,    (int16_t)2,     (int16_t)567,   (int16_t)-65,   (int16_t)4,     (int16_t)0,     (int16_t)-43,
    (int16_t)-685,  (int16_t)-70,   (int16_t)-7,    (int16_t)1721,  (int16_t)241,   (int16_t)-857,  (int16_t)2,     (int16_t)-2585
};

const int16_t DataSet_1_16i::outputs::MRSHV[64] = {
    (int16_t)11410, (int16_t)-16,   (int16_t)0,     (int16_t)-27538,        (int16_t)1971,  (int16_t)-31569,        (int16_t)-18871,        (int16_t)2947,
    (int16_t)-1,    (int16_t)-10559,        (int16_t)-5498, (int16_t)1493,  (int16_t)9682,  (int16_t)-460,  (int16_t)-64,   (int16_t)23203,
    (int16_t)13653, (int16_t)-8,    (int16_t)-1,    (int16_t)-32069,        (int16_t)-686,  (int16_t)4552,  (int16_t)-23502,        (int16_t)5,
    (int16_t)-122,  (int16_t)1169,  (int16_t)26002, (int16_t)-5,    (int16_t)7522,  (int16_t)-4,    (int16_t)-2,    (int16_t)7417,
    (int16_t)23067, (int16_t)-2,    (int16_t)0,     (int16_t)-10491,        (int16_t)108,   (int16_t)28077, (int16_t)-26090,        (int16_t)-3,
    (int16_t)5,     (int16_t)17615, (int16_t)-6561, (int16_t)-1527, (int16_t)15073, (int16_t)7,     (int16_t)31,    (int16_t)19940,
    (int16_t)-26909,        (int16_t)29,    (int16_t)2,     (int16_t)18148, (int16_t)-65,   (int16_t)16384, (int16_t)4073,  (int16_t)-43,
    (int16_t)-685,  (int16_t)-17841,        (int16_t)-13193,        (int16_t)1721,  (int16_t)30870, (int16_t)-857,  (int16_t)2,     (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::RSHS[64] = {
    (int16_t)1426,  (int16_t)-3843, (int16_t)236,   (int16_t)-3443, (int16_t)1971,  (int16_t)-3947, (int16_t)-2359, (int16_t)2947,
    (int16_t)-959,  (int16_t)-1320, (int16_t)-688,  (int16_t)1493,  (int16_t)1210,  (int16_t)-3676, (int16_t)-253,  (int16_t)2900,
    (int16_t)1706,  (int16_t)-2036, (int16_t)-2195, (int16_t)-4009, (int16_t)-2743, (int16_t)569,   (int16_t)-2938, (int16_t)334,
    (int16_t)-970,  (int16_t)146,   (int16_t)3250,  (int16_t)-572,  (int16_t)940,   (int16_t)-452,  (int16_t)-911,  (int16_t)927,
    (int16_t)2883,  (int16_t)-1929, (int16_t)127,   (int16_t)-1312, (int16_t)3469,  (int16_t)3509,  (int16_t)-3262, (int16_t)-2737,
    (int16_t)654,   (int16_t)2201,  (int16_t)-821,  (int16_t)-764,  (int16_t)1884,  (int16_t)1868,  (int16_t)2032,  (int16_t)2492,
    (int16_t)-3364, (int16_t)3725,  (int16_t)2210,  (int16_t)2268,  (int16_t)-2057, (int16_t)2048,  (int16_t)509,   (int16_t)-1361,
    (int16_t)-1369, (int16_t)-2231, (int16_t)-1650, (int16_t)860,   (int16_t)3858,  (int16_t)-1714, (int16_t)2375,  (int16_t)-2585
};

const int16_t DataSet_1_16i::outputs::MRSHS[64] = {
    (int16_t)11410, (int16_t)-3843, (int16_t)236,   (int16_t)-27538,        (int16_t)1971,  (int16_t)-31569,        (int16_t)-18871,        (int16_t)2947,
    (int16_t)-959,  (int16_t)-10559,        (int16_t)-5498, (int16_t)1493,  (int16_t)9682,  (int16_t)-3676, (int16_t)-253,  (int16_t)23203,
    (int16_t)13653, (int16_t)-2036, (int16_t)-2195, (int16_t)-32069,        (int16_t)-2743, (int16_t)4552,  (int16_t)-23502,        (int16_t)334,
    (int16_t)-970,  (int16_t)1169,  (int16_t)26002, (int16_t)-572,  (int16_t)7522,  (int16_t)-452,  (int16_t)-911,  (int16_t)7417,
    (int16_t)23067, (int16_t)-1929, (int16_t)127,   (int16_t)-10491,        (int16_t)3469,  (int16_t)28077, (int16_t)-26090,        (int16_t)-2737,
    (int16_t)654,   (int16_t)17615, (int16_t)-6561, (int16_t)-764,  (int16_t)15073, (int16_t)1868,  (int16_t)2032,  (int16_t)19940,
    (int16_t)-26909,        (int16_t)3725,  (int16_t)2210,  (int16_t)18148, (int16_t)-2057, (int16_t)16384, (int16_t)4073,  (int16_t)-1361,
    (int16_t)-1369, (int16_t)-17841,        (int16_t)-13193,        (int16_t)860,   (int16_t)30870, (int16_t)-1714, (int16_t)2375,  (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::ROLV[64] = {
    (int16_t)25745, (int16_t)31807, (int16_t)24812, (int16_t)-8920, (int16_t)-4879, (int16_t)-3947, (int16_t)-27796,        (int16_t)-7950,
    (int16_t)-9151, (int16_t)-5280, (int16_t)7082,  (int16_t)30025, (int16_t)-23477,        (int16_t)18595, (int16_t)927,   (int16_t)-21167,
    (int16_t)-21910,        (int16_t)5635,  (int16_t)-8780, (int16_t)-16035,        (int16_t)18837, (int16_t)16526, (int16_t)-28470,        (int16_t)-6636,
    (int16_t)28088, (int16_t)-32184,        (int16_t)-27063,        (int16_t)-29768,        (int16_t)-30603,        (int16_t)-27705,        (int16_t)-20936,        (int16_t)31886,
    (int16_t)-9520, (int16_t)-26505,        (int16_t)255,   (int16_t)11960, (int16_t)27500, (int16_t)-9382, (int16_t)-24215,        (int16_t)-10929,
    (int16_t)-15279,        (int16_t)-12476,        (int16_t)-1641, (int16_t)-24425,        (int16_t)-25232,        (int16_t)467,   (int16_t)127,   (int16_t)9970,
    (int16_t)29131, (int16_t)-22063,        (int16_t)18594, (int16_t)-9080, (int16_t)-16961,        (int16_t)1024,  (int16_t)8701,  (int16_t)31189,
    (int16_t)21485, (int16_t)20410, (int16_t)-16797,        (int16_t)27548, (int16_t)19260, (int16_t)-22692,        (int16_t)10567, (int16_t)31221
};

const int16_t DataSet_1_16i::outputs::MROLV[64] = {
    (int16_t)11410, (int16_t)31807, (int16_t)24812, (int16_t)-27538,        (int16_t)-4879, (int16_t)-31569,        (int16_t)-18871,        (int16_t)-7950,
    (int16_t)-9151, (int16_t)-10559,        (int16_t)-5498, (int16_t)30025, (int16_t)9682,  (int16_t)18595, (int16_t)927,   (int16_t)23203,
    (int16_t)13653, (int16_t)5635,  (int16_t)-8780, (int16_t)-32069,        (int16_t)18837, (int16_t)4552,  (int16_t)-23502,        (int16_t)-6636,
    (int16_t)28088, (int16_t)1169,  (int16_t)26002, (int16_t)-29768,        (int16_t)7522,  (int16_t)-27705,        (int16_t)-20936,        (int16_t)7417,
    (int16_t)23067, (int16_t)-26505,        (int16_t)255,   (int16_t)-10491,        (int16_t)27500, (int16_t)28077, (int16_t)-26090,        (int16_t)-10929,
    (int16_t)-15279,        (int16_t)17615, (int16_t)-6561, (int16_t)-24425,        (int16_t)15073, (int16_t)467,   (int16_t)127,   (int16_t)19940,
    (int16_t)-26909,        (int16_t)-22063,        (int16_t)18594, (int16_t)18148, (int16_t)-16961,        (int16_t)16384, (int16_t)4073,  (int16_t)31189,
    (int16_t)21485, (int16_t)-17841,        (int16_t)-13193,        (int16_t)27548, (int16_t)30870, (int16_t)-22692,        (int16_t)10567, (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::ROLS[64] = {
    (int16_t)25745, (int16_t)16252, (int16_t)15128, (int16_t)-23692,        (int16_t)-4879, (int16_t)9596,  (int16_t)-19891,        (int16_t)-7950,
    (int16_t)4215,  (int16_t)-18930,        (int16_t)21559, (int16_t)30025, (int16_t)11921, (int16_t)26900, (int16_t)-16153,        (int16_t)-10982,
    (int16_t)-21847,        (int16_t)790,   (int16_t)-9395, (int16_t)5596,  (int16_t)21093, (int16_t)-29120,        (int16_t)8597,  (int16_t)21400,
    (int16_t)3511,  (int16_t)9352,  (int16_t)11411, (int16_t)28951, (int16_t)-5360, (int16_t)-28889,        (int16_t)7255,  (int16_t)-6200,
    (int16_t)-12070,        (int16_t)7654,  (int16_t)8160,  (int16_t)-18386,        (int16_t)25435, (int16_t)28011, (int16_t)-12108,        (int16_t)21493,
    (int16_t)-23672,        (int16_t)9850,  (int16_t)13055, (int16_t)16687, (int16_t)-10487,        (int16_t)-11519,        (int16_t)-1023, (int16_t)28450,
    (int16_t)-18660,        (int16_t)-23725,        (int16_t)10386, (int16_t)14114, (int16_t)-531,  (int16_t)2,     (int16_t)32584, (int16_t)-21554,
    (int16_t)-22026,        (int16_t)-11651,        (int16_t)25534, (int16_t)-10440,        (int16_t)-15181,        (int16_t)21422, (int16_t)20938, (int16_t)31221
};

const int16_t DataSet_1_16i::outputs::MROLS[64] = {
    (int16_t)11410, (int16_t)16252, (int16_t)15128, (int16_t)-27538,        (int16_t)-4879, (int16_t)-31569,        (int16_t)-18871,        (int16_t)-7950,
    (int16_t)4215,  (int16_t)-10559,        (int16_t)-5498, (int16_t)30025, (int16_t)9682,  (int16_t)26900, (int16_t)-16153,        (int16_t)23203,
    (int16_t)13653, (int16_t)790,   (int16_t)-9395, (int16_t)-32069,        (int16_t)21093, (int16_t)4552,  (int16_t)-23502,        (int16_t)21400,
    (int16_t)3511,  (int16_t)1169,  (int16_t)26002, (int16_t)28951, (int16_t)7522,  (int16_t)-28889,        (int16_t)7255,  (int16_t)7417,
    (int16_t)23067, (int16_t)7654,  (int16_t)8160,  (int16_t)-10491,        (int16_t)25435, (int16_t)28077, (int16_t)-26090,        (int16_t)21493,
    (int16_t)-23672,        (int16_t)17615, (int16_t)-6561, (int16_t)16687, (int16_t)15073, (int16_t)-11519,        (int16_t)-1023, (int16_t)19940,
    (int16_t)-26909,        (int16_t)-23725,        (int16_t)10386, (int16_t)18148, (int16_t)-531,  (int16_t)16384, (int16_t)4073,  (int16_t)-21554,
    (int16_t)-22026,        (int16_t)-17841,        (int16_t)-13193,        (int16_t)-10440,        (int16_t)30870, (int16_t)21422, (int16_t)20938, (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::RORV[64] = {
    (int16_t)17810, (int16_t)-528,  (int16_t)15128, (int16_t)14154, (int16_t)-14413,        (int16_t)9596,  (int16_t)9435,  (int16_t)-13437,
    (int16_t)4215,  (int16_t)-21117,        (int16_t)-24134,        (int16_t)9685,  (int16_t)-5870, (int16_t)-30156,        (int16_t)-6208, (int16_t)-19130,
    (int16_t)-21862,        (int16_t)3160,  (int16_t)30419, (int16_t)1399,  (int16_t)25938, (int16_t)14594, (int16_t)-22260,        (int16_t)14725,
    (int16_t)-9338, (int16_t)2338,  (int16_t)-26268,        (int16_t)-30533,        (int16_t)22663, (int16_t)31036, (int16_t)14510, (int16_t)-3527,
    (int16_t)17259, (int16_t)7654,  (int16_t)4080,  (int16_t)-8006, (int16_t)27500, (int16_t)-18730,        (int16_t)27041, (int16_t)21493,
    (int16_t)7237,  (int16_t)-12476,        (int16_t)-26241,        (int16_t)31241, (int16_t)30146, (int16_t)19463, (int16_t)-16353,        (int16_t)-25656,
    (int16_t)-14547,        (int16_t)6813,  (int16_t)10386, (int16_t)8759,  (int16_t)-16961,        (int16_t)4,     (int16_t)32584, (int16_t)31189,
    (int16_t)-4781, (int16_t)20410, (int16_t)-28935,        (int16_t)-14663,        (int16_t)11505, (int16_t)23719, (int16_t)20938, (int16_t)-10777
};

const int16_t DataSet_1_16i::outputs::MRORV[64] = {
    (int16_t)11410, (int16_t)-528,  (int16_t)15128, (int16_t)-27538,        (int16_t)-14413,        (int16_t)-31569,        (int16_t)-18871,        (int16_t)-13437,
    (int16_t)4215,  (int16_t)-10559,        (int16_t)-5498, (int16_t)9685,  (int16_t)9682,  (int16_t)-30156,        (int16_t)-6208, (int16_t)23203,
    (int16_t)13653, (int16_t)3160,  (int16_t)30419, (int16_t)-32069,        (int16_t)25938, (int16_t)4552,  (int16_t)-23502,        (int16_t)14725,
    (int16_t)-9338, (int16_t)1169,  (int16_t)26002, (int16_t)-30533,        (int16_t)7522,  (int16_t)31036, (int16_t)14510, (int16_t)7417,
    (int16_t)23067, (int16_t)7654,  (int16_t)4080,  (int16_t)-10491,        (int16_t)27500, (int16_t)28077, (int16_t)-26090,        (int16_t)21493,
    (int16_t)7237,  (int16_t)17615, (int16_t)-6561, (int16_t)31241, (int16_t)15073, (int16_t)19463, (int16_t)-16353,        (int16_t)19940,
    (int16_t)-26909,        (int16_t)6813,  (int16_t)10386, (int16_t)18148, (int16_t)-16961,        (int16_t)16384, (int16_t)4073,  (int16_t)31189,
    (int16_t)-4781, (int16_t)-17841,        (int16_t)-13193,        (int16_t)-14663,        (int16_t)30870, (int16_t)23719, (int16_t)20938, (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::RORS[64] = {
    (int16_t)17810, (int16_t)-3843, (int16_t)24812, (int16_t)-11635,        (int16_t)-14413,        (int16_t)-3947, (int16_t)14025, (int16_t)-13437,
    (int16_t)-9151, (int16_t)15064, (int16_t)-8880, (int16_t)9685,  (int16_t)17594, (int16_t)20900, (int16_t)-24829,        (int16_t)27476,
    (int16_t)-22870,        (int16_t)22540, (int16_t)14189, (int16_t)28759, (int16_t)-27319,        (int16_t)569,   (int16_t)21638, (int16_t)24910,
    (int16_t)-9162, (int16_t)8338,  (int16_t)19634, (int16_t)24004, (int16_t)17324, (int16_t)-25028,        (int16_t)23665, (int16_t)9119,
    (int16_t)27459, (int16_t)-26505,        (int16_t)-32641,        (int16_t)-17696,        (int16_t)28045, (int16_t)-21067,        (int16_t)-11454,        (int16_t)-10929,
    (int16_t)8846,  (int16_t)-5991, (int16_t)-821,  (int16_t)-17148,        (int16_t)10076, (int16_t)1868,  (int16_t)2032,  (int16_t)-30276,
    (int16_t)29404, (int16_t)20109, (int16_t)18594, (int16_t)-30500,        (int16_t)-18441,        (int16_t)2048,  (int16_t)8701,  (int16_t)15023,
    (int16_t)-9561, (int16_t)-2231, (int16_t)-1650, (int16_t)-7332, (int16_t)-12526,        (int16_t)-18098,        (int16_t)10567, (int16_t)-10777
};

const int16_t DataSet_1_16i::outputs::MRORS[64] = {
    (int16_t)11410, (int16_t)-3843, (int16_t)24812, (int16_t)-27538,        (int16_t)-14413,        (int16_t)-31569,        (int16_t)-18871,        (int16_t)-13437,
    (int16_t)-9151, (int16_t)-10559,        (int16_t)-5498, (int16_t)9685,  (int16_t)9682,  (int16_t)20900, (int16_t)-24829,        (int16_t)23203,
    (int16_t)13653, (int16_t)22540, (int16_t)14189, (int16_t)-32069,        (int16_t)-27319,        (int16_t)4552,  (int16_t)-23502,        (int16_t)24910,
    (int16_t)-9162, (int16_t)1169,  (int16_t)26002, (int16_t)24004, (int16_t)7522,  (int16_t)-25028,        (int16_t)23665, (int16_t)7417,
    (int16_t)23067, (int16_t)-26505,        (int16_t)-32641,        (int16_t)-10491,        (int16_t)28045, (int16_t)28077, (int16_t)-26090,        (int16_t)-10929,
    (int16_t)8846,  (int16_t)17615, (int16_t)-6561, (int16_t)-17148,        (int16_t)15073, (int16_t)1868,  (int16_t)2032,  (int16_t)19940,
    (int16_t)-26909,        (int16_t)20109, (int16_t)18594, (int16_t)18148, (int16_t)-18441,        (int16_t)16384, (int16_t)4073,  (int16_t)15023,
    (int16_t)-9561, (int16_t)-17841,        (int16_t)-13193,        (int16_t)-7332, (int16_t)30870, (int16_t)-18098,        (int16_t)10567, (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::NEG[64] = {
    (int16_t)-11410,        (int16_t)30737, (int16_t)-1891, (int16_t)27538, (int16_t)-15774,        (int16_t)31569, (int16_t)18871, (int16_t)-23582,
    (int16_t)7666,  (int16_t)10559, (int16_t)5498,  (int16_t)-11945,        (int16_t)-9682, (int16_t)29406, (int16_t)2020,  (int16_t)-23203,
    (int16_t)-13653,        (int16_t)16286, (int16_t)17559, (int16_t)32069, (int16_t)21940, (int16_t)-4552, (int16_t)23502, (int16_t)-2675,
    (int16_t)7754,  (int16_t)-1169, (int16_t)-26002,        (int16_t)4574,  (int16_t)-7522, (int16_t)3612,  (int16_t)7286,  (int16_t)-7417,
    (int16_t)-23067,        (int16_t)15428, (int16_t)-1020, (int16_t)10491, (int16_t)-27755,        (int16_t)-28077,        (int16_t)26090, (int16_t)21890,
    (int16_t)-5233, (int16_t)-17615,        (int16_t)6561,  (int16_t)6107,  (int16_t)-15073,        (int16_t)-14944,        (int16_t)-16256,        (int16_t)-19940,
    (int16_t)26909, (int16_t)-29802,        (int16_t)-17682,        (int16_t)-18148,        (int16_t)16451, (int16_t)-16384,        (int16_t)-4073, (int16_t)10887,
    (int16_t)10946, (int16_t)17841, (int16_t)13193, (int16_t)-6887, (int16_t)-30870,        (int16_t)13707, (int16_t)-19001,        (int16_t)20674
};

const int16_t DataSet_1_16i::outputs::MNEG[64] = {
    (int16_t)11410, (int16_t)30737, (int16_t)-1891, (int16_t)-27538,        (int16_t)-15774,        (int16_t)-31569,        (int16_t)-18871,        (int16_t)-23582,
    (int16_t)7666,  (int16_t)-10559,        (int16_t)-5498, (int16_t)-11945,        (int16_t)9682,  (int16_t)29406, (int16_t)2020,  (int16_t)23203,
    (int16_t)13653, (int16_t)16286, (int16_t)17559, (int16_t)-32069,        (int16_t)21940, (int16_t)4552,  (int16_t)-23502,        (int16_t)-2675,
    (int16_t)7754,  (int16_t)1169,  (int16_t)26002, (int16_t)4574,  (int16_t)7522,  (int16_t)3612,  (int16_t)7286,  (int16_t)7417,
    (int16_t)23067, (int16_t)15428, (int16_t)-1020, (int16_t)-10491,        (int16_t)-27755,        (int16_t)28077, (int16_t)-26090,        (int16_t)21890,
    (int16_t)-5233, (int16_t)17615, (int16_t)-6561, (int16_t)6107,  (int16_t)15073, (int16_t)-14944,        (int16_t)-16256,        (int16_t)19940,
    (int16_t)-26909,        (int16_t)-29802,        (int16_t)-17682,        (int16_t)18148, (int16_t)16451, (int16_t)16384, (int16_t)4073,  (int16_t)10887,
    (int16_t)10946, (int16_t)-17841,        (int16_t)-13193,        (int16_t)-6887, (int16_t)30870, (int16_t)13707, (int16_t)-19001,        (int16_t)-20674
};

const int16_t DataSet_1_16i::outputs::ABS[64] = {
    (int16_t)11410, (int16_t)30737, (int16_t)1891,  (int16_t)27538, (int16_t)15774, (int16_t)31569, (int16_t)18871, (int16_t)23582,
    (int16_t)7666,  (int16_t)10559, (int16_t)5498,  (int16_t)11945, (int16_t)9682,  (int16_t)29406, (int16_t)2020,  (int16_t)23203,
    (int16_t)13653, (int16_t)16286, (int16_t)17559, (int16_t)32069, (int16_t)21940, (int16_t)4552,  (int16_t)23502, (int16_t)2675,
    (int16_t)7754,  (int16_t)1169,  (int16_t)26002, (int16_t)4574,  (int16_t)7522,  (int16_t)3612,  (int16_t)7286,  (int16_t)7417,
    (int16_t)23067, (int16_t)15428, (int16_t)1020,  (int16_t)10491, (int16_t)27755, (int16_t)28077, (int16_t)26090, (int16_t)21890,
    (int16_t)5233,  (int16_t)17615, (int16_t)6561,  (int16_t)6107,  (int16_t)15073, (int16_t)14944, (int16_t)16256, (int16_t)19940,
    (int16_t)26909, (int16_t)29802, (int16_t)17682, (int16_t)18148, (int16_t)16451, (int16_t)16384, (int16_t)4073,  (int16_t)10887,
    (int16_t)10946, (int16_t)17841, (int16_t)13193, (int16_t)6887,  (int16_t)30870, (int16_t)13707, (int16_t)19001, (int16_t)20674
};

const int16_t DataSet_1_16i::outputs::MABS[64] = {
    (int16_t)11410, (int16_t)30737, (int16_t)1891,  (int16_t)-27538,        (int16_t)15774, (int16_t)-31569,        (int16_t)-18871,        (int16_t)23582,
    (int16_t)7666,  (int16_t)-10559,        (int16_t)-5498, (int16_t)11945, (int16_t)9682,  (int16_t)29406, (int16_t)2020,  (int16_t)23203,
    (int16_t)13653, (int16_t)16286, (int16_t)17559, (int16_t)-32069,        (int16_t)21940, (int16_t)4552,  (int16_t)-23502,        (int16_t)2675,
    (int16_t)7754,  (int16_t)1169,  (int16_t)26002, (int16_t)4574,  (int16_t)7522,  (int16_t)3612,  (int16_t)7286,  (int16_t)7417,
    (int16_t)23067, (int16_t)15428, (int16_t)1020,  (int16_t)-10491,        (int16_t)27755, (int16_t)28077, (int16_t)-26090,        (int16_t)21890,
    (int16_t)5233,  (int16_t)17615, (int16_t)-6561, (int16_t)6107,  (int16_t)15073, (int16_t)14944, (int16_t)16256, (int16_t)19940,
    (int16_t)-26909,        (int16_t)29802, (int16_t)17682, (int16_t)18148, (int16_t)16451, (int16_t)16384, (int16_t)4073,  (int16_t)10887,
    (int16_t)10946, (int16_t)-17841,        (int16_t)-13193,        (int16_t)6887,  (int16_t)30870, (int16_t)13707, (int16_t)19001, (int16_t)-20674
};

const uint16_t DataSet_1_16i::outputs::ITOU[64] = {
    (uint16_t)11410,        (uint16_t)34799,        (uint16_t)1891, (uint16_t)37998,        (uint16_t)15774,        (uint16_t)33967,        (uint16_t)46665,        (uint16_t)23582,
    (uint16_t)57870,        (uint16_t)54977,        (uint16_t)60038,        (uint16_t)11945,        (uint16_t)9682, (uint16_t)36130,        (uint16_t)63516,        (uint16_t)23203,
    (uint16_t)13653,        (uint16_t)49250,        (uint16_t)47977,        (uint16_t)33467,        (uint16_t)43596,        (uint16_t)4552, (uint16_t)42034,        (uint16_t)2675,
    (uint16_t)57782,        (uint16_t)1169, (uint16_t)26002,        (uint16_t)60962,        (uint16_t)7522, (uint16_t)61924,        (uint16_t)58250,        (uint16_t)7417,
    (uint16_t)23067,        (uint16_t)50108,        (uint16_t)1020, (uint16_t)55045,        (uint16_t)27755,        (uint16_t)28077,        (uint16_t)39446,        (uint16_t)43646,
    (uint16_t)5233, (uint16_t)17615,        (uint16_t)58975,        (uint16_t)59429,        (uint16_t)15073,        (uint16_t)14944,        (uint16_t)16256,        (uint16_t)19940,
    (uint16_t)38627,        (uint16_t)29802,        (uint16_t)17682,        (uint16_t)18148,        (uint16_t)49085,        (uint16_t)16384,        (uint16_t)4073, (uint16_t)54649,
    (uint16_t)54590,        (uint16_t)47695,        (uint16_t)52343,        (uint16_t)6887, (uint16_t)30870,        (uint16_t)51829,        (uint16_t)19001,        (uint16_t)44862
};


