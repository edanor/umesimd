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

#ifndef UME_UNIT_TEST_MASKS_H_
#define UME_UNIT_TEST_MASKS_H_

int test_UME_SIMDMasks(bool supressMessages);

int test_UME_SIMDMask1(bool supressMessages);
int test_UME_SIMDMask2(bool supressMessages);
int test_UME_SIMDMask4(bool supressMessages);
int test_UME_SIMDMask8(bool supressMessages);
int test_UME_SIMDMask16(bool supressMessages);
int test_UME_SIMDMask32(bool supressMessages);
int test_UME_SIMDMask64(bool supressMessages);
int test_UME_SIMDMask128(bool supressMessages);

using namespace UME::SIMD;

int test_UME_SIMDMasks(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDMasks test";
    INIT_TEST(header, supressMessages);

    int failCount = test_UME_SIMDMask1(supressMessages);
    failCount += test_UME_SIMDMask2(supressMessages);
    failCount += test_UME_SIMDMask4(supressMessages);
    failCount += test_UME_SIMDMask8(supressMessages);
    failCount += test_UME_SIMDMask16(supressMessages);
    failCount += test_UME_SIMDMask32(supressMessages);
    failCount += test_UME_SIMDMask64(supressMessages);
    failCount += test_UME_SIMDMask128(supressMessages);

    return failCount;
}

int test_UME_SIMDMask1(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDMask1 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDMask1 mask;
        CHECK_CONDITION(true, "ZERO-CONSTR()");
    }
    
    genericMaskTest<SIMDMask1, 1, DataSet_1_mask>();

    return g_failCount;
}

int test_UME_SIMDMask2(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDMask2 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDMask2 mask;
        CHECK_CONDITION(true, "ZERO-CONSTR()");
    }
    
    genericMaskTest<SIMDMask2, 2, DataSet_1_mask>();


    return g_failCount;
}

int test_UME_SIMDMask4(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDMask4 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDMask4 mask;
        CHECK_CONDITION(true, "ZERO-CONSTR()");
    }
    {
        bool arr[4] = {true, false, false, true};
        SIMDMask4 mask(arr);
        CHECK_CONDITION(mask[0] == true  && mask[1] == false &&
                        mask[2] == false && mask[3] == true, "LOAD-CONSTR()");
    }
    {  
        SIMD4_64f vec0(1.0, 2.0, 3.0, 4.0);
        SIMD4_64f vec1(2.0, 1.0, 0.0, 5.0);

        SIMDMask4 mask;
        
        mask = vec0.cmpgt(vec1);

        CHECK_CONDITION(mask[0] == false && mask[1] == true &&
                        mask[2] == true  &&  mask[3] == false, "mask compatibility: 64f -> 32u");
    }
    {
        SIMDMask4 mask0(true, true, false, false);
        SIMDMask4 mask1(false, true, false, true);
        mask0.landa(mask1);
        CHECK_CONDITION(mask0[0] == false && mask0[1] == true &&
                        mask0[2] == false && mask0[3] == false, "LANDA");
    }
    {
        SIMDMask4 mask0(true, true, false, false);
        SIMDMask4 mask1(false, true, false, true);
        mask0 &= mask1;
        CHECK_CONDITION(mask0[0] == false && mask0[1] == true &&
                        mask0[2] == false && mask0[3] == false, "LANDA(operator&=)");
    }
    {
        SIMDMask4 mask0(true, true, false, false);
        SIMDMask4 mask1(false, true, false, true);
        mask0.lora(mask1);
        CHECK_CONDITION(mask0[0] == true  && mask0[1] == true &&
                        mask0[2] == false && mask0[3] == true, "LORA");
    }
    {
        SIMDMask4 mask0(true, true, false, false);
        SIMDMask4 mask1(false, true, false, true);
        mask0 |= mask1;
        CHECK_CONDITION(mask0[0] == true  && mask0[1] == true &&
                        mask0[2] == false && mask0[3] == true, "LORA(operator|=)");
    }
    {
        SIMDMask4 mask0(true, false, false, true);
        SIMDMask4 mask1 = mask0.lnot();
        CHECK_CONDITION(mask1[0] == false && mask1[1] == true &&
                        mask1[2] == true  && mask1[3] == false, "LNOT");
    }
    {
        SIMDMask4 mask0(true, false, false, true);
        SIMDMask4 mask1 = !mask0;
        CHECK_CONDITION(mask1[0] == false && mask1[1] == true &&
                        mask1[2] == true  && mask1[3] == false, "LNOT(operator!)");
    }
    {
        SIMDMask4 mask0(true, false, false, true);
        SIMDMask4 mask1(false, true, false, true);
        mask1.assign(mask0);
        CHECK_CONDITION(mask1[0] == true && mask1[1] == false &&
                        mask1[2] == false && mask1[3] == true, "ASSIGN");
    }
    {
        SIMDMask4 mask0(true, false, false, true);
        SIMDMask4 mask1(false, true, false, true);
        mask1 = mask0;
        CHECK_CONDITION(mask1[0] == true && mask1[1] == false &&
                        mask1[2] == false && mask1[3] == true, "ASSIGN(operator=)");
    }
    {
        SIMDMask4 mask0(true, false, false, true);
        SIMDMask4 mask1(false, false, false, false);
        bool b0 = mask0.hlor();
        bool b1 = mask1.hlor();
        CHECK_CONDITION(b0 == true && b1 == false, "HLOR");
    }
    {
        SIMDMask4 mask0(true, false, false, true);
        SIMDMask4 mask1(false, false, false, false);
        SIMDMask4 mask2(true, true, true, true);
        bool b0 = mask0.hland();
        bool b1 = mask1.hland();
        bool b2 = mask2.hland();
        CHECK_CONDITION(b0 == false && b1 == false && b2 == true, "HLAND");
    }
    
    genericMaskTest<SIMDMask4, 4, DataSet_1_mask>();

    return g_failCount;
}

int test_UME_SIMDMask8(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDMask8 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDMask16 mask;
        CHECK_CONDITION(true, "ZERO-CONSTR()");
    }
    {
        SIMDMask8 mask0(true);
        bool res = true;
        for(uint32_t i = 0; i < mask0.length(); i++) res &= mask0[i];
        CHECK_CONDITION(res == true, "SET-CONSTR");
    }
    {
        SIMDMask8 mask(true, false, false, true, true, true, true, false);
        CHECK_CONDITION(mask[1] == false && mask[6] == true && mask[7] == false, "FULL-CONSTR");
    }
    {
        SIMDMask8 mask0(true);
        SIMDMask8 mask1(true, false, false, true, true, true, true, false);
        mask0.assign(mask1);
        CHECK_CONDITION(mask0[1] == false && mask0[6] == true && mask0[7] == false, "ASSIGN");
    }
    {
        SIMDMask8 mask0(true, false, false, true, true, false, false, true);
        SIMDMask8 mask1(true, false, false, false, false, true, false, true);
        SIMDMask8 mask2;
        mask2 = mask0.land(mask1);
        CHECK_CONDITION(mask2[0] == true && mask2[3] == false && mask2[5] == false && mask2[7] == true, "LAND");
    }
    {
        SIMDMask8 mask0(true, false, false, true, true, false, false, true);
        SIMDMask8 mask1(true, false, false, false, false, true, false, true);
        SIMDMask8 mask2;
        mask2 = mask0 & mask1;
        CHECK_CONDITION(mask2[0] == true && mask2[3] == false && mask2[5] == false && mask2[7] == true, "LAND(operator&)");
    }
    
    genericMaskTest<SIMDMask8, 8, DataSet_1_mask>();

    return g_failCount;
}

int test_UME_SIMDMask16(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDMask16 test";
    INIT_TEST(header, supressMessages);

    {
        SIMDMask16 mask;
        CHECK_CONDITION(true, "ZERO-CONSTR()");
    }
    
    genericMaskTest<SIMDMask16, 16, DataSet_1_mask>();

    return g_failCount;
}

int test_UME_SIMDMask32(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDMask32 test";
    INIT_TEST(header, supressMessages);
    {
        SIMDMask32 mask;
        CHECK_CONDITION(true, "ZERO-CONSTR()");
    }
    
    genericMaskTest<SIMDMask32, 32, DataSet_1_mask>();

    return g_failCount;
}

int test_UME_SIMDMask64(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDMask64 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDMask64 mask;
        CHECK_CONDITION(true, "ZERO-CONSTR()");
    }

    {
        std::random_device rd;
        std::mt19937 gen(rd());

        bool input[64];

        for(int i = 0; i < 64; i++) {
            input[i] = randomValue<bool>(gen);
        }

        SIMDMask64 mask(
            input[63],  input[62],  input[61],  input[60],  input[59],  input[58],  input[57],  input[56],
            input[55],  input[54],  input[53],  input[52],  input[51],  input[50],  input[49],  input[48],
            input[47],  input[46],  input[45],  input[44],  input[43],  input[42],  input[41],  input[40],
            input[39],  input[38],  input[37],  input[36],  input[35],  input[34],  input[33],  input[32],
            input[31],  input[30],  input[29],  input[28],  input[27],  input[26],  input[25],  input[24],
            input[23],  input[22],  input[21],  input[20],  input[19],  input[18],  input[17],  input[16],
            input[15],  input[14],  input[13],  input[12],  input[11],  input[10],  input[9],   input[8],
            input[7],   input[6],   input[5],   input[4],   input[3],   input[2],   input[1],   input[0]);

        bool raw[64];
        mask.store(raw);
        bool cond = true;
        for(int i = 0; i < 64; i++) {
            if(raw[i] != input[63 - i]) {
                cond = false;
                break;
            }
        }

        CHECK_CONDITION(cond, "FULL-CONSTR");
    }

    genericMaskTest<SIMDMask64, 64, DataSet_1_mask>();

    return g_failCount;
}

int test_UME_SIMDMask128(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDMask128 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDMask128 mask;
        CHECK_CONDITION(true, "ZERO-CONSTR()");
    }

    {
        std::random_device rd;
        std::mt19937 gen(rd());

        bool input[128];

        for(int i = 0; i < 128; i++) {
            input[i] = randomValue<bool>(gen);
        }

        SIMDMask128 mask(
            input[127], input[126], input[125], input[124], input[123], input[122], input[121], input[120],
            input[119], input[118], input[117], input[116], input[115], input[114], input[113], input[112],
            input[111], input[110], input[109], input[108], input[107], input[106], input[105], input[104],
            input[103], input[102], input[101], input[100], input[99],  input[98],  input[97],  input[96], 
            input[95],  input[94],  input[93],  input[92],  input[91],  input[90],  input[89],  input[88],
            input[87],  input[86],  input[85],  input[84],  input[83],  input[82],  input[81],  input[80],
            input[79],  input[78],  input[77],  input[76],  input[75],  input[74],  input[73],  input[72],
            input[71],  input[70],  input[69],  input[68],  input[67],  input[66],  input[65],  input[64],
            input[63],  input[62],  input[61],  input[60],  input[59],  input[58],  input[57],  input[56],
            input[55],  input[54],  input[53],  input[52],  input[51],  input[50],  input[49],  input[48],
            input[47],  input[46],  input[45],  input[44],  input[43],  input[42],  input[41],  input[40],
            input[39],  input[38],  input[37],  input[36],  input[35],  input[34],  input[33],  input[32],
            input[31],  input[30],  input[29],  input[28],  input[27],  input[26],  input[25],  input[24],
            input[23],  input[22],  input[21],  input[20],  input[19],  input[18],  input[17],  input[16],
            input[15],  input[14],  input[13],  input[12],  input[11],  input[10],  input[9],   input[8],
            input[7],   input[6],   input[5],   input[4],   input[3],   input[2],   input[1],   input[0]);

        bool raw[128];
        mask.store(raw);
        bool cond = true;
        for(int i = 0; i < 128; i++) {
            if(raw[i] != input[127 - i]) {
                cond = false;
                break;
            }
        }

        CHECK_CONDITION(cond, "FULL-CONSTR");
    }

    genericMaskTest<SIMDMask128, 128, DataSet_1_mask>();

    return g_failCount;
}

#endif
