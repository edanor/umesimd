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

#ifndef UME_UNIT_TEST_SIMD_128B_H_
#define UME_UNIT_TEST_SIMD_128B_H_

#include "UMEUnitTestCommon.h"

int test_UME_SIMD128b(bool supressMessages);

int test_UME_SIMD16_8(bool supressMessages);
int test_UME_SIMD16_8u(bool supressMessages);
int test_UME_SIMD16_8i(bool supressMessages);

int test_UME_SIMD8_16(bool supressMessages);
int test_UME_SIMD8_16u(bool supressMessages);
int test_UME_SIMD8_16i(bool supressMessages);

int test_UME_SIMD4_32(bool supressMessages);
int test_UME_SIMD4_32u(bool supressMessages);
int test_UME_SIMD4_32i(bool supressMessages);
int test_UME_SIMD4_32f(bool supressMessages);

int test_UME_SIMD2_64(bool supressMessages);
int test_UME_SIMD2_64u(bool supressMessages);
int test_UME_SIMD2_64i(bool supressMessages);
int test_UME_SIMD2_64f(bool supressMessages);

using namespace UME::SIMD;
   
int test_UME_SIMD128b(bool supressMessages) 
{
    int simd16_8_res = test_UME_SIMD16_8(supressMessages);
    int simd8_16_res = test_UME_SIMD8_16(supressMessages);
    int simd4_32_res = test_UME_SIMD4_32(supressMessages);
    int simd2_64_res = test_UME_SIMD2_64(supressMessages);

    return simd16_8_res + simd8_16_res + simd4_32_res + simd2_64_res;
}

int test_UME_SIMD16_8(bool supressMessages)
{
    int fail_u = test_UME_SIMD16_8u(supressMessages);
    int fail_i = test_UME_SIMD16_8i(supressMessages);

    return fail_u + fail_i;
}

int test_UME_SIMD8_16(bool supressMessages) {

    int fail_u = test_UME_SIMD8_16u(supressMessages);
    int fail_i = test_UME_SIMD8_16i(supressMessages);

    return fail_u + fail_i;
}

int test_UME_SIMD4_32(bool supressMessages)
{
    int fail_u = test_UME_SIMD4_32u(supressMessages);
    int fail_i = test_UME_SIMD4_32i(supressMessages);
    int fail_f = test_UME_SIMD4_32f(supressMessages);

    return fail_u + fail_i + fail_f;
}

int test_UME_SIMD2_64(bool supressMessages)
{
    int fail_u = test_UME_SIMD2_64u(supressMessages);
    int fail_i = test_UME_SIMD2_64i(supressMessages);
    int fail_f = test_UME_SIMD2_64f(supressMessages);

    return fail_u + fail_i + fail_f;
}

// ****************************************************************************
// * Test functions for specific vector types
// ****************************************************************************

int test_UME_SIMD16_8u(bool supressMessages) 
{
    char header[] = "UME::SIMD::SIMD16_8u test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD16_8u vec1;
        //CHECK_CONDITION(true, "ZERO-CONSTR"); 
    }
    
    {
        SIMD16_8u vec0(0, 1, 2, 3,  4, 5, 6, 7,  8, 9, 10, 11,  12, 13, 14 ,255);
        SIMD16_8u vec1(4);
        SIMD16_8u vec2;
        
        vec2 = vec0.lsh(vec1);
        CHECK_CONDITION(vec2[0] == 0 && vec2[1] == 16, "LSHV");
    }
    {
        SIMD16_8u vec0(0, 1, 2, 3,  4, 5, 6, 7,  8, 9, 10, 11,  12, 13, 14 ,255);
        SIMD16_8u vec1;
        vec1 = vec0.lsh(4);
        CHECK_CONDITION(vec1[0] == 0 && vec1[1] == 16 && vec1[15] == 240, "LSHS");
    }
    {
        SIMD16_8u vec0(0, 1, 2, 3,  4, 5, 6, 7,  8, 9, 10, 11,  12, 13, 14 ,248);
        SIMD16_8u vec1;

        vec1 = vec0.lsh(4);
        CHECK_CONDITION(vec1[0] == 0 && vec1[1] == 16 && vec1[15] == 128, "RSHS");
    }
    
    return g_failCount;
}

int test_UME_SIMD16_8i(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD16_8i test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD16_8i vec0; 
        CHECK_CONDITION(true, "ZERO-CONSTR"); 
    }
    
    return g_failCount;
}

int test_UME_SIMD8_16u(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD8_16u test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD8_16u vec3;
        CHECK_CONDITION(true, "ZERO-CONSTR"); 
    }
    {
        SIMD8_16u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_16u vec1(15);

        vec1 = vec0;
        CHECK_CONDITION(vec1[0] == 1 && vec1[1] == 2, "operator=");
    }
    {
        SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        int16_t vals[] = {1, 2, 3, 4, 5, 6, 7, 8};
        bool res = true;
        vec0.rsha(2);
        for(uint32_t i = 0; i < 8; i++)if(vec0[i] != vals[i]){res = false; break;}
        CHECK_CONDITION(res, "RSHSA");
    }
    {
        SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        SIMD8_16u vec1(1, 2, 3, 4, 5, 6, 7, 8);
        vec0.rsha(2);
        SIMDMask8 mask; 
        mask = vec0.cmpeq(vec1); // 0xFF
        CHECK_CONDITION(mask[0] == true && mask[7] == true, "CMPEQV");
    }
    {
        SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        SIMD8_16u vec1(1, 2, 3, 4, 5, 6, 7, 8);
        vec0.rsha(2);
        SIMDMask8 mask; 
        mask = vec0 == vec1; // 0xFF
        CHECK_CONDITION(mask[0] == true && mask[7] == true, "CMPEQV(operator==)");
    }
    {
        SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        SIMD8_16u vec1(1, 2, 3, 4, 5, 6, 7, 8);
        vec0.rsha(2);
        SIMDMask8 mask;
        mask = vec0.cmpne(vec1); // 0x00
        CHECK_CONDITION(mask[0] == false && mask[7] == false, "CMPNEV");
    }
    {
        SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        SIMD8_16u vec1(1, 2, 3, 4, 5, 6, 7, 8);
        vec0.rsha(2);
        SIMDMask8 mask;
        mask = vec0 != vec1; // 0x00
        CHECK_CONDITION(mask[0] == false && mask[7] == false, "CMPNEV(operator!=)");
    }
    {
        SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        SIMD8_16u vec1(1, 2, 3, 5, 5, 6, 8, 8);
        vec0.rsha(2);
        SIMDMask8 mask;
        mask = vec1.cmpgt(vec0); // 0x48
        CHECK_CONDITION(mask[3] == true && mask[5] == false, "CMPGTV");
    }
    {
        SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        SIMD8_16u vec1(1, 2, 3, 5, 5, 6, 8, 8);
        vec0.rsha(2);
        SIMDMask8 mask;
        mask = vec1 > vec0; // 0x48
        CHECK_CONDITION(mask[3] == true && mask[5] == false, "CMPGTV(operator>)");
    }
    {
        SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        SIMD8_16u vec1(1, 2, 3, 5, 5, 6, 8, 8);
        vec0.rsha(2);
        SIMDMask8 mask;
        mask = vec0.cmplt(vec1); // 0x48
        CHECK_CONDITION(mask[3] == true && mask[5] == false, "CMPLTV");
    }
    {
        SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        SIMD8_16u vec1(1, 2, 3, 5, 5, 6, 8, 8);
        vec0.rsha(2);
        SIMDMask8 mask;
        mask = vec0 < vec1; // 0x48
        CHECK_CONDITION(mask[3] == true && mask[5] == false, "CMPLTV(operator<)");
    }
    {
        SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        SIMD8_16u vec1(1, 2, 3, 3, 5, 5, 8, 8);
        vec0.rsha(2);
        SIMDMask8 mask;
        mask = vec0.cmpge(vec1); // 0xBF
        CHECK_CONDITION(mask[2] == true && mask[6] == false, "CMPGEV");
    }
    {
        SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        SIMD8_16u vec1(1, 2, 3, 3, 5, 5, 8, 8);
        vec0.rsha(2);
        SIMDMask8 mask;
        mask = vec0 >= vec1; // 0xBF
        CHECK_CONDITION(mask[2] == true && mask[6] == false, "CMPGEV(operator>=)");
    }
    {
        SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        SIMD8_16u vec1(1, 2, 3, 3, 5, 5, 8, 8);
        vec0.rsha(2);
        SIMDMask8 mask;
        mask = vec0.cmple(vec1); // 0xD7
        CHECK_CONDITION(mask[3] == false && mask[6] == true, "CMPLEV");
    }
    {
        SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        SIMD8_16u vec1(1, 2, 3, 3, 5, 5, 8, 8);
        vec0.rsha(2);
        SIMDMask8 mask;
        mask = vec0 <= vec1; // 0xD7
        CHECK_CONDITION(mask[3] == false && mask[6] == true, "CMPLEV(operator<=)");
    }
    {
        SIMD8_16u vec0(0xF2F1);
        SIMD8_16u vec1(0x2F1F);
        SIMD8_16u vec2;
        vec2 = vec0.band(vec1);
        CHECK_CONDITION(vec2[0] == 0x2211, "BANDV");
    }
    {
        SIMD8_16u vec0(0xF2F1);
        SIMD8_16u vec1(0x2F1F);
        SIMD8_16u vec2;
        vec2 = vec0 & vec1;
        CHECK_CONDITION(vec2[0] == 0x2211, "ANDV(operator&)");
    }
    {
        SIMD8_16u vec0(0xF2F1);
        SIMD8_16u vec1(0x2F1F);
        vec1.banda(vec0);
        CHECK_CONDITION(vec1[0] == 0x2211, "BANDVA");
    }
    {
        SIMD8_16u vec0(0xF2F1);
        SIMD8_16u vec1(0x2F1F);
        vec1 &= vec0;
        CHECK_CONDITION(vec1[0] == 0x2211, "ANDVA(operator&=)");
    }
    {
        SIMD8_16u vec0(0x7281);
        SIMD8_16u vec1(0x2314);
        SIMD8_16u vec2;
        vec2 = vec0.bor(vec1);
        CHECK_CONDITION(vec2[0] == 0x7395, "BORV");
    }
    {
        SIMD8_16u vec0(0x7281);
        SIMD8_16u vec1(0x2314);
        SIMD8_16u vec2;
        vec2 = vec0 | vec1;
        CHECK_CONDITION(vec2[0] == 0x7395, "BORV(operator|)");
    }
    {
        SIMD8_16u vec0(0x7281);
        SIMD8_16u vec1(0x2314);
        vec0.bora(vec1);
        CHECK_CONDITION(vec0[0] == 0x7395, "BORVA");
    }
    {
        SIMD8_16u vec0(0x7281);
        SIMD8_16u vec1(0x2314);
        vec0 |= vec1;
        CHECK_CONDITION(vec0[0] == 0x7395, "BORVA(operator|=)");
    }
    {
        SIMD8_16u vec0(0x7281);
        SIMD8_16u vec1(0x2314);
        vec0.bxora(vec1);
        CHECK_CONDITION(vec0[0] == 0x5195, "BXORVA");
    }
    {
        SIMD8_16u vec0(0x7281);
        SIMD8_16u vec1(0x2314);
        vec0 ^= vec1;
        CHECK_CONDITION(vec0[0] == 0x5195, "BXORVA(operator^=)");
    }
    {
        SIMD8_16u vec0(0x7281);
        SIMD8_16u vec1(0x2314);
        vec0 = vec1.bnot();
        CHECK_CONDITION(vec0[0] == 0xDCEB, "BNOT");
    }
    {
        SIMD8_16u vec0(7);
        SIMD8_16u vec1(24);
        SIMDMask8 mask2(false, false, false, false, true, true, true, true);
        SIMD8_16u vec3 = vec0.blend(mask2, vec1);
        CHECK_CONDITION(vec3[0] == 24 && vec3[4] == 7, "MBLENDV");
    }
    {
        SIMD8_16u vec0(10, 20, 30, 40, 50, 60, 70, 80);
        uint32_t swizzleMask[8] = {1, 6, 7, 4, 7, 6, 7, 0};
        SIMDSwizzle8 sMask(swizzleMask);
        SIMD8_16u vec2 = vec0.swizzle(sMask);
        SIMD8_16u vec3(20, 70, 80, 50, 80, 70, 80, 10);
        
        CHECK_CONDITION(vec3.cmpe(vec2) == true, "SWIZZLE");
    }
    {
        SIMD8_16u vec0(3);
        uint16_t res = vec0.hadd();
        CHECK_CONDITION(res == 24, "HADD");
    }
    {
        SIMD8_16u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_16u vec1(8, 7, 6, 5, 4, 3, 2, 1);
        SIMD8_16u vec2;
        vec2 = vec0.max(vec1);
        CHECK_CONDITION(vec2[0] == 8 && vec2[6] == 7, "MAXV");
    }
    {
        SIMD8_16u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_16u vec1(8, 7, 6, 5, 4, 3, 2, 1);
        SIMD8_16u vec2;
        vec2 = vec1.min(vec0);
        CHECK_CONDITION(vec2[0] == 1 && vec2[6] == 2, "MINV");
    }

    return g_failCount;
}

int test_UME_SIMD8_16i(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD8_16i test";
    INIT_TEST(header, supressMessages);

    
    {
        SIMD8_16i vec2;
        CHECK_CONDITION(true, "ZERO-CONSTR()"); 
    }
    {
        SIMD8_16i vec0(1, -2, 3, 4, 5, 6, 7, 8);
        SIMD8_16i vec1(15);

        vec1 = vec0;
        CHECK_CONDITION(vec1[0] == 1 && vec1[1] == -2, "operator=");
    }
    {
        SIMD8_16i vec0(1, -2, 3, 4, 5, 6, 7, 8);
        int16_t vals[] = { 100, 200, 300, 400, 500, 600, 700, 800 };
        vec0.load((const int16_t *)vals);
        bool res = true;
        for(uint32_t i = 0; i < 8; i++)
        {
            if(vec0[i] != vals[i])
            {
                res = false;
                break;
            }
        }
        CHECK_CONDITION(res, "LOAD");
    }
    {
        SIMD8_16i vec0(1, -2, 3, 4, 5, 6, 7, 8);
        alignas(16) int16_t vals[]  = { 100, 200, 300, 400, 500, 600, 700, 800 };
        vec0.loada(vals);
        bool res = true;
        for(uint32_t i = 0; i < 8; i++)
        {
            if(vec0[i] != vals[i])
            {
                res = false;
                break;
            }
        }
        CHECK_CONDITION(res, "LOADA");
    }
    {
        SIMD8_16i vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_16i vec1((int16_t)0);
        int16_t vals[] = { 4, 8, 12, 16, 20, 24, 28, 32 };
        bool res = true;
        vec1 = vec0.lsh(2);
        for(uint32_t i = 0; i < 8; i++)if(vec1[i] != vals[i]){res = false; break;}
        CHECK_CONDITION(res, "LSHS");
    }
    {
        SIMD8_16i vec0(1, 2, 3, 4, 5, 6, 7, 8);
        int16_t vals[] = { 4, 8, 12, 16, 20, 24, 28, 32 };
        bool res = true;
        vec0.lsha(2);
        for(uint32_t i = 0; i < 8; i++)if(vec0[i] != vals[i]){res = false; break;}
        CHECK_CONDITION(res, "LSHSA");
    }
    {
        SIMD8_16i vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_16i vec1(3, 4, 5, 6, 12, 14, 16, 18);
        SIMD8_16i vec2;
        vec2 = vec0.rol(vec1);
        
        CHECK_CONDITION(vec2[0] == 8 && vec2[1] == 32 && vec2[5] == -32767 && vec2[7] == 32, "ROLV");
    }
    {
        SIMD8_16i vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_16i vec1(3, 4, 5, 6, 12, 14, 16, 18);
        SIMD8_16i vec2;
        SIMDMask8 mask = SIMDMask8(true, true, false, false, false, false, false, true);
        vec2 = vec0.rol(mask, vec1);
        
        CHECK_CONDITION(vec2[0] == 8 && vec2[1] == 32 && vec2[5] == 6 && vec2[7] == 32, "MROLV");
    }
    {
        SIMD8_16i vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_16i vec1;
        vec1 = vec0.rol(3);
        
        CHECK_CONDITION(vec1[0] == 8 && vec1[7] == 64, "ROLS");
    }
    {
        SIMD8_16i vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_16i vec1;
        SIMDMask8 mask = SIMDMask8(true, true, false, false, false, false, false, false);
        vec1 = vec0.rol(mask, 3);
        
        CHECK_CONDITION(vec1[0] == 8 && vec1[2] == 3 && vec1[7] == 8, "MROLS");
    }
    {
        SIMD8_16i vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_16i vec1(3, 4, 5, 6, 12, 14, 16, 18);
        vec0.rola(vec1);
        
        CHECK_CONDITION(vec0[0] == 8 && vec0[1] == 32 && vec0[5] == -32767 && vec0[7] == 32, "ROLVA");
    }
    {
        SIMD8_16i vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_16i vec1(3, 4, 5, 6, 12, 14, 16, 18);
        SIMDMask8 mask = SIMDMask8(true, true, false, false, false, false, false, true);
        vec0.rola(mask, vec1);
        
        CHECK_CONDITION(vec0[0] == 8 && vec0[1] == 32 && vec0[5] == 6 && vec0[7] == 32, "MROLVA");
    }
    {
        SIMD8_16i vec0(1, 2, 3, 4, 5, 6, 7, 8);
        vec0.rola(3);
        
        CHECK_CONDITION(vec0[0] == 8 && vec0[7] == 64, "ROLSA");
    }
    {
        SIMD8_16i vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMDMask8 mask = SIMDMask8(true, true, false, false, false, false, false, false);
        vec0.rola(mask, 3);
        
        CHECK_CONDITION(vec0[0] == 8 && vec0[2] == 3 && vec0[7] == 8, "MROLSA");
    }
    {
        SIMD8_16i vec0(-1);
        SIMD8_16i vec1 = vec0.abs();

        CHECK_CONDITION(vec1[0] == vec1[7] == 1, "ABS");
    }

    return g_failCount;
}

int test_UME_SIMD4_32u(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD4_32u test";
    
    INIT_TEST(header, supressMessages);
    
    genericUintTest<SIMD4_32u, uint32_t, SIMDMask4, 4, DataSet_1_32u>();
    {
        SIMD4_32u vec1;
        CHECK_CONDITION(true, "ZERO-CONSTR");  
    }
    {  
        SIMD4_32u vec1(8);
        CHECK_CONDITION(vec1[3] == 8,  "SET-CONSTR"); 

    }
    {  
        SIMD4_32u vec1(8, 4, 2, 1);
        CHECK_CONDITION(vec1[0] == 8 && vec1[2] == 2,  "FULL-CONSTR"); 
    }
    {
        SIMD4_32i vec0(8);
        SIMD4_32u vec1;

        vec1 = SIMD4_32u(vec0);
        CHECK_CONDITION(vec1[2] == 8, "ITOU");
    }
    {
        CHECK_CONDITION(SIMD4_32u::length() == 4, "LENGTH");
    }
    {
        CHECK_CONDITION(SIMD4_32u::alignment() == 16, "ALIGNMENT");
    }
    {
        SIMD4_32u vec0(1, 2, 3, 4);
        SIMD4_32u vec1(15);

        vec1.assign(vec0);
        CHECK_CONDITION(vec1[0] == 1 && vec1[1] == 2, "ASSIGNV");
    }      
    {
        alignas(16) uint32_t arr[4] = { 1, 3, 8, 321};
        SIMD4_32u vec0(42);
        vec0.loada(arr);

        CHECK_CONDITION(vec0[0] == 1 && vec0[3] == 321, "LOADA");
    }
    {
        uint32_t arr[4] = { 1, 3, 9, 124};
        SIMD4_32u vec0(9, 32, 28, 1256);
        vec0.store(arr);
        CHECK_CONDITION(arr[0] == 9 && arr[3] == 1256, "STORE");
    }
    {
        alignas(16) uint32_t arr[4] = { 1, 3, 9, 124};
        SIMD4_32u vec0(9, 32, 28, 1256);
        vec0.storea(arr);
        CHECK_CONDITION(arr[0] == 9 && arr[3] == 1256, "STOREA");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(3, 14, 28, 60);
        SIMD4_32u vec2;
        vec2 = vec0.add(vec1);
        CHECK_CONDITION(vec2[0] == 12 && vec2[1] == 22 && vec2[2] == 35 && vec2[3] == 66, "ADDV");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(3, 14, 28, 60);
        SIMD4_32u vec2;
        vec2 = vec0 + vec1;
        CHECK_CONDITION(vec2[0] == 12 && vec2[1] == 22 && vec2[2] == 35 && vec2[3] == 66, "ADDV(operator+)");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(3, 14, 28, 60);
        SIMD4_32u vec2;
        SIMDMask4 mask(true, false, true, true);
        vec2 = vec0.add(mask, vec1);
        CHECK_CONDITION(vec2[0] == 12 && vec2[1] == 8 && vec2[2] == 35 && vec2[3] == 66, "MADDV");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t b = 34;
        SIMD4_32u vec2;
        vec2 = vec0.add(b);
        CHECK_CONDITION(vec2[0] == 43 && vec2[1] == 42 && vec2[2] == 41 && vec2[3] == 40, "ADDS");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t b = 34;
        SIMD4_32u vec2;
        SIMDMask4 mask(true, false, true, true);
        vec2 = vec0.add(mask, b);
        CHECK_CONDITION(vec2[0] == 43 && vec2[1] == 8 && vec2[2] == 41 && vec2[3] == 40, "MADDS");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(3, 14, 28, 60);
        vec0.adda(vec1);
        CHECK_CONDITION(vec0[0] == 12 && vec0[1] == 22 && vec0[2] == 35 && vec0[3] == 66, "ADDVA");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(3, 14, 28, 60);
        vec0 += vec1;
        CHECK_CONDITION(vec0[0] == 12 && vec0[1] == 22 && vec0[2] == 35 && vec0[3] == 66, "ADDVA(operator+=)");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 12;
        vec0.adda(val1);
        CHECK_CONDITION(vec0[0] == 21 && vec0[1] == 20 && vec0[2] == 19 && vec0[3] == 18, "ADDSA");
    }
    {
        SIMD4_32u vec0(0xFFFFFFFF, 0xFFFFFF00, 0x00000003, 0x12345678);
        SIMD4_32u vec1(0x00000005, 0x00000100, 0x15166345, 0xFFFF0000);
        SIMD4_32u vec2;
        vec2 = vec0.sadd(vec1);
        CHECK_CONDITION(vec2[0] == 0xFFFFFFFF && vec2[1] == 0xFFFFFFFF &&
                        vec2[2] == 0x15166348 && vec2[3] == 0xFFFFFFFF, "SADDV");
    }
    {
        SIMD4_32u vec0(0xFFFFFFFF, 0xFFFFFF00, 0x00000003, 0x12345678);
        SIMD4_32u vec1(0x00000005, 0x00000100, 0x15166345, 0xFFFF0000);
        SIMD4_32u vec2;
        SIMDMask4 mask(true, false, true, true);
        vec2 = vec0.sadd(mask, vec1);
        CHECK_CONDITION(vec2[0] == 0xFFFFFFFF && vec2[1] == 0xFFFFFF00 &&
                        vec2[2] == 0x15166348 && vec2[3] == 0xFFFFFFFF, "MSADDV");
    }
    {
        SIMD4_32u vec0(0xFFFFFFFF, 0xFFFFFF00, 0x00000003, 0x12345678);
        uint32_t val1 = 0x00000100;
        SIMD4_32u vec2;
        vec2 = vec0.sadd(val1);
        CHECK_CONDITION(vec2[0] == 0xFFFFFFFF && vec2[1] == 0xFFFFFFFF &&
                        vec2[2] == 0x00000103 && vec2[3] == 0x12345778, "SADDS");
    }
    {
        SIMD4_32u vec0(0xFFFFFFFF, 0xFFFFFF00, 0x00000003, 0x12345678);
        uint32_t val1 = 0x00000100;
        SIMD4_32u vec2;
        SIMDMask4 mask(true, false, true, true);
        vec2 = vec0.sadd(mask, val1);
        CHECK_CONDITION(vec2[0] == 0xFFFFFFFF && vec2[1] == 0xFFFFFF00 &&
                        vec2[2] == 0x00000103 && vec2[3] == 0x12345778, "MSADDS");
    }
    {
        SIMD4_32u vec0(0xFFFFFFFF, 0xFFFFFF00, 0x00000003, 0x12345678);
        SIMD4_32u vec1(0x00000005, 0x00000100, 0x15166345, 0xFFFF0000);
        vec0.sadda(vec1);
        CHECK_CONDITION(vec0[0] == 0xFFFFFFFF && vec0[1] == 0xFFFFFFFF &&
                        vec0[2] == 0x15166348 && vec0[3] == 0xFFFFFFFF, "SADDVA");
    }
    {
        SIMD4_32u vec0(0xFFFFFFFF, 0xFFFFFF00, 0x00000003, 0x12345678);
        SIMD4_32u vec1(0x00000005, 0x00000100, 0x15166345, 0xFFFF0000);
        SIMDMask4 mask(true, false, true, true);
        vec0.sadda(mask, vec1);
        CHECK_CONDITION(vec0[0] == 0xFFFFFFFF && vec0[1] == 0xFFFFFF00 &&
                        vec0[2] == 0x15166348 && vec0[3] == 0xFFFFFFFF, "MSADDVA");
    }
    {
        SIMD4_32u vec0(0xFFFFFFFF, 0xFFFFFF00, 0x00000003, 0x12345678);
        uint32_t val1 = 0x00000100;
        vec0.sadda(val1);
        CHECK_CONDITION(vec0[0] == 0xFFFFFFFF && vec0[1] == 0xFFFFFFFF &&
                        vec0[2] == 0x00000103 && vec0[3] == 0x12345778, "SADDSA");
    }
    {
        SIMD4_32u vec0(0xFFFFFFFF, 0xFFFFFF00, 0x00000003, 0x12345678);
        uint32_t val1 = 0x00000100;
        SIMDMask4 mask(true, false, true, true);
        vec0.sadda(mask, val1);
        CHECK_CONDITION(vec0[0] == 0xFFFFFFFF && vec0[1] == 0xFFFFFF00 &&
                        vec0[2] == 0x00000103 && vec0[3] == 0x12345778, "MSADDSA");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1((uint32_t)0);
        vec1 = vec0.postinc();
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == 9 && vec0[2] == 8 && vec0[3] == 7, "POSTINC 1");
        CHECK_CONDITION(vec1[0] == 9 && vec1[1] == 8 && vec1[2] == 7 && vec1[3] == 6, "POSTINC 2");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1((uint32_t)0);
        vec1 = vec0++;
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == 9 && vec0[2] == 8 && vec0[3] == 7, "POSTINC 1(operator++)");
        CHECK_CONDITION(vec1[0] == 9 && vec1[1] == 8 && vec1[2] == 7 && vec1[3] == 6, "POSTINC 2(operator++)");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1((uint32_t)0);
        vec1 = vec0.prefinc();
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == 9 && vec0[2] == 8 && vec0[3] == 7, "PREFINC 1");
        CHECK_CONDITION(vec1[0] == 10 && vec1[1] == 9 && vec1[2] == 8 && vec1[3] == 7, "PREFINC 2");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1((uint32_t)0);
        vec1 = ++vec0;
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == 9 && vec0[2] == 8 && vec0[3] == 7, "PREFINC 1(operator++)");
        CHECK_CONDITION(vec1[0] == 10 && vec1[1] == 9 && vec1[2] == 8 && vec1[3] == 7, "PREFINC 2(operator++)");
    }
    {
        SIMD4_32u vec0(9, 14, 28, 60);
        SIMD4_32u vec1(3, 8,   7,  6);
        SIMD4_32u vec2;
        vec2 = vec0.sub(vec1);
        CHECK_CONDITION(vec2[0] == 6 && vec2[1] == 6 && vec2[2] == 21 && vec2[3] == 54, "SUBV");
    }
    {
        SIMD4_32u vec0(9, 14, 28, 60);
        SIMD4_32u vec1(3, 8,   7,  6);
        SIMD4_32u vec2;
        vec2 = vec0 - vec1;
        CHECK_CONDITION(vec2[0] == 6 && vec2[1] == 6 && vec2[2] == 21 && vec2[3] == 54, "SUBV(operator-)");
    }
    {
        SIMD4_32u vec0(900, 8, 7, 6);
        uint32_t b = 34;
        SIMD4_32u vec2;
        vec2 = vec0.sub(b);
        CHECK_CONDITION(vec2[0] == 866 && vec2[1] == 0xFFFFFFE6 && vec2[2] == 0xFFFFFFE5 && vec2[3] == 0xFFFFFFE4, "SUBS");
    }
    {
        SIMD4_32u vec0(9, 14, 28, 60);
        SIMD4_32u vec1(3, 8,   7,  6);
        vec0.suba(vec1);
        CHECK_CONDITION(vec0[0] == 6 && vec0[1] == 6 && vec0[2] == 21 && vec0[3] == 54, "SUBVA");
    }
    {
        SIMD4_32u vec0(9, 14, 28, 60);
        SIMD4_32u vec1(3, 8,   7,  6);
        vec0 -= vec1;
        CHECK_CONDITION(vec0[0] == 6 && vec0[1] == 6 && vec0[2] == 21 && vec0[3] == 54, "SUBVA(operator-=)");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 4;
        vec0.suba(val1);
        CHECK_CONDITION(vec0[0] == 5 && vec0[1] == 4 && vec0[2] == 3 && vec0[3] == 2, "SUBSA");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1((uint32_t)0);
        vec1 = vec0.postdec();
        CHECK_CONDITION(vec0[0] == 8 && vec0[1] == 7 && vec0[2] == 6 && vec0[3] == 5, "POSTDEC 1");
        CHECK_CONDITION(vec1[0] == 9 && vec1[1] == 8 && vec1[2] == 7 && vec1[3] == 6, "POSTDEC 2");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1((uint32_t)0);
        vec1 = vec0--;
        CHECK_CONDITION(vec0[0] == 8 && vec0[1] == 7 && vec0[2] == 6 && vec0[3] == 5, "POSTDEC 1(operator--)");
        CHECK_CONDITION(vec1[0] == 9 && vec1[1] == 8 && vec1[2] == 7 && vec1[3] == 6, "POSTDEC 2(operaotr--)");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1((uint32_t)0);
        vec1 = --vec0;
        CHECK_CONDITION(vec0[0] == 8 && vec0[1] == 7 && vec0[2] == 6 && vec0[3] == 5, "PREFDEC 1(operator--)");
        CHECK_CONDITION(vec1[0] == 8 && vec1[1] == 7 && vec1[2] == 6 && vec1[3] == 5, "PREFDEC 2(operator--)");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1((uint32_t)0);
        SIMD4_32u vec2;
        vec2 = vec0.mul(vec1);
        CHECK_CONDITION(vec2[0] == 0 && vec2[1] == 0 && vec2[2] == 0 && vec2[3] == 0, "MULV 1");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1((uint32_t)0);
        SIMD4_32u vec2;
        vec2 = vec0 * vec1;
        CHECK_CONDITION(vec2[0] == 0 && vec2[1] == 0 && vec2[2] == 0 && vec2[3] == 0, "MULV 1(operator*)");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(3);
        SIMD4_32u vec2;
        vec2 = vec0.mul(vec1);
        CHECK_CONDITION(vec2[0] == 27 && vec2[1] == 24 && vec2[2] == 21 && vec2[3] == 18, "MULV 2");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(3);
        SIMD4_32u vec2;
        vec2 = vec0 * vec1;
        CHECK_CONDITION(vec2[0] == 27 && vec2[1] == 24 && vec2[2] == 21 && vec2[3] == 18, "MULV 2(operator*)");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 0;
        SIMD4_32u vec2;
        vec2 = vec0.mul(val1);
        CHECK_CONDITION(vec2[0] == 0 && vec2[1] == 0 && vec2[2] == 0 && vec2[3] == 0, "MULS 1");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        SIMD4_32u vec2;
        vec2 = vec0.mul(val1);
        CHECK_CONDITION(vec2[0] == 27 && vec2[1] == 24 && vec2[2] == 21 && vec2[3] == 18, "MULS 2");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(3);
        vec0.mula(vec1);
        CHECK_CONDITION(vec0[0] == 27 && vec0[1] == 24 && vec0[2] == 21 && vec0[3] == 18, "MULVA");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(3);
        vec0 *= vec1;
        CHECK_CONDITION(vec0[0] == 27 && vec0[1] == 24 && vec0[2] == 21 && vec0[3] == 18, "MULVA(operator*=)");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        vec0.mula(val1);
        CHECK_CONDITION(vec0[0] == 27 && vec0[1] == 24 && vec0[2] == 21 && vec0[3] == 18, "MULSA");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(3);
        SIMD4_32u vec2;
        vec2 = vec0.div(vec1);
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == 2 && vec2[2] == 2 && vec2[3] == 2, "DIVV");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(3);
        SIMD4_32u vec2;
        vec2 = vec0 / vec1;
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == 2 && vec2[2] == 2 && vec2[3] == 2, "DIVV(operator/)");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        SIMD4_32u vec2;
        vec2 = vec0.div(val1);
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == 2 && vec2[2] == 2 && vec2[3] == 2, "DIVS");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(3);
        vec0.diva(vec1);
        CHECK_CONDITION(vec0[0] == 3 && vec0[1] == 2 && vec0[2] == 2 && vec0[3] == 2, "DIVVA");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(3);
        vec0 /= vec1;
        CHECK_CONDITION(vec0[0] == 3 && vec0[1] == 2 && vec0[2] == 2 && vec0[3] == 2, "DIVVA(operator/=)");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        vec0.diva(val1);
        CHECK_CONDITION(vec0[0] == 3 && vec0[1] == 2 && vec0[2] == 2 && vec0[3] == 2, "DIVSA");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 18;
        SIMD4_32u vec2;
        vec2 = vec0.rcp(val1);
        CHECK_CONDITION(vec2[0] == 2 && vec2[1] == 2 && vec2[2] == 2 && vec2[3] == 3, "RCPS");
    }
    {
        alignas(16) uint32_t arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        uint64_t indices[] = {1, 3, 8, 5};
        SIMD4_32u vec0(1);
        vec0.gather(arr, indices);
        CHECK_CONDITION(vec0[0] == 2 && vec0[1] == 4 && vec0[2] == 9 && vec0[3] == 6, "GATHER");
    }
    {
        alignas(16) uint32_t arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        uint64_t indices[] = {1, 3, 8, 5};
        SIMD4_32u vec0(1);
        SIMDMask4 mask(true, false, true, false);
        vec0.gather(mask, arr, indices);
        CHECK_CONDITION(vec0[0] == 2 && vec0[1] == 1 && vec0[2] == 9 && vec0[3] == 1, "MGATHER");
    }
    {
        alignas(16) uint32_t arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        SIMD4_32u indices(1, 3, 8, 5);
        SIMD4_32u vec0(1);
        vec0.gather(arr, indices);
        CHECK_CONDITION(vec0[0] == 2 && vec0[1] == 4 && vec0[2] == 9 && vec0[3] == 6, "GATHERV");
    }
    {
        alignas(16) uint32_t arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        SIMD4_32u indices(1, 3, 8, 5);
        SIMD4_32u vec0(1);
        SIMDMask4 mask(true, false, true, false);
        vec0.gather(mask, arr, indices);
        CHECK_CONDITION(vec0[0] == 2 && vec0[1] == 1 && vec0[2] == 9 && vec0[3] == 1, "MGATHERV");
    }
    {
        alignas(16) uint32_t arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        uint64_t indices[] = {1, 3, 8, 5};
        SIMD4_32u vec0(9, 8, 7, 6);
        vec0.scatter(arr, indices);
        CHECK_CONDITION(arr[1] == 9 && arr[3] == 8 && arr[8] == 7 && arr[9] == 10, "SCATTER");
    }
    {
        alignas(16) uint32_t arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        SIMD4_32u indices(1, 3, 8, 5);
        SIMD4_32u vec0(9, 8, 7, 6);
        vec0.scatter(arr, indices);
        CHECK_CONDITION(arr[1] == 9 && arr[3] == 8 && arr[8] == 7 && arr[9] == 10, "SCATTERV");
    }
    {
        alignas(16) uint32_t arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        SIMD4_32u indices(1, 3, 8, 5);
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMDMask4 mask(true, false, true, false);
        vec0.scatter(mask, arr, indices);
        CHECK_CONDITION(arr[1] == 9 && arr[3] == 4 && arr[8] == 7 && arr[9] == 10, "MSCATTERV");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(1, 2, 3, 4);
        SIMD4_32u vec2;
        vec2 = vec0.lsh(vec1);
        CHECK_CONDITION(vec2[0] == 18 && vec2[1] == 32 && vec2[2] == 56 && vec2[3] == 96, "LSHV");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(1, 2, 3, 4);
        SIMD4_32u vec2;
        SIMDMask4       mask(true, false, false, true);
        vec2 = vec0.lsh(mask, vec1);
        CHECK_CONDITION(vec2[0] == 18 && vec2[1] == 8 && vec2[2] == 7 && vec2[3] == 96, "MLSHV");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        SIMD4_32u vec2;
        vec2 = vec0.lsh(val1);
        CHECK_CONDITION(vec2[0] == 72 && vec2[1] == 64 && vec2[2] == 56 && vec2[3] == 48, "LSHS");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        SIMD4_32u vec2;
        SIMDMask4       mask(true, false, false, true);
        vec2 = vec0.lsh(mask, val1);
        CHECK_CONDITION(vec2[0] == 72 && vec2[1] == 8 && vec2[2] == 7 && vec2[3] == 48, "MLSHS");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(1, 2, 3, 4);
        vec0.lsha(vec1);
        CHECK_CONDITION(vec0[0] == 18 && vec0[1] == 32 && vec0[2] == 56 && vec0[3] == 96, "LSHVA");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(1, 2, 3, 4);
        SIMDMask4       mask(true, false, true, false);
        vec0.lsha(mask, vec1);
        CHECK_CONDITION(vec0[0] == 18 && vec0[1] == 8 && vec0[2] == 56 && vec0[3] == 6, "MLSHVA");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        vec0.lsha(val1);
        CHECK_CONDITION(vec0[0] == 72 && vec0[1] == 64 && vec0[2] == 56 && vec0[3] == 48, "LSHSA");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMDMask4       mask(true, false, true, false);
        uint32_t val1 = 3;
        vec0.lsha(mask, val1);
        CHECK_CONDITION(vec0[0] == 72 && vec0[1] == 8 && vec0[2] == 56 && vec0[3] == 6, "MLSHSA");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(1, 2, 3, 4);
        SIMD4_32u vec2;
        vec2 = vec0.rsh(vec1);
        CHECK_CONDITION(vec2[0] == 4 && vec2[1] == 2 && vec2[2] == 0 && vec2[3] == 0, "RSHV");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        SIMD4_32u vec2;
        vec2 = vec0.rsh(val1);
        CHECK_CONDITION(vec2[0] == 1 && vec2[1] == 1 && vec2[2] == 0 && vec2[3] == 0, "RSHS");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(1, 2, 3, 4);
        vec0.rsha(vec1);
        CHECK_CONDITION(vec0[0] == 4 && vec0[1] == 2 && vec0[2] == 0 && vec0[3] == 0, "RSHVA");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        vec0.rsha(val1);
        CHECK_CONDITION(vec0[0] == 1 && vec0[1] == 1 && vec0[2] == 0 && vec0[3] == 0, "RSHSA");
    }
    {
        SIMD4_32u vec0(0x91111111);
        SIMD4_32u vec1(3, 5, 7, 23);
        SIMD4_32u vec2;

        vec2 = vec0.rol(vec1);
        CHECK_CONDITION(vec2[0] == 0x8888888C && vec2[1] == 0x22222232 && vec2[2] == 0x888888C8 && vec2[3] == 0x88C88888, "ROLV");
    }
    {
        SIMD4_32u vec0(0x91111111);
        SIMD4_32u vec1(3, 5, 7, 23);
        SIMD4_32u vec2;
        SIMDMask4 mask(true, false, true, false);
        vec2 = vec0.rol(mask, vec1);
        CHECK_CONDITION(vec2[0] == 0x8888888C && vec2[1] == 0x91111111 && vec2[2] == 0x888888C8 && vec2[3] == 0x91111111, "MROLV");
    }
    {
        SIMD4_32u vec0(0x80000001, 0x8F000001, 0x8F0000F1, 0xEF0000F1);
        uint32_t val1 = 5;
        SIMD4_32u vec2;
        vec2 = vec0.rol(val1);
        CHECK_CONDITION(vec2[0] == 0x00000030 && vec2[1] == 0xE0000031 && vec2[2] == 0xE0001E31 && vec2[3] == 0xE0001E3D, "ROLS");
    }
    {
        SIMD4_32u vec0(0x80000001, 0x8F000001, 0x8F0000F1, 0xEF0000F1);
        uint32_t val1 = 5;
        SIMD4_32u vec2;
        SIMDMask4 mask(true, false, true, false);
        vec2 = vec0.rol(mask, val1);
        CHECK_CONDITION(vec2[0] == 0x00000030 && vec2[1] == 0x8F000001 && vec2[2] == 0xE0001E31 && vec2[3] == 0xEF0000F1, "MROLS");
    }
    {
        SIMD4_32u vec0(0x91111111);
        SIMD4_32u vec1(3, 5, 7, 23);
        vec0.rola(vec1);
        CHECK_CONDITION(vec0[0] == 0x8888888C && vec0[1] == 0x22222232 && vec0[2] == 0x888888C8 && vec0[3] == 0x88C88888, "ROLVA");
    }
    {
        SIMD4_32u vec0(0x91111111);
        SIMD4_32u vec1(3, 5, 7, 23);
        SIMDMask4 mask(true, false, true, false);
        vec0.rola(mask, vec1);
        CHECK_CONDITION(vec0[0] == 0x8888888C && vec0[1] == 0x91111111 && vec0[2] == 0x888888C8 && vec0[3] == 0x91111111, "MROLVA");
    }
    {
        SIMD4_32u vec0(0x80000001, 0x8F000001, 0x8F0000F1, 0xEF0000F1);
        uint32_t val1 = 5;
        vec0.rola(val1);
        CHECK_CONDITION(vec0[0] == 0x00000030 && vec0[1] == 0xE0000031 && vec0[2] == 0xE0001E31 && vec0[3] == 0xE0001E3D, "ROLSA");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        SIMD4_32u  vec1(1, 9, 0, 5);
        SIMDMask4 mask;
        mask = vec0.cmpeq(vec1);
        CHECK_CONDITION(mask[0] == true && mask[1] == false && mask[2] == false && mask[3] == true, "CMPEQV");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        SIMD4_32u  vec1(1, 9, 0, 5);
        SIMDMask4 mask;
        mask = vec0 == vec1;
        CHECK_CONDITION(mask[0] == true && mask[1] == false && mask[2] == false && mask[3] == true, "CMPEQV(operator==)");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        uint32_t val1 = 3;
        SIMDMask4 mask;
        mask = vec0.cmpeq(val1);
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == true && mask[3] == false, "CMPEQS");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        SIMD4_32u  vec1(1, 9, 0, 5);
        SIMDMask4 mask;
        mask = vec0.cmpne(vec1);
        CHECK_CONDITION(mask[0] == false && mask[1] == true && mask[2] == true && mask[3] == false, "CMPNEV");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        SIMD4_32u  vec1(1, 9, 0, 5);
        SIMDMask4 mask;
        mask = vec0 != vec1;
        CHECK_CONDITION(mask[0] == false && mask[1] == true && mask[2] == true && mask[3] == false, "CMPNEV(operator!=)");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        uint32_t val1 = 3;
        SIMDMask4 mask;
        mask = vec0.cmpne(val1);
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == false && mask[3] == true, "CMPNES");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        SIMD4_32u  vec1(1, 9, 0, 2);
        SIMDMask4 mask;
        mask = vec0.cmpgt(vec1);
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == true && mask[3] == true, "CMPGTV");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        SIMD4_32u  vec1(1, 9, 0, 2);
        SIMDMask4 mask;
        mask = vec0 > vec1;
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == true && mask[3] == true, "CMPGTV(operator>)");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        uint32_t val1 = 3;
        SIMDMask4 mask;
        mask = vec0.cmpgt(val1);
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == false && mask[3] == true, "CMPGTS");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        SIMD4_32u  vec1(1, 9, 0, 2);
        SIMDMask4 mask;
        mask = vec0.cmplt(vec1);
        CHECK_CONDITION(mask[0] == false && mask[1] == true && mask[2] == false && mask[3] == false, "CMPLTV");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        SIMD4_32u  vec1(1, 9, 0, 2);
        SIMDMask4 mask;
        mask = vec0 < vec1;
        CHECK_CONDITION(mask[0] == false && mask[1] == true && mask[2] == false && mask[3] == false, "CMPLTV(operator<)");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        uint32_t val1 = 3;
        SIMDMask4 mask;
        mask = vec0.cmplt(val1);
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == false && mask[3] == false, "CMPLTS");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        SIMD4_32u  vec1(1, 9, 3, 2);
        SIMDMask4 mask;
        mask = vec0.cmpge(vec1);
        CHECK_CONDITION(mask[0] == true && mask[1] == false && mask[2] == true && mask[3] == true, "CMPGEV");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        SIMD4_32u  vec1(1, 9, 3, 2);
        SIMDMask4 mask;
        mask = vec0 >= vec1;
        CHECK_CONDITION(mask[0] == true && mask[1] == false && mask[2] == true && mask[3] == true, "CMPGEV(operator >=)");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        uint32_t val1 = 3;
        SIMDMask4 mask;
        mask = vec0.cmpge(val1);
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == true && mask[3] == true, "CMPGES");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        SIMD4_32u  vec1(1, 9, 3, 2);
        SIMDMask4 mask;
        mask = vec0.cmple(vec1);
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == true && mask[3] == false, "CMPLEV");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        SIMD4_32u  vec1(1, 9, 3, 2);
        SIMDMask4 mask;
        mask = vec0 <= vec1;
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == true && mask[3] == false, "CMPLEV(operator<=)");
    }
    {
        SIMD4_32u  vec0(1, 2, 3, 5);
        uint32_t val1 = 3;
        SIMDMask4 mask;
        mask = vec0.cmple(val1);
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == true && mask[3] == false, "CMPLES");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32u vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);
        SIMD4_32u vec2;

        vec2 = vec0.band(vec1);
        CHECK_CONDITION(
            vec2[0] == 0x01012000 && vec2[1] == 0x00000300 && vec2[2] == 0x09508060 && vec2[3] == 0x000F4020, 
            "BANDV");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32u vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);
        SIMD4_32u vec2;

        vec2 = vec0 & vec1;
        CHECK_CONDITION(
            vec2[0] == 0x01012000 && vec2[1] == 0x00000300 && vec2[2] == 0x09508060 && vec2[3] == 0x000F4020, 
            "BANDV(operator&)");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        uint32_t val1 = 0x571FC5A0;
        SIMD4_32u vec2;

        vec2 = vec0.band(val1);
        CHECK_CONDITION(
            vec2[0] == 0x53130120 && vec2[1] == 0x500F0500 && vec2[2] == 0x0710C0A0 && vec2[3] == 0x000F4020, 
            "BANDS");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        uint32_t val1 = 0x571FC5A0;
        SIMD4_32u vec2;
        SIMDMask4 mask(true, false, false, true);

        vec2 = vec0.band(mask, val1);
        CHECK_CONDITION(
            vec2[0] == 0x53130120 && vec2[1] == 0xF00F0F10 && vec2[2] == 0x0FF0F0F0 && vec2[3] == 0x000F4020, 
            "MBANDS");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32u vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);

        vec0.banda(vec1);
        CHECK_CONDITION(
            vec0[0] == 0x01012000 && vec0[1] == 0x00000300 && vec0[2] == 0x09508060 && vec0[3] == 0x000F4020, 
            "BANDVA");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32u vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);

        vec0 &= vec1;
        CHECK_CONDITION(
            vec0[0] == 0x01012000 && vec0[1] == 0x00000300 && vec0[2] == 0x09508060 && vec0[3] == 0x000F4020, 
            "BANDVA(operator&=)");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32u vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);
        SIMDMask4 mask(true, false, false, true);

        vec0.banda(mask, vec1);
        CHECK_CONDITION(
            vec0[0] == 0x01012000 && vec0[1] == 0xF00F0F10 && vec0[2] == 0x0FF0F0F0 && vec0[3] == 0x000F4020, 
            "MBANDVA");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        uint32_t val1 = 0x571FC5A0;
        SIMDMask4 mask(true, false, false, true);
                
        vec0.banda(mask, val1);
        CHECK_CONDITION(
            vec0[0] == 0x53130120 && vec0[1] == 0xF00F0F10 && vec0[2] == 0x0FF0F0F0 && vec0[3] == 0x000F4020, 
            "MBANDSA");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32u vec1;
        vec1 = vec0.bnot();
        CHECK_CONDITION(
            vec1[0] == 0x0CCCCCCB && vec1[1] == 0x0FF0F0EF && vec1[2] == 0xF00F0F0F && vec1[3] == 0xFFF0BDC0, 
            "BNOT");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32u vec1;
        vec1 = ~vec0;
        CHECK_CONDITION(
            vec1[0] == 0x0CCCCCCB && vec1[1] == 0x0FF0F0EF && vec1[2] == 0xF00F0F0F && vec1[3] == 0xFFF0BDC0, 
            "BNOT(operator~)");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32u vec1;
        SIMDMask4 mask(true, false, true, false);
        vec1 = vec0.bnot(mask);
        CHECK_CONDITION(
            vec1[0] == 0x0CCCCCCB && vec1[1] == 0xF00F0F10 && vec1[2] == 0xF00F0F0F && vec1[3] == 0x000F423F, 
            "MBNOT");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMDMask4 mask(true, false, true, false);
        vec0.bnota(mask);
        CHECK_CONDITION(
            vec0[0] == 0x0CCCCCCB && vec0[1] == 0xF00F0F10 && vec0[2] == 0xF00F0F0F && vec0[3] == 0x000F423F, 
            "MBNOTA");
    }
    {
        SIMD4_32u vec0(3), vec1(5);
        SIMD4_32u vec2(2);
        SIMDMask4 mask(true, false, false, true);
        vec2 = vec0.blend(mask, vec1);
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == 5 && vec2[2] == 5 && vec2[3] == 3, "MBLENDV");
    }
    {
        SIMD4_32u vec0(3);
        uint32_t val1 = 5;
        SIMD4_32u vec2(2);
        SIMDMask4 mask(true, false, false, true);
        vec2 = vec0.blend(mask, val1);
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == 5 && vec2[2] == 5 && vec2[3] == 3, "MBLENDS");
    }
    {
        SIMD4_32u vec0(0xF3333304, 0xFF0F3F00, 0x0FF0F000, 0x0F0F720F);
        uint32_t val1;
        val1 = vec0.hband();
        CHECK_CONDITION(val1 == 0x03003000, "HBAND");
    }
    {
        SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(true, false, false, true);
        uint32_t val1;
        val1 = vec0.hband(mask);
        CHECK_CONDITION(val1 == 0x00030204, "MHBAND");
    }
    {
        SIMD4_32u vec0(0xF3333304, 0xFF0F3F00, 0x0FF0F000, 0x0F0F720F);
        uint32_t val1;
        uint32_t val2 = 0x03003000;
        val1 = vec0.hband(val2);
        CHECK_CONDITION(val1 == 0x03003000, "HBANDS");
    }
    {
        SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(true, false, false, true);
        uint32_t val1;
        uint32_t val2 = 0x00010004;
        val1 = vec0.hband(mask, val2);
        CHECK_CONDITION(val1 == 0x00010004, "MHBANDS");
    }
    {
        SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        uint32_t val1;
        val1 = vec0.hbor();
        CHECK_CONDITION(val1 == 0xFFFFFF0F, "HBOR");
    }
    {
        SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(false, false, true, true);
        uint32_t val1;
        val1 = vec0.hbor(mask);
        CHECK_CONDITION(val1 == 0x0FFFF20F, "MHBOR");
    }
    {
        SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        uint32_t val1;
        uint32_t val2 = 0x00000030;
        val1 = vec0.hbor(val2);
        CHECK_CONDITION(val1 == 0xFFFFFF3F, "HBORS");
    }
    {
        SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(false, false, true, true);
        uint32_t val1;
        uint32_t val2 = 0x00000030;
        val1 = vec0.hbor(mask, val2);
        CHECK_CONDITION(val1 == 0x0FFFF23F, "MHBORS");
    }
    {
        SIMD4_32u vec0(1, 2, 3, 4);
        uint32_t val1 = 0;
        val1 = vec0.hmul();
        CHECK_CONDITION(val1 == 24, "HMUL");
    }
    {
        SIMD4_32u vec0(1, 2, 3, 4);
        SIMDMask4 mask(true, false, true, false);
        uint32_t val1 = 0;
        val1 = vec0.hmul(mask);
        CHECK_CONDITION(val1 == 3, "MHMUL");
    }
    {
        SIMD4_32u vec0(1, 2, 3, 4);
        uint32_t val1 = 42;
        uint32_t res = 0;
        res = vec0.hmul(val1);
        CHECK_CONDITION(res == 1008, "HMULS");
    }

    return g_failCount;
}

int test_UME_SIMD4_32i(bool supressMessages)
{   
    char header[] = "UME::SIMD::SIMD4_32i test";
    
    INIT_TEST(header, supressMessages);
    
    {
        SIMD4_32i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR"); 
    }
    {  
        SIMD4_32i vec0(-3);
        CHECK_CONDITION(vec0[1] == -3, "SET-CONSTR"); 
    }
    {  
        SIMD4_32i vec0(-3, -2, -1, 6);
        CHECK_CONDITION(vec0[0] == -3 && vec0[3] == 6, "FULL-CONSTR"); 
    }
    genericIntTest<SIMD4_32i, int32_t, SIMDMask4, 4, DataSet_1_32i>();
    {
        SIMD4_32u vec0(8);
        SIMD4_32i vec1;

        vec1 = SIMD4_32i(vec0);
        CHECK_CONDITION(vec1[2] == 8, "UTOI");
    }
    {
        CHECK_CONDITION(SIMD4_32i::length() == 4, "LENGTH");
    }
    {
        CHECK_CONDITION(SIMD4_32i::alignment() == 16, "ALIGNEMENT");
    }
    {
        SIMD4_32i vec0(-1, 2, 3, 4);
        SIMD4_32i vec1(15);

        vec1 = vec0;
        CHECK_CONDITION(vec1[0] == -1 && vec1[1] == 2, "operator=");
    }    
    {
        int32_t arr[4] = { 1, 3, 8, -41231};
        SIMD4_32i vec0(-3);
        vec0.load(arr);
        CHECK_CONDITION(vec0[0] == 1 && vec0[3] == -41231, "LOAD");
    }
    {
        alignas(16) int32_t arr[4] = { 1, 3, 8, -41231};
        SIMD4_32i vec0(-3);
        vec0.loada(arr);
        CHECK_CONDITION(vec0[0] == 1 && vec0[3] == -41231, "LOADA");
    }
    {
        int32_t arr[4] = { 1, 3, 9, -124};
        SIMD4_32i vec0(9, 32, -28, -1256);
        vec0.store(arr);
        CHECK_CONDITION(arr[0] == 9 && arr[3] == -1256, "STORE");
    }
    {
        alignas(16) int32_t arr[4] = { 1, 3, 9, -124};
        SIMD4_32i vec0(9, 32, -28, -1256);
        vec0.storea(arr);
        CHECK_CONDITION(arr[0] == 9 && arr[3] == -1256, "STOREA");
    }
    {
        alignas(16) int32_t arr[10] = {1, -2, 3, 4, 5, -6, 7, 8, 9, 10};
        uint64_t indices[] = {1, 3, 8, 5};
        SIMD4_32i vec0(1);
        vec0.gather(arr, indices);
        CHECK_CONDITION(vec0[0] == -2 && vec0[1] == 4 && vec0[2] == 9 && vec0[3] == -6, "GATHERS");
    }
    {
        alignas(16) int32_t arr[10] = {1, -2, 3, 4, 5, -6, 7, 8, 9, 10};
        uint64_t indices[] = {1, 3, 8, 5};
        SIMD4_32i vec0(1);
        SIMDMask4 mask(true, false, true, false);
        vec0.gather(mask, arr, indices);
        CHECK_CONDITION(vec0[0] == -2 && vec0[1] == 1 && vec0[2] == 9 && vec0[3] == 1, "MGATHERS");
    }
    {
        alignas(16) int32_t arr[10] = {1, -2, 3, 4, 5, -6, 7, 8, 9, 10};
        SIMD4_32u indices(1, 3, 8, 5);
        SIMD4_32i vec0(1);
        vec0.gather(arr, indices);
        CHECK_CONDITION(vec0[0] == -2 && vec0[1] == 4 && vec0[2] == 9 && vec0[3] == -6, "GATHERV");
    }
    {
        alignas(16) int32_t arr[10] = {1, -2, 3, 4, 5, -6, 7, 8, 9, 10};
        SIMD4_32u indices(1, 3, 8, 5);
        SIMD4_32i vec0(1);
        SIMDMask4 mask(true, false, true, false);
        vec0.gather(mask, arr, indices);
        CHECK_CONDITION(vec0[0] == -2 && vec0[1] == 1 && vec0[2] == 9 && vec0[3] == 1, "MGATHERV");
    }
    {
        alignas(16) int32_t arr[10] = {1, -2, 3, 4, 5, -6, 7, 8, 9, 10};
        uint64_t indices[] = {1, 3, 8, 5};
        SIMD4_32i vec0(9, -8, 7, 6);
        vec0.scatter(arr, indices);
        CHECK_CONDITION(arr[1] == 9 && arr[3] == -8 && arr[8] == 7 && arr[9] == 10, "SCATTERS");
    }
    {
        alignas(16) int32_t arr[10] = {1, -2, 3, 4, 5, -6, 7, 8, 9, 10};
        SIMD4_32u indices(1, 3, 8, 5);
        SIMD4_32i vec0(9, -8, 7, 6);
        vec0.scatter(arr, indices);
        CHECK_CONDITION(arr[1] == 9 && arr[3] == -8 && arr[8] == 7 && arr[9] == 10, "SCATTERV");
    }
    {
        alignas(16) int32_t arr[10] = {1, -2, 3, 4, 5, -6, 7, 8, 9, 10};
        SIMD4_32u indices(1, 3, 8, 5);
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMDMask4 mask(true, false, true, false);
        vec0.scatter(mask, arr, indices);
        CHECK_CONDITION(arr[1] == 9 && arr[3] == 4 && arr[8] == 7 && arr[9] == 10, "MSCATTERV");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(3, 14, 28, -60);
        SIMD4_32i vec2;
        vec2 = vec0.add(vec1);
        CHECK_CONDITION(vec2[0] == 12 && vec2[1] == 6 && vec2[2] == 35 && vec2[3] == -54, "ADDV");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(3, 14, 28, -60);
        SIMD4_32i vec2;
        vec2 = vec0 + vec1;
        CHECK_CONDITION(vec2[0] == 12 && vec2[1] == 6 && vec2[2] == 35 && vec2[3] == -54, "ADDV(operator+)");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(3, 14, 28, -60);
        SIMD4_32i vec2;
        SIMDMask4 mask(true, false, true, true);
        vec2 = vec0.add(mask, vec1);
        CHECK_CONDITION(vec2[0] == 12 && vec2[1] == -8 && vec2[2] == 35 && vec2[3] == -54, "MADDV");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        int32_t b = 34;
        SIMD4_32i vec2;
        vec2 = vec0.add(b);
        CHECK_CONDITION(vec2[0] == 43 && vec2[1] == 26 && vec2[2] == 41 && vec2[3] == 40, "ADDS");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        int32_t b = 34;
        SIMD4_32i vec2;
        SIMDMask4 mask(true, false, true, true);
        vec2 = vec0.add(mask, b);
        CHECK_CONDITION(vec2[0] == 43 && vec2[1] == -8 && vec2[2] == 41 && vec2[3] == 40, "MADDS");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(3, 14, 28, -60);
        vec0.adda(vec1);
        CHECK_CONDITION(vec0[0] == 12 && vec0[1] == 6 && vec0[2] == 35 && vec0[3] == -54, "ADDVA");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(3, 14, 28, -60);
        vec0 += vec1;
        CHECK_CONDITION(vec0[0] == 12 && vec0[1] == 6 && vec0[2] == 35 && vec0[3] == -54, "ADDVA(operator+=)");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        int32_t val1 = 12;
        vec0.adda(val1);
        CHECK_CONDITION(vec0[0] == 21 && vec0[1] == 4 && vec0[2] == 19 && vec0[3] == 18, "ADDSA");
    }
    {
        SIMD4_32i vec0(0x7FFFFFFF, 0x7FFFFF00, 0x00000003, 0x12345678);
        SIMD4_32i vec1(0x00000005, 0x00000100, 0x15166345, 0x7FFF0000);
        SIMD4_32i vec2;
        vec2 = vec0.sadd(vec1);
        CHECK_CONDITION(vec2[0] == 0x7FFFFFFF && vec2[1] == 0x7FFFFFFF &&
                        vec2[2] == 0x15166348 && vec2[3] == 0x7FFFFFFF, "SADDV");
    }
    {
        SIMD4_32i vec0(0x7FFFFFFF, 0x7FFFFF00, 0x00000003, 0x12345678);
        SIMD4_32i vec1(0x00000005, 0x00000100, 0x15166345, 0x7FFF0000);
        SIMD4_32i vec2;
        SIMDMask4 mask(true, false, true, true);
        vec2 = vec0.sadd(mask, vec1);
        CHECK_CONDITION(vec2[0] == 0x7FFFFFFF && vec2[1] == 0x7FFFFF00 &&
                        vec2[2] == 0x15166348 && vec2[3] == 0x7FFFFFFF, "MSADDV");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(0);
        vec1 = vec0.postinc();
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == -7 && vec0[2] == 8 && vec0[3] == 7, "POSTINC 1");
        CHECK_CONDITION(vec1[0] == 9 && vec1[1] == -8 && vec1[2] == 7 && vec1[3] == 6, "POSTINC 2");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(0);
        vec1 = vec0++;
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == -7 && vec0[2] == 8 && vec0[3] == 7, "POSTINC 1(operator++)");
        CHECK_CONDITION(vec1[0] == 9 && vec1[1] == -8 && vec1[2] == 7 && vec1[3] == 6, "POSTINC 2(operator++)");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(0);
        vec1 = vec0.prefinc();
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == -7 && vec0[2] == 8 && vec0[3] == 7, "PREFINC 1");
        CHECK_CONDITION(vec1[0] == 10 && vec1[1] == -7 && vec1[2] == 8 && vec1[3] == 7, "PREFINC 2");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(0);
        vec1 = ++vec0;
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == -7 && vec0[2] == 8 && vec0[3] == 7, "PREFINC 1(operator++)");
        CHECK_CONDITION(vec1[0] == 10 && vec1[1] == -7 && vec1[2] == 8 && vec1[3] == 7, "PREFINC 2(operator++)");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(3, 14, 28, -60);
        SIMD4_32i vec2;
        vec2 = vec0.sub(vec1);
        CHECK_CONDITION(vec2[0] == 6 && vec2[1] == -22 && vec2[2] == -21 && vec2[3] == 66, "SUBV");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(3, 14, 28, -60);
        SIMD4_32i vec2;
        vec2 = vec0 - vec1;
        CHECK_CONDITION(vec2[0] == 6 && vec2[1] == -22 && vec2[2] == -21 && vec2[3] == 66, "SUBV(operator-)");
    }
    {
        SIMD4_32i vec0(9, 8, 7, 6);
        int32_t b = 34;
        SIMD4_32i vec2;
        vec2 = vec0.sub(b);
        CHECK_CONDITION(vec2[0] == -25 && vec2[1] == -26 && vec2[2] == -27 && vec2[3] == -28, "SUBS");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(3, 14, 28, -60);
        vec0.suba(vec1);
        CHECK_CONDITION(vec0[0] == 6 && vec0[1] == -22 && vec0[2] == -21 && vec0[3] == 66, "SUBVA");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(3, 14, 28, -60);
        vec0-=vec1;
        CHECK_CONDITION(vec0[0] == 6 && vec0[1] == -22 && vec0[2] == -21 && vec0[3] == 66, "SUBVA(operator-=)");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        int32_t val1 = 12;
        vec0.suba(val1);
        CHECK_CONDITION(vec0[0] == -3 && vec0[1] == -20 && vec0[2] == -5 && vec0[3] == -6, "SUBSA");
    }
    {
        SIMD4_32i vec0(0x80000000, 0x80000053, 0xF21ACAF4, 0x12341234);
        SIMD4_32i vec1(0x00000001, 0x00000100, 0x7FFF0000, 0x12341236);
        SIMD4_32i vec2;
        vec2 = vec0.ssub(vec1);
        CHECK_CONDITION(vec2[0] == 0x80000000 && vec2[1] == 0x80000000 &&
                        vec2[2] == 0x80000000 && vec2[3] == 0xFFFFFFFE, "SSUBV");
    }
    {
        SIMD4_32i vec0(0x80000000, 0x80000053, 0xF21ACAF4, 0x12341234);
        SIMD4_32i vec1(0x00000001, 0x00000100, 0x7FFF0000, 0x12341236);
        SIMD4_32i vec2;
        SIMDMask4 mask(true, false, false, true);
        vec2 = vec0.ssub(mask, vec1);
        CHECK_CONDITION(vec2[0] == 0x80000000 && vec2[1] == 0x80000053 &&
                        vec2[2] == 0xF21ACAF4 && vec2[3] == 0xFFFFFFFE, "MSSUBV");
    }
    {
        SIMD4_32i vec0(0x80000000, 0x80000053, 0xF21ACAF4, 0x12341234);
        int32_t val1 = 0x00000100;
        SIMD4_32i vec2;
        vec2 = vec0.ssub(val1);
        CHECK_CONDITION(vec2[0] == 0x80000000 && vec2[1] == 0x80000000 &&
                        vec2[2] == 0xF21AC9F4 && vec2[3] == 0x12341134, "SSUBS");
    }
    {
        SIMD4_32i vec0(0x80000000, 0x80000053, 0xF21ACAF4, 0x12341234);
        int32_t val1 = 0x00000100;
        SIMD4_32i vec2;
        SIMDMask4 mask(true, false, false, true);
        vec2 = vec0.ssub(mask, val1);
        CHECK_CONDITION(vec2[0] == 0x80000000 && vec2[1] == 0x80000053 &&
                        vec2[2] == 0xF21ACAF4 && vec2[3] == 0x12341134, "MSSUBS");
    }
    {
        SIMD4_32i vec0(0x80000000, 0x80000053, 0xF21ACAF4, 0x12341234);
        SIMD4_32i vec1(0x00000001, 0x00000100, 0x7FFF0000, 0x12341236);
        vec0.ssuba(vec1);
        CHECK_CONDITION(vec0[0] == 0x80000000 && vec0[1] == 0x80000000 &&
                        vec0[2] == 0x80000000 && vec0[3] == 0xFFFFFFFE, "SSUBVA");
    }
    {
        SIMD4_32i vec0(0x80000000, 0x80000053, 0xF21ACAF4, 0x12341234);
        SIMD4_32i vec1(0x00000001, 0x00000100, 0x7FFF0000, 0x12341236);
        SIMDMask4 mask(true, false, false, true);
        vec0.ssuba(mask, vec1);
        CHECK_CONDITION(vec0[0] == 0x80000000 && vec0[1] == 0x80000053 &&
                        vec0[2] == 0xF21ACAF4 && vec0[3] == 0xFFFFFFFE, "MSSUBVA");
    }
    {
        SIMD4_32i vec0(0x80000000, 0x80000053, 0xF21ACAF4, 0x12341234);
        int32_t val1 = 0x00000100;
        vec0.ssuba(val1);
        CHECK_CONDITION(vec0[0] == 0x80000000 && vec0[1] == 0x80000000 &&
                        vec0[2] == 0xF21AC9F4 && vec0[3] == 0x12341134, "SSUBSA");
    }
    {
        SIMD4_32i vec0(0x80000000, 0x80000053, 0xF21ACAF4, 0x12341234);
        int32_t val1 = 0x00000100;
        SIMDMask4 mask(true, false, false, true);
        vec0.ssuba(mask, val1);
        CHECK_CONDITION(vec0[0] == 0x80000000 && vec0[1] == 0x80000053 &&
                        vec0[2] == 0xF21ACAF4 && vec0[3] == 0x12341134, "MSSUBSA");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(3, 14, 28, -60);
        SIMD4_32i vec2;
        vec2 = vec1.subfrom(vec0);
        CHECK_CONDITION(vec2[0] == 6 && vec2[1] == -22 && vec2[2] == -21 && vec2[3] == 66, "SUBFROMV");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(3, 14, 28, -60);
        SIMD4_32i vec2;
        SIMDMask4 mask(true, false, true, false);
        vec2 = vec1.subfrom(mask, vec0);
        CHECK_CONDITION(vec2[0] == 6 && vec2[1] == -8 && vec2[2] == -21 && vec2[3] == 6, "MSUBFROMV");
    }
    {
        SIMD4_32i vec0(9, 8, 7, 6);
        int32_t a = 34;
        SIMD4_32i vec2;
        vec2 = vec0.subfrom(a);
        CHECK_CONDITION(vec2[0] == 25 && vec2[1] == 26 && vec2[2] == 27 && vec2[3] == 28, "SUBFROMS");
    }
    {
        SIMD4_32i vec0(9, 8, 7, 6);
        int32_t a = 34;
        SIMD4_32i vec2;
        SIMDMask4 mask(true, false, true, false);
        vec2 = vec0.subfrom(mask, a);
        CHECK_CONDITION(vec2[0] == 25 && vec2[1] == 34 && vec2[2] == 27 && vec2[3] == 34, "MSUBFROMS");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(3, 14, 28, -60);
        vec1.subfroma(vec0);
        CHECK_CONDITION(vec1[0] == 6 && vec1[1] == -22 && vec1[2] == -21 && vec1[3] == 66, "SUBFROMVA");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(3, 14, 28, -60);
        SIMDMask4 mask(true, false, true, false);
        vec1.subfroma(mask, vec0);
        CHECK_CONDITION(vec1[0] == 6 && vec1[1] == -8 && vec1[2] == -21 && vec1[3] == 6, "MSUBFROMVA");
    }
    {
        SIMD4_32i vec0(9, 8, 7, 6);
        int32_t a = 34;
        vec0.subfroma(a);
        CHECK_CONDITION(vec0[0] == 25 && vec0[1] == 26 && vec0[2] == 27 && vec0[3] == 28, "SUBFROMSA");
    }
    {
        SIMD4_32i vec0(9, 8, 7, 6);
        int32_t a = 34;
        SIMDMask4 mask(true, false, true, false);
        vec0.subfroma(mask, a);
        CHECK_CONDITION(vec0[0] == 25 && vec0[1] == 34 && vec0[2] == 27 && vec0[3] == 34, "MSUBFROMSA");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(0);
        vec1 = vec0.postdec();
        CHECK_CONDITION(vec0[0] == 8 && vec0[1] == -9 && vec0[2] == 6 && vec0[3] == 5, "POSTDEC 1");
        CHECK_CONDITION(vec1[0] == 9 && vec1[1] == -8 && vec1[2] == 7 && vec1[3] == 6, "POSTDEC 2");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(0);
        vec1 = vec0--;
        CHECK_CONDITION(vec0[0] == 8 && vec0[1] == -9 && vec0[2] == 6 && vec0[3] == 5, "POSTDEC 1(operator--)");
        CHECK_CONDITION(vec1[0] == 9 && vec1[1] == -8 && vec1[2] == 7 && vec1[3] == 6, "POSTDEC 2(operator--)");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(0);
        vec1 = vec0.prefdec();
        CHECK_CONDITION(vec0[0] == 8 && vec0[1] == -9 && vec0[2] == 6 && vec0[3] == 5, "PREFDEC 1");
        CHECK_CONDITION(vec1[0] == 8 && vec1[1] == -9 && vec1[2] == 6 && vec1[3] == 5, "PREFDEC 2");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(0);
        vec1 = --vec0;
        CHECK_CONDITION(vec0[0] == 8 && vec0[1] == -9 && vec0[2] == 6 && vec0[3] == 5, "PREFDEC 1(operator--)");
        CHECK_CONDITION(vec1[0] == 8 && vec1[1] == -9 && vec1[2] == 6 && vec1[3] == 5, "PREFDEC 2(operator--)");
    }
    {
        SIMD4_32i vec0(9, 8, 7, 6);
        SIMD4_32i vec1(0);
        SIMD4_32i vec2;
        vec2 = vec0.mul(vec1);
        CHECK_CONDITION(vec2[0] == 0 && vec2[1] == 0 && vec2[2] == 0 && vec2[3] == 0, "MULV 1");
    }
    {
        SIMD4_32i vec0(9, 8, 7, 6);
        SIMD4_32i vec1(0);
        SIMD4_32i vec2;
        vec2 = vec0 * vec1;
        CHECK_CONDITION(vec2[0] == 0 && vec2[1] == 0 && vec2[2] == 0 && vec2[3] == 0, "MULV 1(operator*)");
    }
    {
        SIMD4_32i vec0(9, 8, 7, 6);
        SIMD4_32i vec1(-3);
        SIMD4_32i vec2;
        vec2 = vec0.mul(vec1);
        CHECK_CONDITION(vec2[0] == -27 && vec2[1] == -24 && vec2[2] == -21 && vec2[3] == -18, "MULV 2");
    }
    {
        SIMD4_32i vec0(9, 8, 7, 6);
        SIMD4_32i vec1(-3);
        SIMD4_32i vec2;
        vec2 = vec0 * vec1;
        CHECK_CONDITION(vec2[0] == -27 && vec2[1] == -24 && vec2[2] == -21 && vec2[3] == -18, "MULV 2(operator*)");
    }
    {
        SIMD4_32i vec0(9, 8, 7, 6);
        int32_t val1 = 0;
        SIMD4_32i vec2;
        vec2 = vec0.mul(val1);
        CHECK_CONDITION(vec2[0] == 0 && vec2[1] == 0 && vec2[2] == 0 && vec2[3] == 0, "MULS 1");
    }
    {
        SIMD4_32i vec0(9, 8, 7, 6);
        int32_t val1 = -3;
        SIMD4_32i vec2;
        vec2 = vec0.mul(val1);
        CHECK_CONDITION(vec2[0] == -27 && vec2[1] == -24 && vec2[2] == -21 && vec2[3] == -18, "MULS 2");
    }
    {
        SIMD4_32i vec0(9, 8, 7, 6);
        SIMD4_32i vec1(-3);
        vec0.mula(vec1);
        CHECK_CONDITION(vec0[0] == -27 && vec0[1] == -24 && vec0[2] == -21 && vec0[3] == -18, "MULVA");
    }
    {
        SIMD4_32i vec0(9, 8, 7, 6);
        SIMD4_32i vec1(-3);
        vec0*=vec1;
        CHECK_CONDITION(vec0[0] == -27 && vec0[1] == -24 && vec0[2] == -21 && vec0[3] == -18, "MULVA(operator*=)");
    }
    {
        SIMD4_32i vec0(9, 8, 7, 6);
        int32_t val1 = -3;
        vec0.mula(val1);
        CHECK_CONDITION(vec0[0] == -27 && vec0[1] == -24 && vec0[2] == -21 && vec0[3] == -18, "MULSA");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        SIMD4_32i vec1(3);
        SIMD4_32i vec2;
        vec2 = vec0.div(vec1);
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == 2 && vec2[2] == -2 && vec2[3] == 2, "DIVV");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        SIMD4_32i vec1(3);
        SIMD4_32i vec2;
        vec2 = vec0 / vec1;
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == 2 && vec2[2] == -2 && vec2[3] == 2, "DIVV(operator/)");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        int32_t val1 = -3;
        SIMD4_32i vec2;
        vec2 = vec0.div(val1);
        CHECK_CONDITION(vec2[0] == -3 && vec2[1] == -2 && vec2[2] == 2 && vec2[3] == -2, "DIVS");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        SIMD4_32i vec1(3);
        vec0.diva(vec1);
        CHECK_CONDITION(vec0[0] == 3 && vec0[1] == 2 && vec0[2] == -2 && vec0[3] == 2, "DIVVA");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        SIMD4_32i vec1(3);
        vec0 /= vec1;
        CHECK_CONDITION(vec0[0] == 3 && vec0[1] == 2 && vec0[2] == -2 && vec0[3] == 2, "DIVVA(operator/=)");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        int32_t val1 = 3;
        vec0.diva(val1);
        CHECK_CONDITION(vec0[0] == 3 && vec0[1] == 2 && vec0[2] == -2 && vec0[3] == 2, "DIVSA");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        int32_t val1 = -18;
        SIMD4_32i vec2;
        vec2 = vec0.rcp(val1);
        CHECK_CONDITION(vec2[0] == -2 && vec2[1] == -2 && vec2[2] == 2 && vec2[3] == -3, "RCPS");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        SIMD4_32u vec1(1, 2, 3, 4);
        SIMD4_32i vec2;
        vec2 = vec0.lsh(vec1);
        CHECK_CONDITION(vec2[0] == 18 && vec2[1] == 32 && vec2[3] == 96, "LSHV"); // value of vec2[2] is implementation defined
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        SIMD4_32u vec1(1, 2, 3, 4);
        SIMD4_32i vec2;
        SIMDMask4       mask(true, false, false, true);
        vec2 = vec0.lsh(mask, vec1);
        CHECK_CONDITION(vec2[0] == 18 && vec2[1] == 8 && vec2[3] == 96, "MLSHV"); // value of vec2[2] is implementation defined
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        uint32_t val1 = 3;
        SIMD4_32i vec2;
        vec2 = vec0.lsh(val1);
        CHECK_CONDITION(vec2[0] == 72 && vec2[1] == 64 && vec2[3] == 48, "LSHS"); // value of vec2[2] is implementation defined
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        uint32_t val1 = 3;
        SIMD4_32i vec2;
        SIMDMask4       mask(true, false, false, true);
        vec2 = vec0.lsh(mask, val1);
        CHECK_CONDITION(vec2[0] == 72 && vec2[1] == 8 && vec2[3] == 48, "MLSHS"); // value of vec2[2] is implementation defined
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        SIMD4_32u vec1(1, 2, 3, 4);
        vec0.lsha(vec1);
        CHECK_CONDITION(vec0[0] == 18 && vec0[1] == 32 && vec0[3] == 96, "LSHVA");// value of vec2[2] is implementation defined
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        SIMD4_32u vec1(1, 2, 3, 4);
        SIMDMask4       mask(true, false, true, false);
        vec0.lsha(mask, vec1);
        CHECK_CONDITION(vec0[0] == 18 && vec0[1] == 8 && vec0[3] == 6, "MLSHVA"); // value of vec2[2] is implementation defined
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        uint32_t val1 = 3;
        vec0.lsha(val1);
        CHECK_CONDITION(vec0[0] == 72 && vec0[1] == 64 && vec0[3] == 48, "LSHSA"); // value of vec2[2] is implementation defined
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        SIMDMask4       mask(true, false, true, false);
        uint32_t val1 = 3;
        vec0.lsha(mask, val1);
        CHECK_CONDITION(vec0[0] == 72 && vec0[1] == 8 && vec0[3] == 6, "MLSHSA"); // value of vec2[2] is implementation defined
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        SIMD4_32u vec1(1, 2, 3, 4);
        SIMD4_32i vec2;
        vec2 = vec0.rsh(vec1);
        CHECK_CONDITION(vec2[0] == 4 && vec2[1] == 2 && vec2[3] == 0, "RSHV"); // value of vec2[2] is implementation defined
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        uint32_t val1 = 3;
        SIMD4_32i vec2;
        vec2 = vec0.rsh(val1);
        CHECK_CONDITION(vec2[0] == 1 && vec2[1] == 1 && vec2[3] == 0, "RSHS"); // value of vec2[2] is implementation defined
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        SIMD4_32u vec1(1, 2, 3, 4);
        vec0.rsha(vec1);
        CHECK_CONDITION(vec0[0] == 4 && vec0[1] == 2 && vec0[3] == 0, "RSHVA"); // value of vec2[2] is implementation defined
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        uint32_t val1 = 3;
        vec0.rsha(val1);
        CHECK_CONDITION(vec0[0] == 1 && vec0[1] == 1 && vec0[3] == 0, "RSHSA"); // value of vec2[2] is implementation defined
    }
    {
        SIMD4_32i vec0(0x91111111);
        SIMD4_32u vec1(3, 5, 7, 23);
        SIMD4_32i vec2;
        SIMDMask4 mask(true, false, true, false);
        vec2 = vec0.rol(mask, vec1);
        CHECK_CONDITION(vec2[0] == 0x8888888C && vec2[1] == 0x91111111 && vec2[2] == 0x888888C8 && vec2[3] == 0x91111111, "MROLV");
    }
    {
        SIMD4_32i vec0(0x80000001, 0x8F000001, 0x8F0000F1, 0xEF0000F1);
        uint32_t val1 = 5;
        SIMD4_32i vec2;
        vec2 = vec0.rol(val1);
        CHECK_CONDITION(vec2[0] == 0x00000030 && vec2[1] == 0xE0000031 && vec2[2] == 0xE0001E31 && vec2[3] == 0xE0001E3D, "ROLS");
    }
    {
        SIMD4_32i vec0(0x80000001, 0x8F000001, 0x8F0000F1, 0xEF0000F1);
        uint32_t val1 = 5;
        SIMD4_32i vec2;
        SIMDMask4 mask(true, false, true, false);
        vec2 = vec0.rol(mask, val1);
        CHECK_CONDITION(vec2[0] == 0x00000030 && vec2[1] == 0x8F000001 && vec2[2] == 0xE0001E31 && vec2[3] == 0xEF0000F1, "MROLS");
    }
    {
        SIMD4_32i vec0(0x91111111);
        SIMD4_32u vec1(3, 5, 7, 23);
        vec0.rola(vec1);
        CHECK_CONDITION(vec0[0] == 0x8888888C && vec0[1] == 0x22222232 && vec0[2] == 0x888888C8 && vec0[3] == 0x88C88888, "ROLVA");
    }
    {
        SIMD4_32i vec0(0x91111111);
        SIMD4_32u vec1(3, 5, 7, 23);
        SIMDMask4 mask(true, false, true, false);
        vec0.rola(mask, vec1);
        CHECK_CONDITION(vec0[0] == 0x8888888C && vec0[1] == 0x91111111 && vec0[2] == 0x888888C8 && vec0[3] == 0x91111111, "MROLVA");
    }
    {
        SIMD4_32i vec0(0x80000001, 0x8F000001, 0x8F0000F1, 0xEF0000F1);
        uint32_t val1 = 5;
        vec0.rola(val1);
        CHECK_CONDITION(vec0[0] == 0x00000030 && vec0[1] == 0xE0000031 && vec0[2] == 0xE0001E31 && vec0[3] == 0xE0001E3D, "ROLSA");
    }
    {
        SIMD4_32i  vec0(1, 2, -3, -5);
        SIMD4_32i  vec1(1, 9, 0, -5);
        SIMDMask4 mask;
        mask = vec0.cmpeq(vec1);
        CHECK_CONDITION(mask[0] == true && mask[1] == false && mask[2] == false && mask[3] == true, "CMPEQV");
    }
    {
        SIMD4_32i  vec0(1, 2, -3, -5);
        SIMD4_32i  vec1(1, 9, 0, -5);
        SIMDMask4 mask;
        mask = vec0 == vec1;
        CHECK_CONDITION(mask[0] == true && mask[1] == false && mask[2] == false && mask[3] == true, "CMPEQV(operator==)");
    }
    {
        SIMD4_32i  vec0(1, 2, -3, -5);
        int32_t val1 = -3;
        SIMDMask4 mask;
        mask = vec0.cmpeq(val1);
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == true && mask[3] == false, "CMPEQS");
    }
    {
        SIMD4_32i  vec0(1, 2, -3, -5);
        SIMD4_32i  vec1(1, 9, 0, -5);
        SIMDMask4 mask;
        mask = vec0.cmpne(vec1);
        CHECK_CONDITION(mask[0] == false && mask[1] == true && mask[2] == true && mask[3] == false, "CMPNEV");
    }
    {
        SIMD4_32i  vec0(1, 2, -3, -5);
        SIMD4_32i  vec1(1, 9, 0, -5);
        SIMDMask4 mask;
        mask = vec0 != vec1;
        CHECK_CONDITION(mask[0] == false && mask[1] == true && mask[2] == true && mask[3] == false, "CMPNEV(operator!=)");
    }
    {
        SIMD4_32i  vec0(1, 2, -3, -5);
        int32_t val1 = -3;
        SIMDMask4 mask;
        mask = vec0.cmpne(val1);
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == false && mask[3] == true, "CMPNES");
    }
    {
        SIMD4_32i  vec0(1, 2, 3, -5);
        SIMD4_32i  vec1(1, 9, 0, -2);
        SIMDMask4 mask;
        mask = vec0.cmpgt(vec1);
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == true && mask[3] == false, "CMPGTV");
    }
    {
        SIMD4_32i  vec0(1, 2, 3, -5);
        SIMD4_32i  vec1(1, 9, 0, -2);
        SIMDMask4 mask;
        mask = vec0 > vec1;
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == true && mask[3] == false, "CMPGTV(operator>)");
    }
    {
        SIMD4_32i  vec0(1, 2, -3, -5);
        int32_t val1 = -3;
        SIMDMask4 mask;
        mask = vec0.cmpgt(val1);
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == false && mask[3] == false, "CMPGTS");
    }
    {
        SIMD4_32i  vec0(1, 2, 3, -5);
        SIMD4_32i  vec1(1, 9, 0, -2);
        SIMDMask4 mask;
        mask = vec0.cmplt(vec1);
        CHECK_CONDITION(mask[0] == false && mask[1] == true && mask[2] == false && mask[3] == true, "CMPLTV");
    }
    {
        SIMD4_32i  vec0(1, 2, 3, -5);
        SIMD4_32i  vec1(1, 9, 0, -2);
        SIMDMask4 mask;
        mask = vec0 < vec1;
        CHECK_CONDITION(mask[0] == false && mask[1] == true && mask[2] == false && mask[3] == true, "CMPLTV(operator<)");
    }
    {
        SIMD4_32i  vec0(1, 2, -3, -5);
        int32_t val1 = -3;
        SIMDMask4 mask;
        mask = vec0.cmplt(val1);
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == false && mask[3] == true, "CMPLTS");
    }
    {
        SIMD4_32i  vec0(1, 2, -3, -5);
        SIMD4_32i  vec1(1, 9, -3, -2);
        SIMDMask4 mask;
        mask = vec0.cmpge(vec1);
        CHECK_CONDITION(mask[0] == true && mask[1] == false && mask[2] == true && mask[3] == false, "CMPGEV");
    }
    {
        SIMD4_32i  vec0(1, 2, -3, -5);
        SIMD4_32i  vec1(1, 9, -3, -2);
        SIMDMask4 mask;
        mask = vec0 >= vec1;
        CHECK_CONDITION(mask[0] == true && mask[1] == false && mask[2] == true && mask[3] == false, "CMPGEV(operator>=)");
    }
    {
        SIMD4_32i  vec0(1, 2, -3, -5);
        int32_t val1 = -3;
        SIMDMask4 mask;
        mask = vec0.cmpge(val1);
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == true && mask[3] == false, "CMPGES");
    }
    {
        SIMD4_32i  vec0(1, 2, -3, -5);
        SIMD4_32i  vec1(1, 9, -3, -7);
        SIMDMask4 mask;
        mask = vec0.cmple(vec1);
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == true && mask[3] == false, "CMPLEV");
    }
    {
        SIMD4_32i  vec0(1, 2, -3, -5);
        SIMD4_32i  vec1(1, 9, -3, -7);
        SIMDMask4 mask;
        mask = vec0 <= vec1;
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == true && mask[3] == false, "CMPLEV(operator<=)");
    }
    {
        SIMD4_32i  vec0(1, 2, -3, -5);
        int32_t val1 = -3;
        SIMDMask4 mask;
        mask = vec0.cmple(val1);
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == true && mask[3] == true, "CMPLES");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32i vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);
        SIMD4_32i vec2;

        vec2 = vec0.band(vec1);
        CHECK_CONDITION(
            vec2[0] == 0x01012000 && vec2[1] == 0x00000300 && vec2[2] == 0x09508060 && vec2[3] == 0x000F4020, 
            "BANDV");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32i vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);
        SIMD4_32i vec2;

        vec2 = vec0 & vec1;
        CHECK_CONDITION(
            vec2[0] == 0x01012000 && vec2[1] == 0x00000300 && vec2[2] == 0x09508060 && vec2[3] == 0x000F4020, 
            "BANDV(operator&)");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        int32_t val1 = 0x571FC5A0;
        SIMD4_32i vec2;

        vec2 = vec0.band(val1);
        CHECK_CONDITION(
            vec2[0] == 0x53130120 && vec2[1] == 0x500F0500 && vec2[2] == 0x0710C0A0 && vec2[3] == 0x000F4020, 
            "BANDS");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        int32_t val1 = 0x571FC5A0;
        SIMD4_32i vec2;
        SIMDMask4 mask(true, false, false, true);

        vec2 = vec0.band(mask, val1);
        CHECK_CONDITION(
            vec2[0] == 0x53130120 && vec2[1] == 0xF00F0F10 && vec2[2] == 0x0FF0F0F0 && vec2[3] == 0x000F4020, 
            "MBANDS");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32i vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);

        vec0.banda(vec1);
        CHECK_CONDITION(
            vec0[0] == 0x01012000 && vec0[1] == 0x00000300 && vec0[2] == 0x09508060 && vec0[3] == 0x000F4020, 
            "BANDVA");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32i vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);

        vec0 &= vec1;
        CHECK_CONDITION(
            vec0[0] == 0x01012000 && vec0[1] == 0x00000300 && vec0[2] == 0x09508060 && vec0[3] == 0x000F4020, 
            "BANDVA(operator&=)");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32i vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);
        SIMDMask4 mask(true, false, false, true);

        vec0.banda(mask, vec1);
        CHECK_CONDITION(
            vec0[0] == 0x01012000 && vec0[1] == 0xF00F0F10 && vec0[2] == 0x0FF0F0F0 && vec0[3] == 0x000F4020, 
            "MBANDVA");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        int32_t val1 = 0x571FC5A0;
        SIMDMask4 mask(true, false, false, true);
        
        vec0.banda(mask, val1);
        CHECK_CONDITION(
            vec0[0] == 0x53130120 && vec0[1] == 0xF00F0F10 && vec0[2] == 0x0FF0F0F0 && vec0[3] == 0x000F4020, 
            "MBANDSA");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32i vec1;
        vec1 = vec0.bnot();
        CHECK_CONDITION(
            vec1[0] == 0x0CCCCCCB && vec1[1] == 0x0FF0F0EF && vec1[2] == 0xF00F0F0F && vec1[3] == 0xFFF0BDC0, 
            "BNOT");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32i vec1;
        vec1 = ~vec0;
        CHECK_CONDITION(
            vec1[0] == 0x0CCCCCCB && vec1[1] == 0x0FF0F0EF && vec1[2] == 0xF00F0F0F && vec1[3] == 0xFFF0BDC0, 
            "BNOT(operator~)");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32i vec1;
        SIMDMask4 mask(true, false, true, false);
        vec1 = vec0.bnot(mask);
        CHECK_CONDITION(
            vec1[0] == 0x0CCCCCCB && vec1[1] == 0xF00F0F10 && vec1[2] == 0xF00F0F0F && vec1[3] == 0x000F423F, 
            "MBNOT");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMDMask4 mask(true, false, true, false);
        vec0.bnota(mask);
        CHECK_CONDITION(
            vec0[0] == 0x0CCCCCCB && vec0[1] == 0xF00F0F10 && vec0[2] == 0xF00F0F0F && vec0[3] == 0x000F423F, 
            "MBNOTA");
    }
    {
        SIMD4_32i vec0(3), vec1(-5);
        SIMD4_32i vec2(-2);
        SIMDMask4 mask(true, false, false, true);
        vec2 = vec0.blend(mask, vec1);
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == -5 && vec2[2] == -5 && vec2[3] == 3, "MBLENDV");
    }
    {
        SIMD4_32i vec0(3);
        int32_t val1 = -5;
        SIMD4_32i vec2(-2);
        SIMDMask4       mask(true, false, false, true);
        vec2 = vec0.blend(mask, val1);
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == -5 && vec2[2] == -5 && vec2[3] == 3, "MBLENDS");
    }
    {
        SIMD4_32i vec0(1, -2, 3, 4);
        int32_t val1 = 0;
        val1 = vec0.hmul();
        CHECK_CONDITION(val1 == -24, "HMUL");
    }
    {
        SIMD4_32i vec0(1, -2, 3, 4);
        SIMDMask4 mask(true, false, true, false);
        int32_t val1 = 0;
        val1 = vec0.hmul(mask);
        CHECK_CONDITION(val1 == 3, "MHMUL");
    }
    {
        SIMD4_32i vec0(1, -2, 3, 4);
        int32_t val1 = -42;
        int32_t res = 0;
        res = vec0.hmul(val1);
        CHECK_CONDITION(res == 1008, "HMULS");
    }
    {
        SIMD4_32i vec0(1, -2, 3, 4);
        SIMDMask4 mask(true, false, true, false);
        int32_t val1 = -42;
        int32_t res = 0;
        res = vec0.hmul(mask, val1);
        CHECK_CONDITION(res == -126, "HMULS");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xFF0F3F00, 0x0FF0F000, 0x0F0F720F);
        int32_t val1;
        val1 = vec0.hband();
        CHECK_CONDITION(val1 == 0x03003000, "HBAND");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(true, false, false, true);
        int32_t val1;
        val1 = vec0.hband(mask);
        CHECK_CONDITION(val1 == 0x00030204, "MHBAND");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xFF0F3F00, 0x0FF0F000, 0x0F0F720F);
        int32_t val1;
        int32_t val2 = 0x03003000;
        val1 = vec0.hband(val2);
        CHECK_CONDITION(val1 == 0x03003000, "HBANDS");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(true, false, false, true);
        int32_t val1;
        int32_t val2 = 0x00010004;
        val1 = vec0.hband(mask, val2);
        CHECK_CONDITION(val1 == 0x00010004, "MHBANDS");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        int32_t val1;
        val1 = vec0.hbor();
        CHECK_CONDITION(val1 == 0xFFFFFF0F, "HOR");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(false, false, true, true);
        int32_t val1;
        val1 = vec0.hbor(mask);
        CHECK_CONDITION(val1 == 0x0FFFF20F, "MHBOR");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        int32_t val1;
        int32_t val2 = 0x00000030;
        val1 = vec0.hbor(val2);
        CHECK_CONDITION(val1 == 0xFFFFFF3F, "HBORS");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(false, false, true, true);
        int32_t val1;
        int32_t val2 = 0x00000030;
        val1 = vec0.hbor(mask, val2);
        CHECK_CONDITION(val1 == 0x0FFFF23F, "MHBORS");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        int32_t val1;
        val1 = vec0.hbxor();
        CHECK_CONDITION(val1 == 0x0CC38E0B, "HBXOR");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(false, false, true, true);
        int32_t val1;
        val1 = vec0.hbxor(mask);
        CHECK_CONDITION(val1 == 0x0FFFB20F, "MHBXOR");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        int32_t val1;
        int32_t val2 = 0x00000030;
        val1 = vec0.hbxor(val2);
        CHECK_CONDITION(val1 == 0x0CC38E3B, "HBXORS");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(false, false, true, true);
        int32_t val1;
        int32_t val2 = 0x00000030;
        val1 = vec0.hbxor(mask, val2);
        CHECK_CONDITION(val1 == 0x0FFFB23F, "MHBXORS");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1;
        vec1 = vec0.neg();
        CHECK_CONDITION(vec1[0] == -9 && vec1[1] == 8 && vec1[2] == -7 && vec1[3] == -6, "NEG");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1;
        vec1 = -vec0;
        CHECK_CONDITION(vec1[0] == -9 && vec1[1] == 8 && vec1[2] == -7 && vec1[3] == -6, "NEG(operator-)");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1;
        vec1 = -vec0;
        CHECK_CONDITION(vec1[0] == -9 && vec1[1] == 8 && vec1[2] == -7 && vec1[3] == -6, "NEG(operator-)");
    }
    return g_failCount;
}

int test_UME_SIMD4_32f(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD4_32f test";
    INIT_TEST(header, supressMessages);

    {
        SIMD4_32f vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR"); 
    }    
    {
        SIMD4_32f vec0(3.14f);
        CHECK_CONDITION(vec0[3] == 3.14f, "SET-CONSTR");
    }
    {
        float arr[4] = {1.11f, 2.22f, 3.33f, 4.44f};
        SIMD4_32f vec0(arr);
        CHECK_CONDITION(vec0[3] == 4.44f && vec0[1] == 2.22f, "LOAD-CONSTR");
    }
    {
        SIMD4_32f vec0(1.11f, 2.22f, 3.33f, 4.44f);
        CHECK_CONDITION(vec0[3] == 4.44f && vec0[1] == 2.22f, "FULL-CONSTR");
    }
    {
        SIMD4_32f vec0(3.14f);
        SIMD4_32f vec1(2.71f);
        vec1 = vec0;
        CHECK_CONDITION(vec1[3] == 3.14f, "operator=");
    }
        
    genericFloatTest<SIMD4_32f, float, SIMD4_32i, SIMDMask4, 4, DataSet_1_32f>();

    {
        SIMD4_32f vec0(1.0f);
        SIMD4_32f vec1(2.0f);
        SIMD4_32f vec2;
        vec2 = vec0.add(vec1);
        CHECK_CONDITION(vec2[0] == 3.0f, "ADDV");
    }
    {
        SIMD4_32f vec0(1.0f);
        SIMD4_32f vec1(2.0f);
        SIMD4_32f vec2;
        vec2 = vec0 + vec1;
        CHECK_CONDITION(vec2[0] == 3.0f, "ADDV(operator+)");
    }
    {
        SIMD4_32f vec0(1.0f);
        SIMD4_32f vec1(2.0f);
        SIMD4_32f vec2;
        vec2 = addv(vec0, vec1);
        CHECK_CONDITION(vec2[0] == 3.0f, "ADDV function");
    }
    {
        SIMD4_32f vec0(1.0f);
        float val1 = 2.0f;
        SIMD4_32f vec2;
        vec2 = vec0.add(val1);
        CHECK_CONDITION(vec2[3] > 2.99f && vec2[3] < 3.01f, "ADDS");
    }
    {
        SIMD4_32f vec0(1.0f);
        SIMD4_32f vec2;
        vec2 = adds(vec0, 2.0f);
        CHECK_CONDITION(vec2[0] == 3.0f, "ADDS function");
    }
    {
        SIMD4_32f vec0(3.0f);
        SIMD4_32f vec1(4.0f);
        SIMD4_32f vec2;
        vec2 = vec0.mul(vec1);
        CHECK_CONDITION(vec2[1] > 11.99f && vec2[1] < 12.01f, "MULV");
    }
    {
        SIMD4_32f vec0(3.0f);
        SIMD4_32f vec1(4.0f);
        SIMD4_32f vec2;
        vec2 = vec0 * vec1;
        CHECK_CONDITION(vec2[1] > 11.99f && vec2[1] < 12.01f, "MULV(operator*)");
    }
    {
        SIMD4_32f vec0(3.0f);
        float val1 = 4.0f;
        SIMD4_32f vec2;
        vec2 = vec0.mul(val1);
        CHECK_CONDITION(vec2[1] > 11.99f && vec2[1] < 12.01f, "MULS");
    }
    {
        SIMD4_32f vec0(12.34f, 321.1231f, 0.321f, -0.045f);
        SIMD4_32f vec1;
        vec1 = vec0.rcp();
        CHECK_CONDITION( vec1[0] > 0.081f  && vec1[0] < 0.082f  &&
                         vec1[1] > 0.0031f && vec1[1] < 0.0032f &&
                         vec1[2] > 3.11f   && vec1[2] < 3.12f   &&
                         vec1[3] > -22.23  && vec1[3] < -22.21, "RCP");
    }
    {
        SIMD4_32f vec0(12.34f, 321.1231f, 0.321f, -0.045f);
        SIMD4_32f vec1;
        SIMDMask4 mask(true, false, false, true);
        vec1 = vec0.rcp(mask);
        CHECK_CONDITION( vec1[0] > 0.081f  && vec1[0] < 0.082f  &&
                         vec1[1] > 321.12f && vec1[1] < 321.13f &&
                         vec1[2] > 0.320f  && vec1[2] < 0.322f   &&
                         vec1[3] > -22.23  && vec1[3] < -22.21, "MRCP");
    }
    {
        SIMD4_32f vec0(12.34f, 321.1231f, 0.321f, -0.045f);
        SIMD4_32f vec1;
        vec1 = vec0.rcp(5.3f);
        CHECK_CONDITION( vec1[0] > 0.42f   && vec1[0] < 0.43f   &&
                         vec1[1] > 0.016f  && vec1[1] < 0.017f  &&
                         vec1[2] > 16.51f  && vec1[2] < 16.52f  &&
                         vec1[3] > -117.8f && vec1[3] < -117.7f, "RCPS");
    }
    {
        SIMD4_32f vec0(12.34f, 321.1231f, 0.321f, -0.045f);
        SIMD4_32f vec1;
        SIMDMask4 mask(true, false, true, false);
        vec1 = vec0.rcp(mask, 5.3f);
        CHECK_CONDITION( vec1[0] > 0.42f    && vec1[0] < 0.43f   &&
                         vec1[1] > 321.12f  && vec1[1] < 321.13f &&
                         vec1[2] > 16.51f   && vec1[2] < 16.52f  &&
                         vec1[3] > -0.046f  && vec1[3] < -0.044f, "MRCPS");
    }
    {
        SIMD4_32f vec0(12.34f, 321.1231f, 0.321f, -0.045f);
        vec0.rcpa();
        CHECK_CONDITION( vec0[0] > 0.081f  && vec0[0] < 0.082f  &&
                         vec0[1] > 0.0031f && vec0[1] < 0.0032f &&
                         vec0[2] > 3.11f   && vec0[2] < 3.12f   &&
                         vec0[3] > -22.23  && vec0[3] < -22.21, "RCPA");
    }
    {
        SIMD4_32f vec0(12.34f, 321.1231f, 0.321f, -0.045f);
        SIMDMask4 mask(true, false, false, true);
        vec0.rcpa(mask);
        CHECK_CONDITION( vec0[0] > 0.081f  && vec0[0] < 0.082f  &&
                         vec0[1] > 321.12f && vec0[1] < 321.13f &&
                         vec0[2] > 0.320f  && vec0[2] < 0.322f   &&
                         vec0[3] > -22.23  && vec0[3] < -22.21, "MRCPA");
    }
    {
        SIMD4_32f vec0(12.34f, 321.1231f, 0.321f, -0.045f);
        vec0.rcpa(5.3f);
        CHECK_CONDITION( vec0[0] > 0.42f   && vec0[0] < 0.43f   &&
                         vec0[1] > 0.016f  && vec0[1] < 0.017f  &&
                         vec0[2] > 16.51f  && vec0[2] < 16.52f  &&
                         vec0[3] > -117.8f && vec0[3] < -117.7f, "RCPSA");
    }
    {
        SIMD4_32f vec0(12.34f, 321.1231f, 0.321f, -0.045f);
        SIMDMask4 mask(true, false, true, false);
        vec0.rcpa(mask, 5.3f);
        CHECK_CONDITION( vec0[0] > 0.42f    && vec0[0] < 0.43f   &&
                         vec0[1] > 321.12f  && vec0[1] < 321.13f &&
                         vec0[2] > 16.51f   && vec0[2] < 16.52f  &&
                         vec0[3] > -0.046f  && vec0[3] < -0.044f, "MRCPSA");
    }

    {
        SIMD2_32f vec0(12.34f, 321.1231f);
        SIMD2_32f vec1(0.321f, -0.045f);
        float expected[4] = {12.34f, 321.1231f, 0.321f, -0.045f};
        float values[4];
        SIMD4_32f vec2(-1.0f);
        vec2.pack(vec0, vec1);
        vec2.store(values);
        CHECK_CONDITION(valuesInRange(values, expected, 4, 0.01f), "PACK");
    }
    {
        SIMD4_32f vec0(12.34f, 321.1231f, 0.321f, -0.045f);
        SIMD2_32f vec1, vec2;
        float expected1[2] = {12.34f, 321.1231f};
        float expected2[2] = {0.321f, -0.045f};
        float values1[2];
        float values2[2];

        vec0.unpack(vec1, vec2);
        
        vec1.store(values1);
        vec2.store(values2);
        CHECK_CONDITION(valuesInRange(values1, expected1, 2, 0.01f) &&
                        valuesInRange(values2, expected2, 2, 0.01f), "UNPACK");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -1420975.124f);
        SIMD4_32f vec1(87213.12496f, 9851.124987f, -18775.1667777f, -817641.124976f);
        SIMD4_32f vec2(98273.821753f, 147.194f, 1204987.659871f, -19874.111111f);
        SIMD4_32f vec3;
        float lowRange[4] = {10770755507.34620307f, -100462841.15076064568f, 
                             2331111618.2840661570f, 1161847679076.159986023f};
        float hiRange[4]  = {10770755507.34620309f, -100462841.15076064566f, 
                             2331111618.2840661572f, 1161847679076.159986025f};
        bool result = true;

        vec3 = vec0.fmuladd(vec1, vec2);
        for(uint32_t i = 0; i < 4; i++) if(vec3[i] < lowRange[i] || vec3[i] > hiRange[i]) result = false;

        CHECK_CONDITION(result, "FMULADDV");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -1420975.124f);
        SIMD4_32f vec1(87213.12496f, 9851.124987f, -18775.1667777f, -817641.124976f);
        SIMD4_32f vec2(98273.821753f, 147.194f, 1204987.659871f, -19874.111111f);
        SIMD4_32f vec3;
        float lowRange[4] =  {10770755507.34620307f, -10198.12342f, 
                             2331111618.2840661570f, -1420975.125f};
        float hiRange[4]  = {10770755507.34620309f, -10198.12340f, 
                             2331111618.2840661572f, -1420975.123f};
        bool result = true;
        SIMDMask4 mask(true, false, true, false);

        vec3 = vec0.fmuladd(mask, vec1, vec2);
        for(uint32_t i = 0; i < 4; i++) if(vec3[i] < lowRange[i] || vec3[i] > hiRange[i]) result = false;

        CHECK_CONDITION(result, "MFMULADDV");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -1420975.124f);
        SIMD4_32f vec1(87213.12496f, 9851.124987f, -18775.1667777f, -817641.124976f);
        SIMD4_32f vec2(98273.821753f, 147.194f, 1204987.659871f, -19874.111111f);
        SIMD4_32f vec3; 

        float lowRange[4] = { 1.07705e+10f,  -1.00464e+08f,
                              2.3286e+09f, 1.16184e+12f};
        float hiRange[4]  = { 1.07707e+10f,  -1.00462e+08f,
                              2.3288e+09f, 1.16186e+12f};
        bool result = true;

        vec3 = vec0.fmulsub(vec1, vec2);
        for(uint32_t i = 0; i < 4; i++) if(vec3[i] < lowRange[i] || vec3[i] > hiRange[i]) { result = false; break;};

        CHECK_CONDITION(result, "FMULSUBV");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -1420975.124f);
        SIMD4_32f vec1(87213.12496f, 9851.124987f, -18775.1667777f, -817641.124976f);
        SIMD4_32f vec2(98273.821753f, 147.194f, 1204987.659871f, -19874.111111f);
        SIMD4_32f vec3;
        float lowRange[4] = { 1.07705e+10f,  -10198.12342f,
                              2.3286e+09f,   -1420975.125f};
        float hiRange[4]  = { 1.07707e+10f,  -10198.12340f,
                              2.3288e+09f,   -1420975.123f};
        bool result = true;
        SIMDMask4 mask(true, false, true, false);

        vec3 = vec0.fmulsub(mask, vec1, vec2);
        for(uint32_t i = 0; i < 4; i++) if(vec3[i] < lowRange[i] || vec3[i] > hiRange[i]) { result = false; break;};

        CHECK_CONDITION(result, "MFMULSUBV");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -1420975.124f);
        SIMD4_32f vec1(87213.12496f, 9851.124987f, -18775.1667777f, -817641.124976f);
        SIMD4_32f vec2(98273.821753f, 147.194f, 1204987.659871f, -19874.111111f);
        SIMD4_32f vec3;
        float lowRange[4] = {2.070739e10f, -51076.1f, -1.72157e11f, 4.44905e10f};
        float hiRange[4]  = {2.070740e10f, -51076.0f, -1.72156e11f, 4.44906e10f};
        bool result = true;

        vec3 = vec0.faddmul(vec1, vec2);
        for(uint32_t i = 0; i < 4; i++) if(vec3[i] < lowRange[i] || vec3[i] > hiRange[i]) { result = false; break;};

        CHECK_CONDITION(result, "FADDMULV");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -1420975.124f);
        SIMD4_32f vec1(87213.12496f, 9851.124987f, -18775.1667777f, -817641.124976f);
        SIMD4_32f vec2(98273.821753f, 147.194f, 1204987.659871f, -19874.111111f);
        SIMD4_32f vec3;
        float lowRange[4] = {2.070739e10f, -10198.12342f, -1.72157e11f, -1420975.125f};
        float hiRange[4]  = {2.070740e10f, -10198.12340f, -1.72156e11f, -1420975.123f};
        bool result = true;
        SIMDMask4 mask(true, false, true, false);

        vec3 = vec0.faddmul(mask, vec1, vec2);
        for(uint32_t i = 0; i < 4; i++) if(vec3[i] < lowRange[i] || vec3[i] > hiRange[i]) { result = false; break;};

        CHECK_CONDITION(result, "MFADDMULV");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -1420975.124f);
        SIMD4_32f vec1(87213.12496f, 9851.124987f, -18775.1667777f, -817641.124976f);
        SIMD4_32f vec2(98273.821753f, 147.194f, 1204987.659871f, -19874.111111f);
        SIMD4_32f vec3;
        float lowRange[4] = {3.565865e9f, -2951130.0f, -1.269093e11f, 1.199072e10f};
        float hiRange[4]  = {3.565866e9f, -2951128.0f, -1.269092e11f, 1.199073e10f};
        bool result = true;

        vec3 = vec0.fsubmul(vec1, vec2);
        for(uint32_t i = 0; i < 4; i++) if(vec3[i] < lowRange[i] || vec3[i] > hiRange[i]) { result = false; break;};

        CHECK_CONDITION(result, "FSUBMULV");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -1420975.124f);
        SIMD4_32f vec1(87213.12496f, 9851.124987f, -18775.1667777f, -817641.124976f);
        SIMD4_32f vec2(98273.821753f, 147.194f, 1204987.659871f, -19874.111111f);
        SIMD4_32f vec3;
        float lowRange[4] = {3.565865e9f, -10198.12342f, -1.269093e11f, -1420975.125f};
        float hiRange[4]  = {3.565866e9f, -10198.12340f, -1.269092e11f, -1420975.123f};
        bool result = true;
        SIMDMask4 mask(true, false, true, false);

        vec3 = vec0.fsubmul(mask, vec1, vec2);
        for(uint32_t i = 0; i < 4; i++) if(vec3[i] < lowRange[i] || vec3[i] > hiRange[i]) { result = false; break;};

        CHECK_CONDITION(result, "MFSUBMULV");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -1420975.124f);
        SIMD4_32f vec1(87213.12496f, 9851.124987f, -18775.1667777f, -817641.124976f);
        SIMD4_32f vec2;
        float values[4] = {123498.125f, 9851.12500f, -18775.1660f, -817641.125f};
        bool result = true;
        vec2 = vec0.max(vec1);
        for(uint32_t i = 0; i < 4; i++) if(!valueInRange(vec2[i], values[i], 0.01f)) { result = false; break;};
        CHECK_CONDITION(result, "MAXV");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -1420975.124f);
        SIMD4_32f vec1(87213.12496f, 9851.124987f, -18775.1667777f, -817641.124976f);
        SIMD4_32f vec2;
        SIMDMask4 mask(true, false, true, false);
        float values[4] = {123498.125f, -10198.1230f, -18775.1660f, -1420975.13f};
        bool result = true;
        vec2 = vec0.max(mask, vec1);
        for(uint32_t i = 0; i < 4; i++) if(!valueInRange(vec2[i], values[i], 0.01f)) { result = false; break;};
        CHECK_CONDITION(result, "MMAXV");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -1420975.124f);
        float val1 = -120000.0f;
        SIMD4_32f vec2;
        float values[4] = {123498.125f, -10198.1230f, -120000.000f, -120000.000f};
        bool result = true;
        vec2 = vec0.max(val1);
        for(uint32_t i = 0; i < 4; i++) if(!valueInRange(vec2[i], values[i], 0.01f)) { result = false; break;};
        CHECK_CONDITION(result, "MAXS");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -1420975.124f);
        float val1 = -120000.0f;
        SIMD4_32f vec2;
        SIMDMask4 mask(true, false, true, false);
        float values[4] = {123498.125f, -10198.1230f, -120000.000f, -1420975.13f};
        bool result = true;
        vec2 = vec0.max(mask, val1);
        for(uint32_t i = 0; i < 4; i++) if(!valueInRange(vec2[i], values[i], 0.01f)) { result = false; break;};
        CHECK_CONDITION(result, "MMAXS");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -1420975.124f);
        SIMD4_32f vec1(87213.12496f, 9851.124987f, -18775.1667777f, -817641.124976f);
        float values[4] = {123498.125f, 9851.12500f, -18775.1660f, -817641.125f};
        bool result = true;
        vec0.maxa(vec1);
        for(uint32_t i = 0; i < 4; i++) if(!valueInRange(vec0[i], values[i], 0.01f)) { result = false; break;};
        CHECK_CONDITION(result, "MAXVA");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -1420975.124f);
        SIMD4_32f vec1(87213.12496f, 9851.124987f, -18775.1667777f, -817641.124976f);
        SIMDMask4 mask(true, false, true, false);
        float values[4] = {123498.125f, -10198.1230f, -18775.1660f, -1420975.13f};
        bool result = true;
        vec0.maxa(mask, vec1);
        for(uint32_t i = 0; i < 4; i++) if(!valueInRange(vec0[i], values[i], 0.01f)) { result = false; break;};
        CHECK_CONDITION(result, "MMAXVA");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -1420975.124f);
        float val1 = -120000.0f;
        float values[4] = {123498.125f, -10198.1230f, -120000.000f, -120000.000f};
        bool result = true;
        vec0.maxa(val1);
        for(uint32_t i = 0; i < 4; i++) if(!valueInRange(vec0[i], values[i], 0.01f)) { result = false; break;};
        CHECK_CONDITION(result, "MAXSA");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -1420975.124f);
        float val1 = -120000.0f;
        SIMDMask4 mask(true, false, true, false);
        float values[4] = {123498.125f, -10198.1230f, -120000.000f, -1420975.13f};
        bool result = true;
        vec0.maxa(mask, val1);
        for(uint32_t i = 0; i < 4; i++) if(!valueInRange(vec0[i], values[i], 0.01f)) { result = false; break;};
        CHECK_CONDITION(result, "MMAXSA");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -817641.124976f);
        SIMD4_32f vec1(87213.12496f, 9851.124987f, -18775.1667777f, -1420975.124f);
        SIMD4_32f vec2;
        float values[4] = { 87213.1250f, -10198.1230f, -124095.125f, -1420975.13f};
        bool result = true;
        vec2 = vec0.min(vec1);
        for(uint32_t i = 0; i < 4; i++) if(!valueInRange(vec2[i], values[i], 0.01f)) { result = false; break;};
        CHECK_CONDITION(result, "MINV");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -18775.1667777f, -817641.124976f);
        SIMD4_32f vec1(87213.12496f, 9851.124987f, -124095.123f, -1420975.124f);
        SIMD4_32f vec2;
        SIMDMask4 mask(true, false, true, false);
        float values[4] = { 87213.1250f, -10198.1230f, -124095.125f, -817641.125f};
        bool result = true;
        vec2 = vec0.min(mask, vec1);
        for(uint32_t i = 0; i < 4; i++) if(!valueInRange(vec2[i], values[i], 0.01f)) { result = false; break;};
        CHECK_CONDITION(result, "MMINV");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -817641.124976f);
        float val1 = -120000.0f;
        SIMD4_32f vec2;
        float values[4] = { -120000.000f, -120000.000f, -124095.125f, -817641.125f};
        bool result = true;
        vec2 = vec0.min(val1);
        for(uint32_t i = 0; i < 4; i++) if(!valueInRange(vec2[i], values[i], 0.01f)) { result = false; break;};
        CHECK_CONDITION(result, "MINS");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -817641.124976f);
        float val1 = -120000.0f;
        SIMD4_32f vec2;
        SIMDMask4 mask(true, false, true, false);
        float values[4] = { -120000.000f, -10198.1230f, -124095.125f, -817641.125f};
        bool result = true;
        vec2 = vec0.min(mask, val1);
        for(uint32_t i = 0; i < 4; i++) if(!valueInRange(vec2[i], values[i], 0.01f)) { result = false; break;};
        CHECK_CONDITION(result, "MMINS");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -817641.124976f);
        SIMD4_32f vec1(87213.12496f, 9851.124987f, -18775.1667777f, -1420975.124f);
        float values[4] = { 87213.1250f, -10198.1230f, -124095.125f, -1420975.13f};
        bool result = true;
        vec0.mina(vec1);
        for(uint32_t i = 0; i < 4; i++) if(!valueInRange(vec0[i], values[i], 0.01f)) { result = false; break;};
        CHECK_CONDITION(result, "MINVA");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -18775.1667777f, -817641.124976f);
        SIMD4_32f vec1(87213.12496f, 9851.124987f, -124095.123f, -1420975.124f);
        SIMDMask4 mask(true, false, true, false);
        float values[4] = { 87213.1250f, -10198.1230f, -124095.125f, -817641.125f};
        bool result = true;
        vec0.mina(mask, vec1);
        for(uint32_t i = 0; i < 4; i++) if(!valueInRange(vec0[i], values[i], 0.01f)) { result = false; break;};
        CHECK_CONDITION(result, "MMINVA");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -817641.124976f);
        float val1 = -120000.0f;
        float values[4] = { -120000.000f, -120000.000f, -124095.125f, -817641.125f};
        bool result = true;
        vec0.mina(val1);
        for(uint32_t i = 0; i < 4; i++) if(!valueInRange(vec0[i], values[i], 0.01f)) { result = false; break;};
        CHECK_CONDITION(result, "MINSA");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, -124095.123f, -817641.124976f);
        float val1 = -120000.0f;
        SIMDMask4 mask(true, false, true, false);
        float values[4] = { -120000.000f, -10198.1230f, -124095.125f, -817641.125f};
        bool result = true;
        vec0.mina(mask, val1);
        for(uint32_t i = 0; i < 4; i++) if(!valueInRange(vec0[i], values[i], 0.01f)) { result = false; break;};
        CHECK_CONDITION(result, "MMINSA");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, 817641.124976f, -124095.123f);
        float expected = 817641.124976f;
        float res;
        bool result = true;
        res = vec0.hmax();
        CHECK_CONDITION(valueInRange(res, expected, 0.01f), "HMAX");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, 817641.124976f, -124095.123f);
        SIMDMask4 mask(true, false, false, true);
        float expected = 123498.123f;
        float res;
        bool result = true;
        res = vec0.hmax(mask);
        CHECK_CONDITION(valueInRange(res, expected, 0.01f), "MHMAX");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, 817641.124976f, -124095.123f);
        uint32_t res;
        bool result = true;
        res = vec0.imax();
        CHECK_CONDITION(res == 2, "IMAX");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, 817641.124976f, -124095.123f);
        SIMDMask4 mask(true, false, false, true);
        uint32_t res;
        bool result = true;
        res = vec0.imax(mask);
        CHECK_CONDITION(res == 0, "MIMAX");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, 817641.124976f, -124095.123f);
        float expected = -124095.123f;
        float res;
        bool result = true;
        res = vec0.hmin();
        CHECK_CONDITION(valueInRange(res, expected, 0.01f), "HMIN");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, 817641.124976f, -124095.123f);
        SIMDMask4 mask(true, true, false, false);
        float expected = -10198.12341f;
        float res;
        bool result = true;
        res = vec0.hmin(mask);
        CHECK_CONDITION(valueInRange(res, expected, 0.01f), "MHMIN");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, 817641.124976f, -124095.123f);
        uint32_t res;
        bool result = true;
        res = vec0.imin();
        CHECK_CONDITION(res ==3, "IMIN");
    }
    {
        SIMD4_32f vec0(123498.123f, -10198.12341f, 817641.124976f, -124095.123f);
        SIMDMask4 mask(true, true, true, false);
        uint32_t res;
        bool result = true;
        res = vec0.imin(mask);
        CHECK_CONDITION(res == 1, "MIMIN");
    }
    {
        SIMD4_32f vec0(4.0f);
        SIMD4_32f vec1;
        vec1 = vec0.sqr();
        CHECK_CONDITION(vec1[0] > 15.99f && vec1[3] < 16.01f, "SQR");
    }
    {
        SIMD4_32f vec0(4.0f);
        SIMD4_32f vec1;
        SIMDMask4 mask(true, false, true, false);
        vec1 = vec0.sqr(mask);
        CHECK_CONDITION(vec1[0] > 15.99f && vec1[0] < 16.01f &&
                        vec1[1] > 3.99f  && vec1[1] < 4.01f, "MSQR");
    }
    {
        SIMD4_32f vec0(4.0f);
        vec0.sqra();
        CHECK_CONDITION(vec0[3] > 15.99f && vec0[3] < 16.01f, "SQRA");
    }
    {
        SIMD4_32f vec0(4.0f);
        SIMDMask4 mask(true, false, true, false);
        vec0.sqra(mask);
        CHECK_CONDITION(vec0[0] > 15.99f && vec0[0] < 16.01f &&
                        vec0[1] > 3.99f  && vec0[1] < 4.01f, "MSQRA");
    }
    {
        SIMD4_32f vec0(4.0f);
        SIMD4_32f vec1;
        vec1 = vec0.sqrt();
        CHECK_CONDITION(vec1[0] > 1.99f && vec1[3] < 2.01f, "SQRT")
    }
    {
        SIMD4_32f vec0(1.0f, 2.0f, -3.0f, -4.12f);
        SIMD4_32f vec1(2.15f, 3.79f, 3.00f, 2.00f);
        SIMD4_32f vec2;
        vec2 = vec0.pow(vec1);
        CHECK_CONDITION(vec2[0] > 0.99f   && vec2[0] < 1.01f    &&
                        vec2[1] > 13.83f  && vec2[1] < 13.84f   &&
                        vec2[2] > -27.01f && vec2[2] < -26.99f  &&
                        vec2[3] > 16.97   && vec2[3] < 16.98, "POWV");
    }
    {
        SIMD4_32f vec0(1.0f, 2.0f, -3.0f, -4.12f);
        SIMD4_32f vec1(2.15f, 3.79f, 3.00f, 2.00f);
        SIMD4_32f vec2;
        SIMDMask4 mask(true, false, true, false);
        vec2 = vec0.pow(mask, vec1);
        CHECK_CONDITION(vec2[0] > 0.99f   && vec2[0] < 1.01f    &&
                        vec2[1] > 1.99f   && vec2[1] < 2.01f    &&
                        vec2[2] > -27.01f && vec2[2] < -26.99f  &&
                        vec2[3] > -4.13f  && vec2[3] < -4.11f, "MPOWV");
    }
    {
        SIMD4_32f vec0(1.0f, 2.0f, -3.0f, -4.12f);
        float val1 = 3.0f;
        SIMD4_32f vec2;
        vec2 = vec0.pow(val1);
        CHECK_CONDITION(vec2[0] > 0.99f   && vec2[0] < 1.01f    &&
                        vec2[1] > 7.99f   && vec2[1] < 8.01f   &&
                        vec2[2] > -27.01f && vec2[2] < -26.99f  &&
                        vec2[3] > -69.94  && vec2[3] < -69.93, "POWS");
    }
    {
        SIMD4_32f vec0(1.0f, 2.0f, -3.0f, -4.12f);
        float val1 = 3.0f;
        SIMD4_32f vec2;
        SIMDMask4 mask(true, false, true, false);
        vec2 = vec0.pow(mask, val1);
        CHECK_CONDITION(vec2[0] > 0.99f   && vec2[0] < 1.01f    &&
                        vec2[1] > 1.99f   && vec2[1] < 2.01f    &&
                        vec2[2] > -27.01f && vec2[2] < -26.99f  &&
                        vec2[3] > -4.13f  && vec2[3] < -4.11f, "MPOWS");
    }
    {
        SIMD4_32f vec0(3.8f);
        SIMD4_32f vec1;
        vec1 = vec0.round();
        SIMD4_32i vec2;
        vec2 = vec0.trunc();
        CHECK_CONDITION(vec1[0] > 3.99f && vec1[3] < 4.01f, "ROUND");
        CHECK_CONDITION(vec2[0] == 3 && vec2[3] == 3, "TRUNC");
    }
    {   
        SIMD4_32f vec0(3.14f);
        SIMD4_32f vec1;
        vec1 = vec0.floor();
        CHECK_CONDITION(vec1[0] > 2.99f && vec1[0] < 3.01f && vec1[3] > 2.99f && vec1[3] < 3.01f, "FLOOR");
    }
    {   
        SIMD4_32f vec0(3.14f);
        SIMD4_32f vec1;
        SIMDMask4 mask(true, true, false, false);
        vec1 = vec0.floor(mask);
        CHECK_CONDITION(vec1[0] > 2.99f && vec1[0] < 3.01f && vec1[3] > 3.13f && vec1[3] < 3.15f, "MFLOOR");
    }
    {   
        SIMD4_32f vec0(3.14f);
        SIMD4_32f vec1;
        vec1 = vec0.ceil();
        CHECK_CONDITION(vec1[0] > 3.99f && vec1[0] < 4.01f && vec1[3] > 3.99f && vec1[3] < 4.01f, "CEIL");
    }
    {   
        SIMD4_32f vec0(3.14f);
        SIMD4_32f vec1;
        SIMDMask4 mask(true, true, false, false);
        vec1 = vec0.ceil(mask);
        CHECK_CONDITION(vec1[0] > 3.99f && vec1[0] < 4.01f && vec1[3] > 3.13f && vec1[3] < 3.15f, "MCEIL");
    }
    {
        alignas(16) uint32_t init[4] = { 0x7F800000, 0xFF800000, 0x7F800123, 0xFF700456 };
        SIMD4_32f vec0;
        SIMDMask4 mask;
        vec0.loada((const float*)&init[0]);
        mask = vec0.isfin();
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == false && mask[3] == true, "ISFIN");
    }
    {
        alignas(16) uint32_t init[4] = { 0x7F800000, 0xFF800000, 0x7F800123, 0xFF700456 };
        SIMD4_32f vec0;
        SIMDMask4 mask;
        vec0.loada((const float*)&init[0]);
        mask = vec0.isinf();
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == false && mask[3] == false, "ISINF");
    }
    {
        alignas(16) uint32_t init[4] = { 0x7F800000, 0xFF800001, 0x7F800123, 0xFF700456 };
        SIMD4_32f vec0;
        SIMDMask4 mask;
        vec0.loada((const float*)&init[0]);
        mask = vec0.isan();
        CHECK_CONDITION(mask[0] == true && mask[1] == false && mask[2] == false && mask[3] == true, "ISAN");
    }
    {
        alignas(16) uint32_t init[4] = { 0x7F800000, 0xFF800001, 0x7F800123, 0xFF700456 };
        SIMD4_32f vec0;
        SIMDMask4 mask;
        vec0.loada((const float*)&init[0]);
        mask = vec0.isnan();
        CHECK_CONDITION(mask[0] == false && mask[1] == true && mask[2] == true && mask[3] == false, "ISNAN");
    }
    {
        alignas(16) uint32_t init[4] = { 0x7F800000, 0xFF800001, 0x80700456, 0x80f00456 };
        SIMD4_32f vec0;
        SIMDMask4 mask;
        vec0.loada((const float*)&init[0]);
        mask = vec0.isnorm();
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == false && mask[3] == true, "ISNORM");
    }
    {
        alignas(16) uint32_t init[4] = { 0x7F800000, 0xFF800001, 0x80700456, 0x80f00456 };
        SIMD4_32f vec0;
        SIMDMask4 mask;
        vec0.loada((const float*)&init[0]);
        mask = vec0.issub();
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == true && mask[3] == false, "ISSUB");
    }
    {   
        SIMD4_32f vec0(2.14f, -12.34f, 9.23f, -256.3f);
        SIMD4_32f vec1;
        vec1 = vec0.sin();
        CHECK_CONDITION( vec1[0] > 0.842f && vec1[0] < 0.843f  &&
                         vec1[1] > 0.224f && vec1[1] < 0.225f  &&
                         vec1[2] > 0.193f && vec1[2] < 0.194f  &&
                         vec1[3] > 0.966f && vec1[3] < 0.967f, "SIN");
    }   
    {   
        SIMD4_32f vec0(2.14f, -12.34f, 9.23f, -256.3f);
        SIMD4_32f vec1;
        SIMDMask4 mask(true, true, false, false);
        vec1 = vec0.sin(mask);
        CHECK_CONDITION( vec1[0] > 0.842f  && vec1[0] < 0.843f  &&
                         vec1[1] > 0.224f  && vec1[1] < 0.225f  &&
                         vec1[2] > 9.22f   && vec1[2] < 9.24f   &&
                         vec1[3] > -256.4f && vec1[3] < -256.2f, "MSIN");

    }   
    {
        SIMD4_32f vec0(2.14f, -12.34f, 9.23f, -256.3f);
        SIMD4_32f vec1;
        vec1 = vec0.cos();
        CHECK_CONDITION( vec1[0] > -0.539f && vec1[0] < -0.538f  &&
                         vec1[1] > 0.974f  && vec1[1] < 0.975f   &&
                         vec1[2] > -0.982f && vec1[2] < -0.981f  &&
                         vec1[3] > 0.257f  && vec1[3] < 0.258f, "COS");
    }
    {
        SIMD4_32f vec0(2.14f, -12.34f, 9.23f, -256.3f);
        SIMD4_32f vec1;
        SIMDMask4 mask(true, true, false, false);
        vec1 = vec0.cos(mask);
        CHECK_CONDITION( vec1[0] > -0.539f && vec1[0] < -0.538f  &&
                         vec1[1] > 0.974f  && vec1[1] < 0.975f   &&
                         vec1[2] > 9.22f   && vec1[2] < 9.24f    &&
                         vec1[3] > -256.4f && vec1[3] < -256.2f, "MCOS");
    }
    {
        SIMD4_32f vec0(2.14f, -12.34f, 9.23f, -256.3f);
        SIMD4_32f vec1;
        vec1 = vec0.tan();
        CHECK_CONDITION( vec1[0] > -1.563f && vec1[0] < -1.562f  &&
                         vec1[1] > 0.230f  && vec1[1] < 0.231f   &&
                         vec1[2] > -0.198f && vec1[2] < -0.197f  &&
                         vec1[3] > 3.756f  && vec1[3] < 3.757f, "TAN");
    }
    {
        SIMD4_32f vec0(2.14f, -12.34f, 9.23f, -256.3f);
        SIMD4_32f vec1;
        SIMDMask4 mask(true, true, false, false);
        vec1 = vec0.tan(mask);
        CHECK_CONDITION( vec1[0] > -1.563f && vec1[0] < -1.562f  &&
                         vec1[1] > 0.230f  && vec1[1] < 0.231f   &&
                         vec1[2] > 9.22f   && vec1[2] < 9.24f    &&
                         vec1[3] > -256.4f && vec1[3] < -256.2f, "MTAN");
    }
    {
        SIMD4_32f vec0(2.14f, -12.34f, 9.23f, -256.3f);
        SIMD4_32f vec1;
        vec1 = vec0.ctan();
        CHECK_CONDITION( vec1[0] > -0.640f && vec1[0] < -0.639f  &&
                         vec1[1] > 4.341f  && vec1[1] < 4.342f   &&
                         vec1[2] > -5.069f && vec1[2] < -5.068f  &&
                         vec1[3] > 0.266f  && vec1[3] < 0.267f, "CTAN");
    }
    {
        SIMD4_32f vec0(2.14f, -12.34f, 9.23f, -256.3f);
        SIMD4_32f vec1;
        SIMDMask4 mask(true, true, false, false);
        vec1 = vec0.ctan(mask);
        CHECK_CONDITION( vec1[0] > -0.640f && vec1[0] < -0.639f  &&
                         vec1[1] > 4.341f  && vec1[1] < 4.342f   &&
                         vec1[2] > 9.22f   && vec1[2] < 9.24f    &&
                         vec1[3] > -256.4f && vec1[3] < -256.2f, "MCTAN");
    }

    return g_failCount;
}

int test_UME_SIMD2_64i(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD2_64i test";
    INIT_TEST(header, supressMessages);

    {
        SIMD2_64i vec0;  
        CHECK_CONDITION(true, "ZERO-CONSTR"); 
    }

    return g_failCount;
}

int test_UME_SIMD2_64u(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD2_64u test";
    INIT_TEST(header, supressMessages);

    {
        SIMD2_64u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR"); 
    }

    return g_failCount;
}

int test_UME_SIMD2_64f(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD2_64f test";
    INIT_TEST(header, supressMessages);

    {
        SIMD2_64f vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR"); 
    }

    return g_failCount;
}

#endif
