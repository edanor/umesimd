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
//  “ICE-DIP is a European Industrial Doctorate project funded by the European Community’s 
//  7th Framework programme Marie Curie Actions under grant PITN-GA-2012-316596”.
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
        CHECK_CONDITION(true, "ZERO-CONSTR"); 
    }
    
    {
        SIMD16_8u vec0;
        SIMD16_8u vec1(4);
        SIMD16_8u vec2(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15);
        
        vec0 = vec2.lsh(vec1);
        // LSHV
        CHECK_CONDITION(vec0[0] == 0 && vec0[1] == 16, "LSHV");
        vec2.lsh(4);
        // LSHS
        CHECK_CONDITION(vec0[0] == 0 && vec0[1] == 16, "LSHS");
        vec0 = vec2.rsh(4);
        // RSHS
        CHECK_CONDITION(vec0[0] == 0 && vec0[1] == 1, "RSHS");
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
        mask = vec0.cmpne(vec1); // 0x00
        CHECK_CONDITION(mask[0] == false && mask[7] == false, "CMPNEV");
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
        mask = vec0.cmplt(vec1); // 0x48
        CHECK_CONDITION(mask[3] == true && mask[5] == false, "CMPLTV");
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
        mask = vec0.cmple(vec1); // 0xD7
        CHECK_CONDITION(mask[3] == false && mask[6] == true, "CMPLEV");
    }
    {
        SIMD8_16u vec0(0xF2F1);
        SIMD8_16u vec1(0x2F1F);
        SIMD8_16u vec2;
        vec2 = vec0.andv(vec1);
        CHECK_CONDITION(vec2[0] == 0x2211, "ANDV");
    }
    {
        SIMD8_16u vec0(0xF2F1);
        SIMD8_16u vec1(0x2F1F);
        vec1.anda(vec0);
        CHECK_CONDITION(vec1[0] == 0x2211, "ANDVA");
    }
    {
        SIMD8_16u vec0(0x7281);
        SIMD8_16u vec1(0x2314);
        SIMD8_16u vec2;
        vec2 = vec0.orv(vec1);
        CHECK_CONDITION(vec2[0] == 0x7395, "ORV");
    }
    {
        SIMD8_16u vec0(0x7281);
        SIMD8_16u vec1(0x2314);
        vec0.ora(vec1);
        CHECK_CONDITION(vec0[0] == 0x7395, "ORVA");
    }
    {
        SIMD8_16u vec0(0x7281);
        SIMD8_16u vec1(0x2314);
        vec0.xora(vec1);
        CHECK_CONDITION(vec0[0] == 0x5195, "XORVA");
    }
    {
        SIMD8_16u vec0(0x7281);
        SIMD8_16u vec1(0x2314);
        vec0 = vec1.notv();
        CHECK_CONDITION(vec0[0] == 0xDCEB, "NOT");
    }
    {
        SIMD8_16u vec0(7);
        SIMD8_16u vec1(24);
        SIMDMask8 mask2(false, false, false, false, true, true, true, true);
        SIMD8_16u vec3 = vec0.blend(mask2, vec1);
        CHECK_CONDITION(vec3[0] == 24 && vec3[4] == 7, "MBLENDV");
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
        SIMD8_16i vec1(0);
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
        SIMD8_16i vec0(-1);
        SIMD8_16i vec1 = vec0.abs();

        CHECK_CONDITION(vec1[0] == vec1[7] == 1, "ABS");
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

    return g_failCount;
}

int test_UME_SIMD4_32u(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD4_32u test";
    
    INIT_TEST(header, supressMessages);

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
        uint32_t val1 = 12;
        vec0.adda(val1);
        CHECK_CONDITION(vec0[0] == 21 && vec0[1] == 20 && vec0[2] == 19 && vec0[3] == 18, "ADDSA");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(0);
        vec1 = vec0.postInc();
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == 9 && vec0[2] == 8 && vec0[3] == 7, "POSTINC 1");
        CHECK_CONDITION(vec1[0] == 9 && vec1[1] == 8 && vec1[2] == 7 && vec1[3] == 6, "POSTINC 2");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(0);
        vec1 = vec0.prefInc();
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == 9 && vec0[2] == 8 && vec0[3] == 7, "PREFINC 1");
        CHECK_CONDITION(vec1[0] == 10 && vec1[1] == 9 && vec1[2] == 8 && vec1[3] == 7, "PREFINC 2");
    }
    {
        SIMD4_32u vec0(9, 14, 28, 60);
        SIMD4_32u vec1(3, 8,   7,  6);
        SIMD4_32u vec2;
        vec2 = vec0.sub(vec1);
        CHECK_CONDITION(vec2[0] == 6 && vec2[1] == 6 && vec2[2] == 21 && vec2[3] == 54, "SUBV");
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
        SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 4;
        vec0.suba(val1);
        CHECK_CONDITION(vec0[0] == 5 && vec0[1] == 4 && vec0[2] == 3 && vec0[3] == 2, "SUBSA");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(0);
        vec1 = vec0.postDec();
        CHECK_CONDITION(vec0[0] == 8 && vec0[1] == 7 && vec0[2] == 6 && vec0[3] == 5, "POSTDEC 1");
        CHECK_CONDITION(vec1[0] == 9 && vec1[1] == 8 && vec1[2] == 7 && vec1[3] == 6, "POSTDEC 2");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(0);
        vec1 = vec0.prefDec();
        CHECK_CONDITION(vec0[0] == 8 && vec0[1] == 7 && vec0[2] == 6 && vec0[3] == 5, "PREFDEC 1");
        CHECK_CONDITION(vec1[0] == 8 && vec1[1] == 7 && vec1[2] == 6 && vec1[3] == 5, "PREFDEC 2");
    }
    {
        SIMD4_32u vec0(9, 8, 7, 6);
        SIMD4_32u vec1(0);
        SIMD4_32u vec2;
        vec2 = vec0.mul(vec1);
        CHECK_CONDITION(vec2[0] == 0 && vec2[1] == 0 && vec2[2] == 0 && vec2[3] == 0, "MULV 1");
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
        CHECK_CONDITION(vec2[0] == 0x8888888C && vec2[1] == 0x22222232 && vec2[2] == 0x888888C8 && vec2[3] == 0x88C88888, "MROLV");
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
        uint32_t val1 = 3;
        SIMDMask4 mask;
        mask = vec0.cmple(val1);
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == true && mask[3] == false, "CMPLES");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32u vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);
        SIMD4_32u vec2;

        vec2 = vec0.andv(vec1);
        CHECK_CONDITION(
            vec2[0] == 0x01012000 && vec2[1] == 0x00000300 && vec2[2] == 0x09508060 && vec2[3] == 0x000F4020, 
            "ANDV");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        uint32_t val1 = 0x571FC5A0;
        SIMD4_32u vec2;

        vec2 = vec0.ands(val1);
        CHECK_CONDITION(
            vec2[0] == 0x53130120 && vec2[1] == 0x500F0500 && vec2[2] == 0x0710C0A0 && vec2[3] == 0x000F4020, 
            "ANDS");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        uint32_t val1 = 0x571FC5A0;
        SIMD4_32u vec2;
        SIMDMask4 mask(true, false, false, true);

        vec2 = vec0.ands(mask, val1);
        CHECK_CONDITION(
            vec2[0] == 0x53130120 && vec2[1] == 0xF00F0F10 && vec2[2] == 0x0FF0F0F0 && vec2[3] == 0x000F4020, 
            "MANDS");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32u vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);

        vec0.anda(vec1);
        CHECK_CONDITION(
            vec0[0] == 0x01012000 && vec0[1] == 0x00000300 && vec0[2] == 0x09508060 && vec0[3] == 0x000F4020, 
            "ANDVA");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32u vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);
        SIMDMask4 mask(true, false, false, true);

        vec0.anda(mask, vec1);
        CHECK_CONDITION(
            vec0[0] == 0x01012000 && vec0[1] == 0xF00F0F10 && vec0[2] == 0x0FF0F0F0 && vec0[3] == 0x000F4020, 
            "MANDVA");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        uint32_t val1 = 0x571FC5A0;
        SIMDMask4 mask(true, false, false, true);
                
        vec0.anda(mask, val1);
        CHECK_CONDITION(
            vec0[0] == 0x53130120 && vec0[1] == 0xF00F0F10 && vec0[2] == 0x0FF0F0F0 && vec0[3] == 0x000F4020, 
            "MANDSA");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32u vec1;
        vec1 = vec0.notv();
        CHECK_CONDITION(
            vec1[0] == 0x0CCCCCCB && vec1[1] == 0x0FF0F0EF && vec1[2] == 0xF00F0F0F && vec1[3] == 0xFFF0BDC0, 
            "NOT");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32u vec1;
        SIMDMask4 mask(true, false, true, false);
        vec1 = vec0.notv(mask);
        CHECK_CONDITION(
            vec1[0] == 0x0CCCCCCB && vec1[1] == 0xF00F0F10 && vec1[2] == 0xF00F0F0F && vec1[3] == 0x000F423F, 
            "MNOT");
    }
    {
        SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMDMask4 mask(true, false, true, false);
        vec0.nota(mask);
        CHECK_CONDITION(
            vec0[0] == 0x0CCCCCCB && vec0[1] == 0xF00F0F10 && vec0[2] == 0xF00F0F0F && vec0[3] == 0x000F423F, 
            "MNOTA");
    }
    {
        SIMD4_32u vec0(3), vec1(5);
        SIMD4_32u vec2(2);
        SIMDMask4       mask(true, false, false, true);
        vec2 = vec0.blend(mask, vec1);
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == 5 && vec2[2] == 5 && vec2[3] == 3, "MBLENDV");
    }
    {
        SIMD4_32u vec0(3);
        uint32_t val1 = 5;
        SIMD4_32u vec2(2);
        SIMDMask4       mask(true, false, false, true);
    
        vec2 = vec0.blend(mask, val1);
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == 5 && vec2[2] == 5 && vec2[3] == 3, "MBLENDS");
    }
    {
        SIMD4_32u vec0(0xF3333304, 0xFF0F3F00, 0x0FF0F000, 0x0F0F720F);
        uint32_t val1;
        val1 = vec0.hand();
        CHECK_CONDITION(val1 == 0x03003000, "HAND");
    }
    {
        SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(true, false, false, true);
        uint32_t val1;
        val1 = vec0.hand(mask);
        CHECK_CONDITION(val1 == 0x00030204, "MHAND");
    }
    {
        SIMD4_32u vec0(0xF3333304, 0xFF0F3F00, 0x0FF0F000, 0x0F0F720F);
        uint32_t val1;
        uint32_t val2 = 0x03003000;
        val1 = vec0.hand(val2);
        CHECK_CONDITION(val1 == 0x03003000, "HANDS");
    }
    {
        SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(true, false, false, true);
        uint32_t val1;
        uint32_t val2 = 0x00010004;
        val1 = vec0.hand(mask, val2);
        CHECK_CONDITION(val1 == 0x00010004, "MHANDS");
    }
    {
        SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        uint32_t val1;
        val1 = vec0.hor();
        CHECK_CONDITION(val1 == 0xFFFFFF0F, "HOR");
    }
    {
        SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(false, false, true, true);
        uint32_t val1;
        val1 = vec0.hor(mask);
        CHECK_CONDITION(val1 == 0x0FFFF20F, "MHOR");
    }
    {
        SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        uint32_t val1;
        uint32_t val2 = 0x00000030;
        val1 = vec0.hor(val2);
        CHECK_CONDITION(val1 == 0xFFFFFF3F, "HORS");
    }
    {
        SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(false, false, true, true);
        uint32_t val1;
        uint32_t val2 = 0x00000030;
        val1 = vec0.hor(mask, val2);
        CHECK_CONDITION(val1 == 0x0FFFF23F, "MHORS");
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
        int32_t val1 = 12;
        vec0.adda(val1);
        CHECK_CONDITION(vec0[0] == 21 && vec0[1] == 4 && vec0[2] == 19 && vec0[3] == 18, "ADDSA");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(0);
        vec1 = vec0.postInc();
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == -7 && vec0[2] == 8 && vec0[3] == 7, "POSTINC 1");
        CHECK_CONDITION(vec1[0] == 9 && vec1[1] == -8 && vec1[2] == 7 && vec1[3] == 6, "POSTINC 2");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(0);
        vec1 = vec0.prefInc();
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == -7 && vec0[2] == 8 && vec0[3] == 7, "PREFINC 1");
        CHECK_CONDITION(vec1[0] == 10 && vec1[1] == -7 && vec1[2] == 8 && vec1[3] == 7, "PREFINC 2");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(3, 14, 28, -60);
        SIMD4_32i vec2;
        vec2 = vec0.sub(vec1);
        CHECK_CONDITION(vec2[0] == 6 && vec2[1] == -22 && vec2[2] == -21 && vec2[3] == 66, "SUBV");
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
        int32_t val1 = 12;
        vec0.suba(val1);
        CHECK_CONDITION(vec0[0] == -3 && vec0[1] == -20 && vec0[2] == -5 && vec0[3] == -6, "SUBSA");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(0);
        vec1 = vec0.postDec();
        CHECK_CONDITION(vec0[0] == 8 && vec0[1] == -9 && vec0[2] == 6 && vec0[3] == 5, "POSTDEC 1");
        CHECK_CONDITION(vec1[0] == 9 && vec1[1] == -8 && vec1[2] == 7 && vec1[3] == 6, "POSTDEC 2");
    }
    {
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1(0);
        vec1 = vec0.prefDec();
        CHECK_CONDITION(vec0[0] == 8 && vec0[1] == -9 && vec0[2] == 6 && vec0[3] == 5, "PREFDEC 1");
        CHECK_CONDITION(vec1[0] == 8 && vec1[1] == -9 && vec1[2] == 6 && vec1[3] == 5, "PREFDEC 2");
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
        SIMD4_32i vec1(-3);
        SIMD4_32i vec2;
        vec2 = vec0.mul(vec1);
        CHECK_CONDITION(vec2[0] == -27 && vec2[1] == -24 && vec2[2] == -21 && vec2[3] == -18, "MULV 2");
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
        CHECK_CONDITION(vec2[0] == 18 && vec2[1] == 32 && vec2[2] == -56 && vec2[3] == 96, "LSHV");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        SIMD4_32u vec1(1, 2, 3, 4);
        SIMD4_32i vec2;
        SIMDMask4       mask(true, false, false, true);
        vec2 = vec0.lsh(mask, vec1);
        CHECK_CONDITION(vec2[0] == 18 && vec2[1] == 8 && vec2[2] == -7 && vec2[3] == 96, "MLSHV");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        uint32_t val1 = 3;
        SIMD4_32i vec2;
        vec2 = vec0.lsh(val1);
        CHECK_CONDITION(vec2[0] == 72 && vec2[1] == 64 && vec2[2] == -56 && vec2[3] == 48, "LSHS");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        uint32_t val1 = 3;
        SIMD4_32i vec2;
        SIMDMask4       mask(true, false, false, true);
        vec2 = vec0.lsh(mask, val1);
        CHECK_CONDITION(vec2[0] == 72 && vec2[1] == 8 && vec2[2] == -7 && vec2[3] == 48, "MLSHS");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        SIMD4_32u vec1(1, 2, 3, 4);
        vec0.lsha(vec1);
        CHECK_CONDITION(vec0[0] == 18 && vec0[1] == 32 && vec0[2] == -56 && vec0[3] == 96, "LSHVA");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        SIMD4_32u vec1(1, 2, 3, 4);
        SIMDMask4       mask(true, false, true, false);
        vec0.lsha(mask, vec1);
        CHECK_CONDITION(vec0[0] == 18 && vec0[1] == 8 && vec0[2] == -56 && vec0[3] == 6, "MLSHVA");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        uint32_t val1 = 3;
        vec0.lsha(val1);
        CHECK_CONDITION(vec0[0] == 72 && vec0[1] == 64 && vec0[2] == -56 && vec0[3] == 48, "LSHSA");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        SIMDMask4       mask(true, false, true, false);
        uint32_t val1 = 3;
        vec0.lsha(mask, val1);
        CHECK_CONDITION(vec0[0] == 72 && vec0[1] == 8 && vec0[2] == -56 && vec0[3] == 6, "MLSHSA");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        SIMD4_32u vec1(1, 2, 3, 4);
        SIMD4_32i vec2;
        vec2 = vec0.rsh(vec1);
        CHECK_CONDITION(vec2[0] == 4 && vec2[1] == 2 && vec2[2] == -1 && vec2[3] == 0, "RSHV");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        uint32_t val1 = 3;
        SIMD4_32i vec2;
        vec2 = vec0.rsh(val1);
        CHECK_CONDITION(vec2[0] == 1 && vec2[1] == 1 && vec2[2] == -1 && vec2[3] == 0, "RSHS");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        SIMD4_32u vec1(1, 2, 3, 4);
        vec0.rsha(vec1);
        CHECK_CONDITION(vec0[0] == 4 && vec0[1] == 2 && vec0[2] == -1 && vec0[3] == 0, "RSHVA");
    }
    {
        SIMD4_32i vec0(9, 8, -7, 6);
        uint32_t val1 = 3;
        vec0.rsha(val1);
        CHECK_CONDITION(vec0[0] == 1 && vec0[1] == 1 && vec0[2] == -1 && vec0[3] == 0, "RSHSA");
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
        int32_t val1 = -3;
        SIMDMask4 mask;
        mask = vec0.cmple(val1);
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == true && mask[3] == true, "CMPLES");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32i vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);
        SIMD4_32i vec2;

        vec2 = vec0.andv(vec1);
        CHECK_CONDITION(
            vec2[0] == 0x01012000 && vec2[1] == 0x00000300 && vec2[2] == 0x09508060 && vec2[3] == 0x000F4020, 
            "ANDV");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        int32_t val1 = 0x571FC5A0;
        SIMD4_32i vec2;

        vec2 = vec0.ands(val1);
        CHECK_CONDITION(
            vec2[0] == 0x53130120 && vec2[1] == 0x500F0500 && vec2[2] == 0x0710C0A0 && vec2[3] == 0x000F4020, 
            "ANDS");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        int32_t val1 = 0x571FC5A0;
        SIMD4_32i vec2;
        SIMDMask4 mask(true, false, false, true);

        vec2 = vec0.ands(mask, val1);
        CHECK_CONDITION(
            vec2[0] == 0x53130120 && vec2[1] == 0xF00F0F10 && vec2[2] == 0x0FF0F0F0 && vec2[3] == 0x000F4020, 
            "MANDS");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32i vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);

        vec0.anda(vec1);
        CHECK_CONDITION(
            vec0[0] == 0x01012000 && vec0[1] == 0x00000300 && vec0[2] == 0x09508060 && vec0[3] == 0x000F4020, 
            "ANDVA");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32i vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);
        SIMDMask4 mask(true, false, false, true);

        vec0.anda(mask, vec1);
        CHECK_CONDITION(
            vec0[0] == 0x01012000 && vec0[1] == 0xF00F0F10 && vec0[2] == 0x0FF0F0F0 && vec0[3] == 0x000F4020, 
            "MANDVA");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        int32_t val1 = 0x571FC5A0;
        SIMDMask4 mask(true, false, false, true);
        
        vec0.anda(mask, val1);
        CHECK_CONDITION(
            vec0[0] == 0x53130120 && vec0[1] == 0xF00F0F10 && vec0[2] == 0x0FF0F0F0 && vec0[3] == 0x000F4020, 
            "MANDSA");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32i vec1;
        vec1 = vec0.notv();
        CHECK_CONDITION(
            vec1[0] == 0x0CCCCCCB && vec1[1] == 0x0FF0F0EF && vec1[2] == 0xF00F0F0F && vec1[3] == 0xFFF0BDC0, 
            "NOT");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMD4_32i vec1;
        SIMDMask4 mask(true, false, true, false);
        vec1 = vec0.notv(mask);
        CHECK_CONDITION(
            vec1[0] == 0x0CCCCCCB && vec1[1] == 0xF00F0F10 && vec1[2] == 0xF00F0F0F && vec1[3] == 0x000F423F, 
            "MNOT");
    }
    {
        SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        SIMDMask4 mask(true, false, true, false);
        vec0.nota(mask);
        CHECK_CONDITION(
            vec0[0] == 0x0CCCCCCB && vec0[1] == 0xF00F0F10 && vec0[2] == 0xF00F0F0F && vec0[3] == 0x000F423F, 
            "MNOTA");
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
        SIMD4_32i vec0(0xF3333304, 0xFF0F3F00, 0x0FF0F000, 0x0F0F720F);
        int32_t val1;
        val1 = vec0.hand();
        CHECK_CONDITION(val1 == 0x03003000, "HAND");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(true, false, false, true);
        int32_t val1;
        val1 = vec0.hand(mask);
        CHECK_CONDITION(val1 == 0x00030204, "MHAND");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xFF0F3F00, 0x0FF0F000, 0x0F0F720F);
        int32_t val1;
        int32_t val2 = 0x03003000;
        val1 = vec0.hand(val2);
        CHECK_CONDITION(val1 == 0x03003000, "HANDS");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(true, false, false, true);
        int32_t val1;
        int32_t val2 = 0x00010004;
        val1 = vec0.hand(mask, val2);
        CHECK_CONDITION(val1 == 0x00010004, "MHANDS");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        int32_t val1;
        val1 = vec0.hor();
        CHECK_CONDITION(val1 == 0xFFFFFF0F, "HOR");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(false, false, true, true);
        int32_t val1;
        val1 = vec0.hor(mask);
        CHECK_CONDITION(val1 == 0x0FFFF20F, "MHOR");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        int32_t val1;
        int32_t val2 = 0x00000030;
        val1 = vec0.hor(val2);
        CHECK_CONDITION(val1 == 0xFFFFFF3F, "HORS");
    }
    {
        SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        SIMDMask4 mask(false, false, true, true);
        int32_t val1;
        int32_t val2 = 0x00000030;
        val1 = vec0.hor(mask, val2);
        CHECK_CONDITION(val1 == 0x0FFFF23F, "MHORS");
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
        SIMD4_32i vec0(9, -8, 7, 6);
        SIMD4_32i vec1;
        vec1 = vec0.neg();
        CHECK_CONDITION(vec1[0] == -9 && vec1[1] == 8 && vec1[2] == -7 && vec1[3] == -6, "NEG");
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
        SIMD4_32f vec0(1.11f, 2.22f, 3.33f, 4.44f);
        CHECK_CONDITION(vec0[3] == 4.44f && vec0[1] == 2.22f, "FULL-CONSTR");
    }
    {
        SIMD4_32f vec0(3.14f);
        SIMD4_32f vec1(2.71f);
        vec1 = vec0;
        CHECK_CONDITION(vec1[3] == 3.14f, "operator=");
    }
    {
        SIMD4_32f vec0(1.0f);
        SIMD4_32f vec1(2.0f);
        SIMD4_32f vec2;
        vec2 = vec0.add(vec1);
        CHECK_CONDITION(vec2[0] == 3.0f, "ADDV");
    }
    {
        SIMD4_32f vec0(1.0f);
        float val1 = 2.0f;
        SIMD4_32f vec2;
        vec2 = vec0.add(val1);
        CHECK_CONDITION(vec2[3] > 2.99f && vec2[3] < 3.01f, "ADDS");
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
        float val1 = 4.0f;
        SIMD4_32f vec2;
        vec2 = vec0.mul(val1);
        CHECK_CONDITION(vec2[1] > 11.99f && vec2[1] < 12.01f, "MULS");
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
        vec1 = vec0.sqrt();
        CHECK_CONDITION(vec1[0] > 1.99f && vec1[3] < 2.01f, "SQRT")
    }

    {
        SIMD4_32f vec0(3.8f);
        SIMD4_32f vec1;
        //vec1 = round(vec0);
        SIMD4_32f vec2;
        //vec2 = truncate(vec0);
        CHECK_CONDITION(vec1[0] > 3.99f && vec1[3] < 4.01f, "ROUND");
        CHECK_CONDITION(vec2[0] > 2.99f && vec2[3] < 3.01f, "TRUNC");
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
