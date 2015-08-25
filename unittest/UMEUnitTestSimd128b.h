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


// 128 bit integer vectors

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

int test_UME_SIMD16_8u(bool supressMessages) 
{
    char header[] = "UME::SIMD::SIMD16_8u test";
    INIT_TEST(header, supressMessages);
    
    {
        UME::SIMD::SIMD16_8u vec1;
        CHECK_CONDITION(true, "SIMD16_8u()"); 
    }
    
    {
        UME::SIMD::SIMD16_8u vec0;
        UME::SIMD::SIMD16_8u vec1(4);
        UME::SIMD::SIMD16_8u vec2(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15);
        
        vec0 = vec2.lsh(vec1);
        // LSHV
        CHECK_CONDITION(vec0[0] == 0 && vec0[1] == 16, "SIMD16_8u::LSHV");
        vec2.lsh(4);
        // LSHS
        CHECK_CONDITION(vec0[0] == 0 && vec0[1] == 16, "SIMD16_8u::LSHS");
        vec0 = vec2.rsh(4);
        // RSHS
        CHECK_CONDITION(vec0[0] == 0 && vec0[1] == 1, "SIMD16_8u::RSHS");
    }
    
    return g_failCount;
}

int test_UME_SIMD16_8i(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD16_8i test";
    INIT_TEST(header, supressMessages);
    
    {
        UME::SIMD::SIMD16_8i vec0; 
        CHECK_CONDITION(true, "SIMD16_8i()"); 
    }
    
    return g_failCount;
}


int test_UME_SIMD16_8(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD16_8 test";
    INIT_TEST(header, supressMessages);

    int fail_u = test_UME_SIMD16_8u(supressMessages);
    int fail_i = test_UME_SIMD16_8i(supressMessages);

    return g_failCount;
}

int test_UME_SIMD8_16(bool supressMessages)
{
    char header[] = "UME i::SIMD::SIMD8_16 test";
    INIT_TEST(header, supressMessages);
    
    {
        UME::SIMD::SIMD8_16u vec3;
    }
    {
        CHECK_CONDITION(true, "SIMD8_16u()"); 
        UME::SIMD::SIMD8_16i vec2;
    }
    {
        UME::SIMD::SIMD8_16i vec0(1, -2, 3, 4, 5, 6, 7, 8);
        UME::SIMD::SIMD8_16i vec1(15);

        vec1 = vec0;
        CHECK_CONDITION(vec1[0] == 1 && vec1[1] == -2, "SIMD8_16i::operator=");
    }

    {
        UME::SIMD::SIMD8_16u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        UME::SIMD::SIMD8_16u vec1(15);

        vec1 = vec0;
        CHECK_CONDITION(vec1[0] == 1 && vec1[1] == 2, "SIMD8_16u::operator=");
    }
    // LOAD
    {
        UME::SIMD::SIMD8_16i vec0(1, -2, 3, 4, 5, 6, 7, 8);
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
        CHECK_CONDITION(res, "SIMD8_16i::LOAD");
    }
    // LOADA
    {
        UME::SIMD::SIMD8_16i vec0(1, -2, 3, 4, 5, 6, 7, 8);
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
        CHECK_CONDITION(res, "SIMD8_16i::LOADA");
    }
    // LSHSA
    {
        UME::SIMD::SIMD8_16i vec0(1, 2, 3, 4, 5, 6, 7, 8);
        UME::SIMD::SIMD8_16i vec1(0);
        int16_t vals[] = { 4, 8, 12, 16, 20, 24, 28, 32 };
        bool res = true;
        vec1 = vec0.lsh(2);
        for(uint32_t i = 0; i < 8; i++)if(vec1[i] != vals[i]){res = false; break;}
        CHECK_CONDITION(res, "SIMD8_16i::LSHS");
    }
    // LSHSA
    {
        UME::SIMD::SIMD8_16i vec0(1, 2, 3, 4, 5, 6, 7, 8);
        int16_t vals[] = { 4, 8, 12, 16, 20, 24, 28, 32 };
        bool res = true;
        vec0.lsha(2);
        for(uint32_t i = 0; i < 8; i++)if(vec0[i] != vals[i]){res = false; break;}
        CHECK_CONDITION(res, "SIMD8_16i::LSHSA");
    }
    // RSHSA
    {
        UME::SIMD::SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        int16_t vals[] = {1, 2, 3, 4, 5, 6, 7, 8};
        bool res = true;
        vec0.rsha(2);
        for(uint32_t i = 0; i < 8; i++)if(vec0[i] != vals[i]){res = false; break;}
        CHECK_CONDITION(res, "SIMD8_16u::RSHSA");
    }
    // CMPEQV
    {
        UME::SIMD::SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        UME::SIMD::SIMD8_16u vec1(1, 2, 3, 4, 5, 6, 7, 8);
        vec0.rsha(2);
        UME::SIMD::SIMDMask8 mask; 
        mask = vec0.cmpeq(vec1); // 0xFF
        CHECK_CONDITION(mask[0] == true && mask[7] == true, "SIMD8_16u::CMPEQV");
    }
    // CMPNEV
    {
        UME::SIMD::SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        UME::SIMD::SIMD8_16u vec1(1, 2, 3, 4, 5, 6, 7, 8);
        vec0.rsha(2);
        UME::SIMD::SIMDMask8 mask;
        mask = vec0.cmpne(vec1); // 0x00
        CHECK_CONDITION(mask[0] == false && mask[7] == false, "SIMD8_16u::CMPNEV");
    }
    // CMPGTV
    {
        UME::SIMD::SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        UME::SIMD::SIMD8_16u vec1(1, 2, 3, 5, 5, 6, 8, 8);
        vec0.rsha(2);
        UME::SIMD::SIMDMask8 mask;
        mask = vec1.cmpgt(vec0); // 0x48
        CHECK_CONDITION(mask[3] == true && mask[5] == false, "SIMD8_16u::CMPGTV");
    }
    // CMPLTV
    {
        UME::SIMD::SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        UME::SIMD::SIMD8_16u vec1(1, 2, 3, 5, 5, 6, 8, 8);
        vec0.rsha(2);
        UME::SIMD::SIMDMask8 mask;
        mask = vec0.cmplt(vec1); // 0x48
        CHECK_CONDITION(mask[3] == true && mask[5] == false, "SIMD8_16u::CMPLTV");
    }
    // CMPGEV
    {
        UME::SIMD::SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        UME::SIMD::SIMD8_16u vec1(1, 2, 3, 3, 5, 5, 8, 8);
        vec0.rsha(2);
        UME::SIMD::SIMDMask8 mask;
        mask = vec0.cmpge(vec1); // 0xBF
        CHECK_CONDITION(mask[2] == true && mask[6] == false, "SIMD8_16u::CMPGEV");
    }
    // CMPLEV
    {
        UME::SIMD::SIMD8_16u vec0(4, 8, 12, 16, 20, 24, 28, 32);
        UME::SIMD::SIMD8_16u vec1(1, 2, 3, 3, 5, 5, 8, 8);
        vec0.rsha(2);
        UME::SIMD::SIMDMask8 mask;
        mask = vec0.cmple(vec1); // 0xD7
        CHECK_CONDITION(mask[3] == false && mask[6] == true, "SIMD8_16u::CMPLEV");
    }
    // ANDV
    {
        UME::SIMD::SIMD8_16u vec0(0xF2F1);
        UME::SIMD::SIMD8_16u vec1(0x2F1F);
        UME::SIMD::SIMD8_16u vec2;
        vec2 = vec0.andv(vec1);
        CHECK_CONDITION(vec2[0] == 0x2211, "SIMD8_16u::ANDV");
    }
    // ANDVA
    {
        UME::SIMD::SIMD8_16u vec0(0xF2F1);
        UME::SIMD::SIMD8_16u vec1(0x2F1F);
        vec1.anda(vec0);
        CHECK_CONDITION(vec1[0] == 0x2211, "SIMD8_16u::ANDVA");
    }
    // ORV
    {
        UME::SIMD::SIMD8_16u vec0(0x7281);
        UME::SIMD::SIMD8_16u vec1(0x2314);
        UME::SIMD::SIMD8_16u vec2;
        vec2 = vec0.orv(vec1);
        CHECK_CONDITION(vec2[0] == 0x7395, "SIMD8_16u::ORV");
    }
    // ORVA
    {
        UME::SIMD::SIMD8_16u vec0(0x7281);
        UME::SIMD::SIMD8_16u vec1(0x2314);
        vec0.ora(vec1);
        CHECK_CONDITION(vec0[0] == 0x7395, "SIMD8_16u::ORVA");
    }
    // XORVA
    {
        UME::SIMD::SIMD8_16u vec0(0x7281);
        UME::SIMD::SIMD8_16u vec1(0x2314);
        vec0.xora(vec1);
        CHECK_CONDITION(vec0[0] == 0x5195, "SIMD8_16u::XORVA");
    }
    // NOT
    {
        UME::SIMD::SIMD8_16u vec0(0x7281);
        UME::SIMD::SIMD8_16u vec1(0x2314);
        vec0 = vec1.notv();
        CHECK_CONDITION(vec0[0] == 0xDCEB, "SIMD8_16u::NOT");
    }
    // MBLENDV
    {
        UME::SIMD::SIMD8_16u vec0(7);
        UME::SIMD::SIMD8_16u vec1(24);
        UME::SIMD::SIMDMask8 mask2(false, false, false, false, true, true, true, true);
        UME::SIMD::SIMD8_16u vec3 = vec0.blend(mask2, vec1);
        CHECK_CONDITION(vec3[0] == 24 && vec3[4] == 7, "SIMD8_16u::MBLENDV");
    }
    // HADD
    {
        UME::SIMD::SIMD8_16u vec0(3);
        uint16_t res = vec0.hadd();
        CHECK_CONDITION(res == 24, "SIMD8_16u::HADD");
    }
    // MAXV
    {
        UME::SIMD::SIMD8_16u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        UME::SIMD::SIMD8_16u vec1(8, 7, 6, 5, 4, 3, 2, 1);
        UME::SIMD::SIMD8_16u vec2;
        vec2 = vec0.max(vec1);
        CHECK_CONDITION(vec2[0] == 8 && vec2[6] == 7, "SIMD8_16u::MAXV");
    }
    // MINV
    {
        UME::SIMD::SIMD8_16u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        UME::SIMD::SIMD8_16u vec1(8, 7, 6, 5, 4, 3, 2, 1);
        UME::SIMD::SIMD8_16u vec2;
        vec2 = vec1.min(vec0);
        CHECK_CONDITION(vec2[0] == 1 && vec2[6] == 2, "SIMD8_16u::MINV");
    }
    // ABS
    {
        UME::SIMD::SIMD8_16i vec0(-1);
        UME::SIMD::SIMD8_16i vec1 = vec0.abs();

        CHECK_CONDITION(vec1[0] == vec1[7] == 1, "SIMD8_16i::ABS");
    }
    // ROLS
    {
        UME::SIMD::SIMD8_16i vec0(1, 2, 3, 4, 5, 6, 7, 8);
        UME::SIMD::SIMD8_16i vec1;
        vec1 = vec0.rol(3);
        CHECK_CONDITION(vec1[0] == 8 && vec1[7] == 56, "SIMD8_16i::ROLS");
    }

    return g_failCount;
}

int test_UME_SIMD4_32(bool supressMessages)
{
    int fail_u = test_UME_SIMD4_32u(supressMessages);
    int fail_i = test_UME_SIMD4_32i(supressMessages);
    int fail_f = test_UME_SIMD4_32f(supressMessages);


    return fail_u + fail_i + fail_f;
}

int test_UME_SIMD4_32u(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD4_32u test";
    
    INIT_TEST(header, supressMessages);

    // ZERO-CONSTR
    {
        UME::SIMD::SIMD4_32u vec1;
        CHECK_CONDITION(true, "SIMD4_32u()"); 
    }
    // VEC (SCALAR_TYPE i)
    {  
        UME::SIMD::SIMD4_32u vec1(8);
        CHECK_CONDITION(vec1[3] == 8,  "SIMD4_32u(uint32_t)"); 

    }
    // VEC (SCALAR_TYPE i0, ... SCALAR_TYPE i_VEC_LEN)
    {  
        UME::SIMD::SIMD4_32u vec1(8, 4, 2, 1);
        CHECK_CONDITION(vec1[0] == 8 && vec1[2] == 2,  "SIMD4_32u(uint32_t, ...)"); 
    }
    
    // VEC_UINT (VEC_INT)
    {
        UME::SIMD::SIMD4_32i vec0(8);
        UME::SIMD::SIMD4_32u vec1;

        vec1 = UME::SIMD::SIMD4_32u(vec0);
        CHECK_CONDITION(vec1[2] == 8, "SIMD4_32u(SIMD4_32i)");
    }
    // LENGTH
    {
        CHECK_CONDITION(UME::SIMD::SIMD4_32u::length() == 4, "SIMD4_32u::LENGTH");
    }
    // ALIGNMENT
    {
        CHECK_CONDITION(UME::SIMD::SIMD4_32u::alignment() == 16, "SIMD4_32u::ALIGNMENT");
    }
    // ASSIGNV
    {
        UME::SIMD::SIMD4_32u vec0(1, 2, 3, 4);
        UME::SIMD::SIMD4_32u vec1(15);

        vec1.assign(vec0);
        CHECK_CONDITION(vec1[0] == 1 && vec1[1] == 2, "SIMD4_32u::ASSIGNV");
    }      
    // LOADA
    {
        alignas(16) uint32_t arr[4] = { 1, 3, 8, 321};
        UME::SIMD::SIMD4_32u vec0(42);
        vec0.loada(arr);

        CHECK_CONDITION(vec0[0] == 1 && vec0[3] == 321, "SIMD4_32u::LOADA");
    }
    // STORE
    {
        uint32_t arr[4] = { 1, 3, 9, 124};
        UME::SIMD::SIMD4_32u vec0(9, 32, 28, 1256);
        vec0.store(arr);
        CHECK_CONDITION(arr[0] == 9 && arr[3] == 1256, "SIMD4_32u::STORE");
    }
    // STOREA
    {
        alignas(16) uint32_t arr[4] = { 1, 3, 9, 124};
        UME::SIMD::SIMD4_32u vec0(9, 32, 28, 1256);
        vec0.storea(arr);
        CHECK_CONDITION(arr[0] == 9 && arr[3] == 1256, "SIMD4_32u::STOREA");
    }
    // ADDV
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(3, 14, 28, 60);
        UME::SIMD::SIMD4_32u vec2;
        vec2 = vec0.add(vec1);
        CHECK_CONDITION(vec2[0] == 12 && vec2[1] == 22 && vec2[2] == 35 && vec2[3] == 66, "SIMD4_32u::ADDV");
    }
    // MADDV
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(3, 14, 28, 60);
        UME::SIMD::SIMD4_32u vec2;
        UME::SIMD::SIMDMask4 mask(true, false, true, true);
        vec2 = vec0.add(mask, vec1);
        CHECK_CONDITION(vec2[0] == 12 && vec2[1] == 8 && vec2[2] == 35 && vec2[3] == 66, "SIMD4_32u::MADDV");
    }
    // ADDS
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t b = 34;
        UME::SIMD::SIMD4_32u vec2;
        vec2 = vec0.add(b);
        CHECK_CONDITION(vec2[0] == 43 && vec2[1] == 42 && vec2[2] == 41 && vec2[3] == 40, "SIMD4_32u::ADDS");
    }
    // MADDS
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t b = 34;
        UME::SIMD::SIMD4_32u vec2;
        UME::SIMD::SIMDMask4 mask(true, false, true, true);
        vec2 = vec0.add(mask, b);
        CHECK_CONDITION(vec2[0] == 43 && vec2[1] == 8 && vec2[2] == 41 && vec2[3] == 40, "SIMD4_32u::MADDS");
    }
    // ADDVA
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(3, 14, 28, 60);
        vec0.adda(vec1);
        CHECK_CONDITION(vec0[0] == 12 && vec0[1] == 22 && vec0[2] == 35 && vec0[3] == 66, "SIMD4_32u::ADDVA");
    }
    // ADDSA
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 12;
        vec0.adda(val1);
        CHECK_CONDITION(vec0[0] == 21 && vec0[1] == 20 && vec0[2] == 19 && vec0[3] == 18, "SIMD4_32u::ADDSA");
    }
    // POSTINC
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(0);
        vec1 = vec0.postInc();
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == 9 && vec0[2] == 8 && vec0[3] == 7, "SIMD4_32u::POSTINC 1");
        CHECK_CONDITION(vec1[0] == 9 && vec1[1] == 8 && vec1[2] == 7 && vec1[3] == 6, "SIMD4_32u::POSTINC 2");
    }
    // PREFINC
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(0);
        vec1 = vec0.prefInc();
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == 9 && vec0[2] == 8 && vec0[3] == 7, "SIMD4_32u::PREFINC 1");
        CHECK_CONDITION(vec1[0] == 10 && vec1[1] == 9 && vec1[2] == 8 && vec1[3] == 7, "SIMD4_32u::PREFINC 2");
    }
    // SUBV
    {
        UME::SIMD::SIMD4_32u vec0(9, 14, 28, 60);
        UME::SIMD::SIMD4_32u vec1(3, 8,   7,  6);
        UME::SIMD::SIMD4_32u vec2;
        vec2 = vec0.sub(vec1);
        CHECK_CONDITION(vec2[0] == 6 && vec2[1] == 6 && vec2[2] == 21 && vec2[3] == 54, "SIMD4_32u::SUBV");
    }
    // SUBS
    {
        UME::SIMD::SIMD4_32u vec0(900, 8, 7, 6);
        uint32_t b = 34;
        UME::SIMD::SIMD4_32u vec2;
        vec2 = vec0.sub(b);
        CHECK_CONDITION(vec2[0] == 866 && vec2[1] == 0xFFFFFFE6 && vec2[2] == 0xFFFFFFE5 && vec2[3] == 0xFFFFFFE4, "SIMD4_32u::SUBS");
    }
    // SUBVA
    {
        UME::SIMD::SIMD4_32u vec0(9, 14, 28, 60);
        UME::SIMD::SIMD4_32u vec1(3, 8,   7,  6);
        vec0.suba(vec1);
        CHECK_CONDITION(vec0[0] == 6 && vec0[1] == 6 && vec0[2] == 21 && vec0[3] == 54, "SIMD4_32u::SUBVA");
    }
    // SUBSA
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 4;
        vec0.suba(val1);
        CHECK_CONDITION(vec0[0] == 5 && vec0[1] == 4 && vec0[2] == 3 && vec0[3] == 2, "SIMD4_32u::SUBSA");
    }
    // POSTDEC
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(0);
        vec1 = vec0.postDec();
        CHECK_CONDITION(vec0[0] == 8 && vec0[1] == 7 && vec0[2] == 6 && vec0[3] == 5, "SIMD4_32u::POSTDEC 1");
        CHECK_CONDITION(vec1[0] == 9 && vec1[1] == 8 && vec1[2] == 7 && vec1[3] == 6, "SIMD4_32u::POSTDEC 2");
    }
    // PREFDEC
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(0);
        vec1 = vec0.prefDec();
        CHECK_CONDITION(vec0[0] == 8 && vec0[1] == 7 && vec0[2] == 6 && vec0[3] == 5, "SIMD4_32u::PREFDEC 1");
        CHECK_CONDITION(vec1[0] == 8 && vec1[1] == 7 && vec1[2] == 6 && vec1[3] == 5, "SIMD4_32u::PREFDEC 2");
    }
    // MULV
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(0);
        UME::SIMD::SIMD4_32u vec2;
        vec2 = vec0.mul(vec1);
        CHECK_CONDITION(vec2[0] == 0 && vec2[1] == 0 && vec2[2] == 0 && vec2[3] == 0, "SIMD4_32u::MULV 1");
    }
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(3);
        UME::SIMD::SIMD4_32u vec2;
        vec2 = vec0.mul(vec1);
        CHECK_CONDITION(vec2[0] == 27 && vec2[1] == 24 && vec2[2] == 21 && vec2[3] == 18, "SIMD4_32u::MULV 2");
    }
    // MULS
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 0;
        UME::SIMD::SIMD4_32u vec2;
        vec2 = vec0.mul(val1);
        CHECK_CONDITION(vec2[0] == 0 && vec2[1] == 0 && vec2[2] == 0 && vec2[3] == 0, "SIMD4_32u::MULS 1");
    }
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        UME::SIMD::SIMD4_32u vec2;
        vec2 = vec0.mul(val1);
        CHECK_CONDITION(vec2[0] == 27 && vec2[1] == 24 && vec2[2] == 21 && vec2[3] == 18, "SIMD4_32u::MULS 2");
    }
    // MULV
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(3);
        vec0.mula(vec1);
        CHECK_CONDITION(vec0[0] == 27 && vec0[1] == 24 && vec0[2] == 21 && vec0[3] == 18, "SIMD4_32u::MULVA");
    }
    // MULSA
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        vec0.mula(val1);
        CHECK_CONDITION(vec0[0] == 27 && vec0[1] == 24 && vec0[2] == 21 && vec0[3] == 18, "SIMD4_32u::MULSA");
    }
    // DIVV
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(3);
        UME::SIMD::SIMD4_32u vec2;
        vec2 = vec0.div(vec1);
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == 2 && vec2[2] == 2 && vec2[3] == 2, "SIMD4_32u::DIVV");
    }
    // DIVS
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        UME::SIMD::SIMD4_32u vec2;
        vec2 = vec0.div(val1);
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == 2 && vec2[2] == 2 && vec2[3] == 2, "SIMD4_32u::DIVS");
    }
    // DIVVA
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(3);
        vec0.diva(vec1);
        CHECK_CONDITION(vec0[0] == 3 && vec0[1] == 2 && vec0[2] == 2 && vec0[3] == 2, "SIMD4_32u::DIVVA");
    }
    // DIVSA
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        vec0.diva(val1);
        CHECK_CONDITION(vec0[0] == 3 && vec0[1] == 2 && vec0[2] == 2 && vec0[3] == 2, "SIMD4_32u::DIVSA");
    }
    // RCPS
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 18;
        UME::SIMD::SIMD4_32u vec2;
        vec2 = vec0.rcp(val1);
        CHECK_CONDITION(vec2[0] == 2 && vec2[1] == 2 && vec2[2] == 2 && vec2[3] == 3, "SIMD4_32u::RCPS");
    }
    // GATHER
    {
        alignas(16) uint32_t arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        uint64_t indices[] = {1, 3, 8, 5};
        UME::SIMD::SIMD4_32u vec0(1);
        vec0.gather(arr, indices);
        CHECK_CONDITION(vec0[0] == 2 && vec0[1] == 4 && vec0[2] == 9 && vec0[3] == 6, "SIMD4_32u::GATHER");
    }
    // MGATHER
    {
        alignas(16) uint32_t arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        uint64_t indices[] = {1, 3, 8, 5};
        UME::SIMD::SIMD4_32u vec0(1);
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        vec0.gather(mask, arr, indices);
        CHECK_CONDITION(vec0[0] == 2 && vec0[1] == 1 && vec0[2] == 9 && vec0[3] == 1, "SIMD4_32u::MGATHER");
    }
    // GATHERV
    {
        alignas(16) uint32_t arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        UME::SIMD::SIMD4_32u indices(1, 3, 8, 5);
        UME::SIMD::SIMD4_32u vec0(1);
        vec0.gather(arr, indices);
        CHECK_CONDITION(vec0[0] == 2 && vec0[1] == 4 && vec0[2] == 9 && vec0[3] == 6, "SIMD4_32u::GATHERV");
    }
    // MGATHERV
    {
        alignas(16) uint32_t arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        UME::SIMD::SIMD4_32u indices(1, 3, 8, 5);
        UME::SIMD::SIMD4_32u vec0(1);
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        vec0.gather(mask, arr, indices);
        CHECK_CONDITION(vec0[0] == 2 && vec0[1] == 1 && vec0[2] == 9 && vec0[3] == 1, "SIMD4_32u::MGATHERV");
    }
    // SCATTER
    {
        alignas(16) uint32_t arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        uint64_t indices[] = {1, 3, 8, 5};
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        vec0.scatter(arr, indices);
        CHECK_CONDITION(arr[1] == 9 && arr[3] == 8 && arr[8] == 7 && arr[9] == 10, "SIMD4_32u::SCATTER");
    }
    // SCATTERV
    {
        alignas(16) uint32_t arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        UME::SIMD::SIMD4_32u indices(1, 3, 8, 5);
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        vec0.scatter(arr, indices);
        CHECK_CONDITION(arr[1] == 9 && arr[3] == 8 && arr[8] == 7 && arr[9] == 10, "SIMD4_32u::SCATTERV");
    }
    // MSCATTERV
    {
        alignas(16) uint32_t arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        UME::SIMD::SIMD4_32u indices(1, 3, 8, 5);
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        vec0.scatter(mask, arr, indices);
        CHECK_CONDITION(arr[1] == 9 && arr[3] == 4 && arr[8] == 7 && arr[9] == 10, "SIMD4_32u::MSCATTERV");
    }
    // LSHV
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(1, 2, 3, 4);
        UME::SIMD::SIMD4_32u vec2;
        vec2 = vec0.lsh(vec1);
        CHECK_CONDITION(vec2[0] == 18 && vec2[1] == 32 && vec2[2] == 56 && vec2[3] == 96, "SIMD4_32u::LSHV");
    }
    // MLSHV
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(1, 2, 3, 4);
        UME::SIMD::SIMD4_32u vec2;
        UME::SIMD::SIMDMask4       mask(true, false, false, true);
        vec2 = vec0.lsh(mask, vec1);
        CHECK_CONDITION(vec2[0] == 18 && vec2[1] == 8 && vec2[2] == 7 && vec2[3] == 96, "SIMD4_32u::MLSHV");
    }
    // LSHS
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        UME::SIMD::SIMD4_32u vec2;
        vec2 = vec0.lsh(val1);
        CHECK_CONDITION(vec2[0] == 72 && vec2[1] == 64 && vec2[2] == 56 && vec2[3] == 48, "SIMD4_32u::LSHS");
    }
    // MLSHS
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        UME::SIMD::SIMD4_32u vec2;
        UME::SIMD::SIMDMask4       mask(true, false, false, true);
        vec2 = vec0.lsh(mask, val1);
        CHECK_CONDITION(vec2[0] == 72 && vec2[1] == 8 && vec2[2] == 7 && vec2[3] == 48, "SIMD4_32u::MLSHS");
    }
    // LSHVA
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(1, 2, 3, 4);
        vec0.lsha(vec1);
        CHECK_CONDITION(vec0[0] == 18 && vec0[1] == 32 && vec0[2] == 56 && vec0[3] == 96, "SIMD4_32u::LSHVA");
    }
    // MLSHVA
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(1, 2, 3, 4);
        UME::SIMD::SIMDMask4       mask(true, false, true, false);
        vec0.lsha(mask, vec1);
        CHECK_CONDITION(vec0[0] == 18 && vec0[1] == 8 && vec0[2] == 56 && vec0[3] == 6, "SIMD4_32u::MLSHVA");
    }
    // LSHSA
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        vec0.lsha(val1);
        CHECK_CONDITION(vec0[0] == 72 && vec0[1] == 64 && vec0[2] == 56 && vec0[3] == 48, "SIMD4_32u::LSHSA");
    }
    // MLSHSA 
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMDMask4       mask(true, false, true, false);
        uint32_t val1 = 3;
        vec0.lsha(mask, val1);
        CHECK_CONDITION(vec0[0] == 72 && vec0[1] == 8 && vec0[2] == 56 && vec0[3] == 6, "SIMD4_32u::MLSHSA");
    }
    // RSHV
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(1, 2, 3, 4);
        UME::SIMD::SIMD4_32u vec2;
        vec2 = vec0.rsh(vec1);
        CHECK_CONDITION(vec2[0] == 4 && vec2[1] == 2 && vec2[2] == 0 && vec2[3] == 0, "SIMD4_32u::RSHV");
    }
    // RSHS
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        UME::SIMD::SIMD4_32u vec2;
        vec2 = vec0.rsh(val1);
        CHECK_CONDITION(vec2[0] == 1 && vec2[1] == 1 && vec2[2] == 0 && vec2[3] == 0, "SIMD4_32u::RSHS");
    }
    // RSHVA
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32u vec1(1, 2, 3, 4);
        vec0.rsha(vec1);
        CHECK_CONDITION(vec0[0] == 4 && vec0[1] == 2 && vec0[2] == 0 && vec0[3] == 0, "SIMD4_32u::RSHVA");
    }
    // RSHSA
    {
        UME::SIMD::SIMD4_32u vec0(9, 8, 7, 6);
        uint32_t val1 = 3;
        vec0.rsha(val1);
        CHECK_CONDITION(vec0[0] == 1 && vec0[1] == 1 && vec0[2] == 0 && vec0[3] == 0, "SIMD4_32u::RSHSA");
    }
    // ROLV
    {
        UME::SIMD::SIMD4_32u vec0(0x91111111);
        UME::SIMD::SIMD4_32u vec1(3, 5, 7, 23);
        UME::SIMD::SIMD4_32u vec2;

        vec2 = vec0.rol(vec1);
        CHECK_CONDITION(vec2[0] == 0x8888888C && vec2[1] == 0x22222232 && vec2[2] == 0x888888C8 && vec2[3] == 0x88C88888, "SIMD4_32u::ROLV");
    }
    // MROLV)
    {
        UME::SIMD::SIMD4_32u vec0(0x91111111);
        UME::SIMD::SIMD4_32u vec1(3, 5, 7, 23);
        UME::SIMD::SIMD4_32u vec2;
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        vec2 = vec0.rol(mask, vec1);
        CHECK_CONDITION(vec2[0] == 0x8888888C && vec2[1] == 0x22222232 && vec2[2] == 0x888888C8 && vec2[3] == 0x88C88888, "SIMD4_32u::MROLV");
    }
    // ROLS
    {
        UME::SIMD::SIMD4_32u vec0(0x80000001, 0x8F000001, 0x8F0000F1, 0xEF0000F1);
        uint32_t val1 = 5;
        UME::SIMD::SIMD4_32u vec2;
        vec2 = vec0.rol(val1);
        CHECK_CONDITION(vec2[0] == 0x00000030 && vec2[1] == 0xE0000031 && vec2[2] == 0xE0001E31 && vec2[3] == 0xE0001E3D, "SIMD4_32u::ROLS");
    }
    // MROLS
    {
        UME::SIMD::SIMD4_32u vec0(0x80000001, 0x8F000001, 0x8F0000F1, 0xEF0000F1);
        uint32_t val1 = 5;
        UME::SIMD::SIMD4_32u vec2;
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        vec2 = vec0.rol(mask, val1);
        CHECK_CONDITION(vec2[0] == 0x00000030 && vec2[1] == 0x8F000001 && vec2[2] == 0xE0001E31 && vec2[3] == 0xEF0000F1, "SIMD4_32u::MROLS");
    }
    // ROLVA
    {
        UME::SIMD::SIMD4_32u vec0(0x91111111);
        UME::SIMD::SIMD4_32u vec1(3, 5, 7, 23);
        vec0.rola(vec1);
        CHECK_CONDITION(vec0[0] == 0x8888888C && vec0[1] == 0x22222232 && vec0[2] == 0x888888C8 && vec0[3] == 0x88C88888, "SIMD4_32u::ROLVA");
    }
    // MROLVA
    {
        UME::SIMD::SIMD4_32u vec0(0x91111111);
        UME::SIMD::SIMD4_32u vec1(3, 5, 7, 23);
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        vec0.rola(mask, vec1);
        CHECK_CONDITION(vec0[0] == 0x8888888C && vec0[1] == 0x91111111 && vec0[2] == 0x888888C8 && vec0[3] == 0x91111111, "SIMD4_32u::MROLVA");
    }
    // ROLSA
    {
        UME::SIMD::SIMD4_32u vec0(0x80000001, 0x8F000001, 0x8F0000F1, 0xEF0000F1);
        uint32_t val1 = 5;
        vec0.rola(val1);
        CHECK_CONDITION(vec0[0] == 0x00000030 && vec0[1] == 0xE0000031 && vec0[2] == 0xE0001E31 && vec0[3] == 0xE0001E3D, "SIMD4_32u::ROLSA");
    }
    // CMPEQV
    {
        UME::SIMD::SIMD4_32u  vec0(1, 2, 3, 5);
        UME::SIMD::SIMD4_32u  vec1(1, 9, 0, 5);
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmpeq(vec1);
        CHECK_CONDITION(mask[0] == true && mask[1] == false && mask[2] == false && mask[3] == true, "SIMD4_32u::CMPEQV");
    }
    // CMPEQS
    {
        UME::SIMD::SIMD4_32u  vec0(1, 2, 3, 5);
        uint32_t val1 = 3;
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmpeq(val1);
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == true && mask[3] == false, "SIMD4_32u::CMPEQS");
    }
    // CMPNEV
    {
        UME::SIMD::SIMD4_32u  vec0(1, 2, 3, 5);
        UME::SIMD::SIMD4_32u  vec1(1, 9, 0, 5);
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmpne(vec1);
        CHECK_CONDITION(mask[0] == false && mask[1] == true && mask[2] == true && mask[3] == false, "SIMD4_32u::CMPNEV");
    }
    // CMPNES
    {
        UME::SIMD::SIMD4_32u  vec0(1, 2, 3, 5);
        uint32_t val1 = 3;
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmpne(val1);
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == false && mask[3] == true, "SIMD4_32u::CMPNES");
    }
    // CMPGTV
    {
        UME::SIMD::SIMD4_32u  vec0(1, 2, 3, 5);
        UME::SIMD::SIMD4_32u  vec1(1, 9, 0, 2);
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmpgt(vec1);
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == true && mask[3] == true, "SIMD4_32u::CMPGTV");
    }
    // CMPGTS
    {
        UME::SIMD::SIMD4_32u  vec0(1, 2, 3, 5);
        uint32_t val1 = 3;
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmpgt(val1);
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == false && mask[3] == true, "SIMD4_32u::CMPGTS");
    }
    // CMPLTV
    {
        UME::SIMD::SIMD4_32u  vec0(1, 2, 3, 5);
        UME::SIMD::SIMD4_32u  vec1(1, 9, 0, 2);
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmplt(vec1);
        CHECK_CONDITION(mask[0] == false && mask[1] == true && mask[2] == false && mask[3] == false, "SIMD4_32u::CMPLTV");
    }
    // CMPLTS
    {
        UME::SIMD::SIMD4_32u  vec0(1, 2, 3, 5);
        uint32_t val1 = 3;
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmplt(val1);
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == false && mask[3] == false, "SIMD4_32u::CMPLTS");
    }
    // CMPGEV
    {
        UME::SIMD::SIMD4_32u  vec0(1, 2, 3, 5);
        UME::SIMD::SIMD4_32u  vec1(1, 9, 3, 2);
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmpge(vec1);
        CHECK_CONDITION(mask[0] == true && mask[1] == false && mask[2] == true && mask[3] == true, "SIMD4_32u::CMPGEV");
    }
    // CMPGES
    {
        UME::SIMD::SIMD4_32u  vec0(1, 2, 3, 5);
        uint32_t val1 = 3;
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmpge(val1);
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == true && mask[3] == true, "SIMD4_32u::CMPGES");
    }
    // CMPLEV
    {
        UME::SIMD::SIMD4_32u  vec0(1, 2, 3, 5);
        UME::SIMD::SIMD4_32u  vec1(1, 9, 3, 2);
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmple(vec1);
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == true && mask[3] == false, "SIMD4_32u::CMPLEV");
    }
    // CMPLES
    {
        UME::SIMD::SIMD4_32u  vec0(1, 2, 3, 5);
        uint32_t val1 = 3;
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmple(val1);
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == true && mask[3] == false, "SIMD4_32u::CMPLES");
    }
    // ANDV
    {
        UME::SIMD::SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        UME::SIMD::SIMD4_32u vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);
        UME::SIMD::SIMD4_32u vec2;

        vec2 = vec0.andv(vec1);
        CHECK_CONDITION(
            vec2[0] == 0x01012000 && vec2[1] == 0x00000300 && vec2[2] == 0x09508060 && vec2[3] == 0x000F4020, 
            "SIMD4_32u::ANDV");
    }
    // ANDS
    {
        UME::SIMD::SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        uint32_t val1 = 0x571FC5A0;
        UME::SIMD::SIMD4_32u vec2;

        vec2 = vec0.ands(val1);
        CHECK_CONDITION(
            vec2[0] == 0x53130120 && vec2[1] == 0x500F0500 && vec2[2] == 0x0710C0A0 && vec2[3] == 0x000F4020, 
            "SIMD4_32u::ANDS");
    }
    // MANDS
    {
        UME::SIMD::SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        uint32_t val1 = 0x571FC5A0;
        UME::SIMD::SIMD4_32u vec2;
        UME::SIMD::SIMDMask4 mask(true, false, false, true);

        vec2 = vec0.ands(mask, val1);
        CHECK_CONDITION(
            vec2[0] == 0x53130120 && vec2[1] == 0xF00F0F10 && vec2[2] == 0x0FF0F0F0 && vec2[3] == 0x000F4020, 
            "SIMD4_32u::MANDS");
    }
    // ANDVA
    {
        UME::SIMD::SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        UME::SIMD::SIMD4_32u vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);

        vec0.anda(vec1);
        CHECK_CONDITION(
            vec0[0] == 0x01012000 && vec0[1] == 0x00000300 && vec0[2] == 0x09508060 && vec0[3] == 0x000F4020, 
            "SIMD4_32u::ANDVA");
    }
    // MANDVA
    {
        UME::SIMD::SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        UME::SIMD::SIMD4_32u vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);
        UME::SIMD::SIMDMask4 mask(true, false, false, true);

        vec0.anda(mask, vec1);
        CHECK_CONDITION(
            vec0[0] == 0x01012000 && vec0[1] == 0xF00F0F10 && vec0[2] == 0x0FF0F0F0 && vec0[3] == 0x000F4020, 
            "SIMD4_32u::MANDVA");
    }
    // MANDSA
    {
        UME::SIMD::SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        uint32_t val1 = 0x571FC5A0;
        UME::SIMD::SIMDMask4 mask(true, false, false, true);
                
        vec0.anda(mask, val1);
        CHECK_CONDITION(
            vec0[0] == 0x53130120 && vec0[1] == 0xF00F0F10 && vec0[2] == 0x0FF0F0F0 && vec0[3] == 0x000F4020, 
            "SIMD4_32u::MANDSA");
    }
    // NOT()
    {
        UME::SIMD::SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        UME::SIMD::SIMD4_32u vec1;
        vec1 = vec0.notv();
        CHECK_CONDITION(
            vec1[0] == 0x0CCCCCCB && vec1[1] == 0x0FF0F0EF && vec1[2] == 0xF00F0F0F && vec1[3] == 0xFFF0BDC0, 
            "SIMD4_32u::NOT");
    }

    // MNOT
    {
        UME::SIMD::SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        UME::SIMD::SIMD4_32u vec1;
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        vec1 = vec0.notv(mask);
        CHECK_CONDITION(
            vec1[0] == 0x0CCCCCCB && vec1[1] == 0xF00F0F10 && vec1[2] == 0xF00F0F0F && vec1[3] == 0x000F423F, 
            "SIMD4_32u::MNOT");
    }
    // MNOTA
    {
        UME::SIMD::SIMD4_32u vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        vec0.nota(mask);
        CHECK_CONDITION(
            vec0[0] == 0x0CCCCCCB && vec0[1] == 0xF00F0F10 && vec0[2] == 0xF00F0F0F && vec0[3] == 0x000F423F, 
            "SIMD4_32u::MNOTA");
    }
    // MBLENDV
    {
        UME::SIMD::SIMD4_32u vec0(3), vec1(5);
        UME::SIMD::SIMD4_32u vec2(2);
        UME::SIMD::SIMDMask4       mask(true, false, false, true);
        vec2 = vec0.blend(mask, vec1);
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == 5 && vec2[2] == 5 && vec2[3] == 3, "SIMD4_32u::MBLENDV");
    }
    // MBLENDS
    {
        UME::SIMD::SIMD4_32u vec0(3);
        uint32_t val1 = 5;
        UME::SIMD::SIMD4_32u vec2(2);
        UME::SIMD::SIMDMask4       mask(true, false, false, true);
    
        vec2 = vec0.blend(mask, val1);
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == 5 && vec2[2] == 5 && vec2[3] == 3, "SIMD4_32u::MBLENDS");
    }
    // HAND
    {
        UME::SIMD::SIMD4_32u vec0(0xF3333304, 0xFF0F3F00, 0x0FF0F000, 0x0F0F720F);
        uint32_t val1;
        val1 = vec0.hand();
        CHECK_CONDITION(val1 == 0x03003000, "SIMD4_32u::HAND");
    }
    // MHAND
    {
        UME::SIMD::SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        UME::SIMD::SIMDMask4 mask(true, false, false, true);
        uint32_t val1;
        val1 = vec0.hand(mask);
        CHECK_CONDITION(val1 == 0x00030204, "SIMD4_32u::MHAND");
    }
    // HANDS
    {
        UME::SIMD::SIMD4_32u vec0(0xF3333304, 0xFF0F3F00, 0x0FF0F000, 0x0F0F720F);
        uint32_t val1;
        uint32_t val2 = 0x03003000;
        val1 = vec0.hand(val2);
        CHECK_CONDITION(val1 == 0x03003000, "SIMD4_32u::HANDS");
    }
    // MHANDS
    {
        UME::SIMD::SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        UME::SIMD::SIMDMask4 mask(true, false, false, true);
        uint32_t val1;
        uint32_t val2 = 0x00010004;
        val1 = vec0.hand(mask, val2);
        CHECK_CONDITION(val1 == 0x00010004, "SIMD4_32u::MHANDS");
    }
    // HOR
    {
        UME::SIMD::SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        uint32_t val1;
        val1 = vec0.hor();
        CHECK_CONDITION(val1 == 0xFFFFFF0F, "SIMD4_32u::HOR");
    }
    // MHOR
    {
        UME::SIMD::SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        UME::SIMD::SIMDMask4 mask(false, false, true, true);
        uint32_t val1;
        val1 = vec0.hor(mask);
        CHECK_CONDITION(val1 == 0x0FFFF20F, "SIMD4_32u::MHOR");
    }
    // HORS
    {
        UME::SIMD::SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        uint32_t val1;
        uint32_t val2 = 0x00000030;
        val1 = vec0.hor(val2);
        CHECK_CONDITION(val1 == 0xFFFFFF3F, "SIMD4_32u::HORS");
    }
    // MHORS
    {
        UME::SIMD::SIMD4_32u vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        UME::SIMD::SIMDMask4 mask(false, false, true, true);
        uint32_t val1;
        uint32_t val2 = 0x00000030;
        val1 = vec0.hor(mask, val2);
        CHECK_CONDITION(val1 == 0x0FFFF23F, "SIMD4_32u::MHORS");
    }

    // HMUL
    {
        UME::SIMD::SIMD4_32u vec0(1, 2, 3, 4);
        uint32_t val1 = 0;
        val1 = vec0.hmul();
        CHECK_CONDITION(val1 == 24, "SIMD4_32u::HMUL");
    }
    // MHMUL
    {
        UME::SIMD::SIMD4_32u vec0(1, 2, 3, 4);
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        uint32_t val1 = 0;
        val1 = vec0.hmul(mask);
        CHECK_CONDITION(val1 == 3, "SIMD4_32u::MHMUL");
    }
    // HMULS
    {
        UME::SIMD::SIMD4_32u vec0(1, 2, 3, 4);
        uint32_t val1 = 42;
        uint32_t res = 0;
        res = vec0.hmul(val1);
        CHECK_CONDITION(res == 1008, "SIMD4_32u::HMULS");
    }

    return g_failCount;
}

int test_UME_SIMD4_32i(bool supressMessages)
{   
    char header[] = "UME::SIMD::SIMD4_32i test";
    
    INIT_TEST(header, supressMessages);

    // VEC()
    {
        UME::SIMD::SIMD4_32i vec0;
        CHECK_CONDITION(true, "SIMD4_32u()"); 
    }
    // VEC(SCALAR_TYPE i)
    {  
        UME::SIMD::SIMD4_32i vec0(-3);
        CHECK_CONDITION(vec0[1] == -3, "SIMD4_32i(int32_t)"); 

    }
    // VEC(SCALAR_TYPE i0, ... SCALAR_TYPE i_VEC_LEN)
    {  
        UME::SIMD::SIMD4_32i vec0(-3, -2, -1, 6);
        CHECK_CONDITION(vec0[0] == -3 && vec0[3] == 6, "SIMD4_32i(int32_t, ...)"); 
    }
    // VEC_UINT(VEC_INT)
    {
        UME::SIMD::SIMD4_32u vec0(8);
        UME::SIMD::SIMD4_32i vec1;

        vec1 = UME::SIMD::SIMD4_32i(vec0);
        CHECK_CONDITION(vec1[2] == 8, "SIMD4_32i(SIMD4_32u)");
    }
    // LENGTH
    {
        CHECK_CONDITION(UME::SIMD::SIMD4_32i::length() == 4, "SIMD4_32i::LENGTH");
    }
    // ALIGNEMENT
    {
        CHECK_CONDITION(UME::SIMD::SIMD4_32i::alignment() == 16, "SIMD4_32i::ALIGNEMENT");
    }
    // operator= (VEC)
    {
        UME::SIMD::SIMD4_32i vec0(-1, 2, 3, 4);
        UME::SIMD::SIMD4_32i vec1(15);

        vec1 = vec0;
        CHECK_CONDITION(vec1[0] == -1 && vec1[1] == 2, "SIMD4_32i::operator=");
    }    
    // LOAD
    {
        int32_t arr[4] = { 1, 3, 8, -41231};
        UME::SIMD::SIMD4_32i vec0(-3);
        vec0.load(arr);
        CHECK_CONDITION(vec0[0] == 1 && vec0[3] == -41231, "SIMD4_32i::LOAD");
    }
    // LOADA
    {
        alignas(16) int32_t arr[4] = { 1, 3, 8, -41231};
        UME::SIMD::SIMD4_32i vec0(-3);
        vec0.loada(arr);
        CHECK_CONDITION(vec0[0] == 1 && vec0[3] == -41231, "SIMD4_32i::LOADA");
    }
    // STORE
    {
        int32_t arr[4] = { 1, 3, 9, -124};
        UME::SIMD::SIMD4_32i vec0(9, 32, -28, -1256);
        vec0.store(arr);
        CHECK_CONDITION(arr[0] == 9 && arr[3] == -1256, "SIMD4_32i::STORE");
    }
    // STOREA
    {
        alignas(16) int32_t arr[4] = { 1, 3, 9, -124};
        UME::SIMD::SIMD4_32i vec0(9, 32, -28, -1256);
        vec0.storea(arr);
        CHECK_CONDITION(arr[0] == 9 && arr[3] == -1256, "SIMD4_32i::STOREA");
    }
    // GATHERS
    {
        alignas(16) int32_t arr[10] = {1, -2, 3, 4, 5, -6, 7, 8, 9, 10};
        uint64_t indices[] = {1, 3, 8, 5};
        UME::SIMD::SIMD4_32i vec0(1);
        vec0.gather(arr, indices);
        CHECK_CONDITION(vec0[0] == -2 && vec0[1] == 4 && vec0[2] == 9 && vec0[3] == -6, "SIMD4_32i::GATHERS");
    }
    // MGATHERS
    {
        alignas(16) int32_t arr[10] = {1, -2, 3, 4, 5, -6, 7, 8, 9, 10};
        uint64_t indices[] = {1, 3, 8, 5};
        UME::SIMD::SIMD4_32i vec0(1);
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        vec0.gather(mask, arr, indices);
        CHECK_CONDITION(vec0[0] == -2 && vec0[1] == 1 && vec0[2] == 9 && vec0[3] == 1, "SIMD4_32i::MGATHERS");
    }
    // GATHERV
    {
        alignas(16) int32_t arr[10] = {1, -2, 3, 4, 5, -6, 7, 8, 9, 10};
        UME::SIMD::SIMD4_32u indices(1, 3, 8, 5);
        UME::SIMD::SIMD4_32i vec0(1);
        vec0.gather(arr, indices);
        CHECK_CONDITION(vec0[0] == -2 && vec0[1] == 4 && vec0[2] == 9 && vec0[3] == -6, "SIMD4_32i::GATHERV");
    }
    // MGATHERV
    {
        alignas(16) int32_t arr[10] = {1, -2, 3, 4, 5, -6, 7, 8, 9, 10};
        UME::SIMD::SIMD4_32u indices(1, 3, 8, 5);
        UME::SIMD::SIMD4_32i vec0(1);
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        vec0.gather(mask, arr, indices);
        CHECK_CONDITION(vec0[0] == -2 && vec0[1] == 1 && vec0[2] == 9 && vec0[3] == 1, "SIMD4_32i::MGATHERV");
    }
    // SCATTERS
    {
        alignas(16) int32_t arr[10] = {1, -2, 3, 4, 5, -6, 7, 8, 9, 10};
        uint64_t indices[] = {1, 3, 8, 5};
        UME::SIMD::SIMD4_32i vec0(9, -8, 7, 6);
        vec0.scatter(arr, indices);
        CHECK_CONDITION(arr[1] == 9 && arr[3] == -8 && arr[8] == 7 && arr[9] == 10, "SIMD4_32i::SCATTERS");
    }
    // SCATTERV
    {
        alignas(16) int32_t arr[10] = {1, -2, 3, 4, 5, -6, 7, 8, 9, 10};
        UME::SIMD::SIMD4_32u indices(1, 3, 8, 5);
        UME::SIMD::SIMD4_32i vec0(9, -8, 7, 6);
        vec0.scatter(arr, indices);
        CHECK_CONDITION(arr[1] == 9 && arr[3] == -8 && arr[8] == 7 && arr[9] == 10, "SIMD4_32i::SCATTERV");
    }
    // MSCATTERV
    {
        alignas(16) int32_t arr[10] = {1, -2, 3, 4, 5, -6, 7, 8, 9, 10};
        UME::SIMD::SIMD4_32u indices(1, 3, 8, 5);
        UME::SIMD::SIMD4_32i vec0(9, -8, 7, 6);
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        vec0.scatter(mask, arr, indices);
        CHECK_CONDITION(arr[1] == 9 && arr[3] == 4 && arr[8] == 7 && arr[9] == 10, "SIMD4_32i::MSCATTERV");
    }
    // ADDV
    {
        UME::SIMD::SIMD4_32i vec0(9, -8, 7, 6);
        UME::SIMD::SIMD4_32i vec1(3, 14, 28, -60);
        UME::SIMD::SIMD4_32i vec2;
        vec2 = vec0.add(vec1);
        CHECK_CONDITION(vec2[0] == 12 && vec2[1] == 6 && vec2[2] == 35 && vec2[3] == -54, "SIMD4_32i::ADDV");
    }
    // MADDV
    {
        UME::SIMD::SIMD4_32i vec0(9, -8, 7, 6);
        UME::SIMD::SIMD4_32i vec1(3, 14, 28, -60);
        UME::SIMD::SIMD4_32i vec2;
        UME::SIMD::SIMDMask4 mask(true, false, true, true);
        vec2 = vec0.add(mask, vec1);
        CHECK_CONDITION(vec2[0] == 12 && vec2[1] == -8 && vec2[2] == 35 && vec2[3] == -54, "SIMD4_32i::MADDV");
    }
    // ADDS
    {
        UME::SIMD::SIMD4_32i vec0(9, -8, 7, 6);
        int32_t b = 34;
        UME::SIMD::SIMD4_32i vec2;
        vec2 = vec0.add(b);
        CHECK_CONDITION(vec2[0] == 43 && vec2[1] == 26 && vec2[2] == 41 && vec2[3] == 40, "SIMD4_32i::ADDS");
    }
    // MADDS
    {
        UME::SIMD::SIMD4_32i vec0(9, -8, 7, 6);
        int32_t b = 34;
        UME::SIMD::SIMD4_32i vec2;
        UME::SIMD::SIMDMask4 mask(true, false, true, true);
        vec2 = vec0.add(mask, b);
        CHECK_CONDITION(vec2[0] == 43 && vec2[1] == -8 && vec2[2] == 41 && vec2[3] == 40, "SIMD4_32i::MADDS");
    }
    // ADDVA
    {
        UME::SIMD::SIMD4_32i vec0(9, -8, 7, 6);
        UME::SIMD::SIMD4_32i vec1(3, 14, 28, -60);
        vec0.adda(vec1);
        CHECK_CONDITION(vec0[0] == 12 && vec0[1] == 6 && vec0[2] == 35 && vec0[3] == -54, "SIMD4_32i::ADDVA");
    }
    // ADDSA
    {
        UME::SIMD::SIMD4_32i vec0(9, -8, 7, 6);
        int32_t val1 = 12;
        vec0.adda(val1);
        CHECK_CONDITION(vec0[0] == 21 && vec0[1] == 4 && vec0[2] == 19 && vec0[3] == 18, "SIMD4_32i::ADDSA");
    }
    // POSTINC
    {
        UME::SIMD::SIMD4_32i vec0(9, -8, 7, 6);
        UME::SIMD::SIMD4_32i vec1(0);
        vec1 = vec0.postInc();
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == -7 && vec0[2] == 8 && vec0[3] == 7, "SIMD4_32i::POSTINC 1");
        CHECK_CONDITION(vec1[0] == 9 && vec1[1] == -8 && vec1[2] == 7 && vec1[3] == 6, "SIMD4_32i::POSTINC 2");
    }
    // PREFINC
    {
        UME::SIMD::SIMD4_32i vec0(9, -8, 7, 6);
        UME::SIMD::SIMD4_32i vec1(0);
        vec1 = vec0.prefInc();
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == -7 && vec0[2] == 8 && vec0[3] == 7, "SIMD4_32i::PREFINC 1");
        CHECK_CONDITION(vec1[0] == 10 && vec1[1] == -7 && vec1[2] == 8 && vec1[3] == 7, "SIMD4_32i::PREFINC 2");
    }
    // SUBV
    {
        UME::SIMD::SIMD4_32i vec0(9, -8, 7, 6);
        UME::SIMD::SIMD4_32i vec1(3, 14, 28, -60);
        UME::SIMD::SIMD4_32i vec2;
        vec2 = vec0.sub(vec1);
        CHECK_CONDITION(vec2[0] == 6 && vec2[1] == -22 && vec2[2] == -21 && vec2[3] == 66, "SIMD4_32i::SUBV");
    }
    // SUBS
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, 7, 6);
        int32_t b = 34;
        UME::SIMD::SIMD4_32i vec2;
        vec2 = vec0.sub(b);
        CHECK_CONDITION(vec2[0] == -25 && vec2[1] == -26 && vec2[2] == -27 && vec2[3] == -28, "SIMD4_32i::SUBS");
    }
    // SUBVA
    {
        UME::SIMD::SIMD4_32i vec0(9, -8, 7, 6);
        UME::SIMD::SIMD4_32i vec1(3, 14, 28, -60);
        vec0.suba(vec1);
        CHECK_CONDITION(vec0[0] == 6 && vec0[1] == -22 && vec0[2] == -21 && vec0[3] == 66, "SIMD4_32i::SUBVA");
    }
    // SUBSA
    {
        UME::SIMD::SIMD4_32i vec0(9, -8, 7, 6);
        int32_t val1 = 12;
        vec0.suba(val1);
        CHECK_CONDITION(vec0[0] == -3 && vec0[1] == -20 && vec0[2] == -5 && vec0[3] == -6, "SIMD4_32i::SUBSA");
    }
    // POSTDEC
    {
        UME::SIMD::SIMD4_32i vec0(9, -8, 7, 6);
        UME::SIMD::SIMD4_32i vec1(0);
        vec1 = vec0.postDec();
        CHECK_CONDITION(vec0[0] == 8 && vec0[1] == -9 && vec0[2] == 6 && vec0[3] == 5, "SIMD4_32i::POSTDEC 1");
        CHECK_CONDITION(vec1[0] == 9 && vec1[1] == -8 && vec1[2] == 7 && vec1[3] == 6, "SIMD4_32i::POSTDEC 2");
    }
    // PREFDEC
    {
        UME::SIMD::SIMD4_32i vec0(9, -8, 7, 6);
        UME::SIMD::SIMD4_32i vec1(0);
        vec1 = vec0.prefDec();
        CHECK_CONDITION(vec0[0] == 8 && vec0[1] == -9 && vec0[2] == 6 && vec0[3] == 5, "SIMD4_32i::PREFDEC 1");
        CHECK_CONDITION(vec1[0] == 8 && vec1[1] == -9 && vec1[2] == 6 && vec1[3] == 5, "SIMD4_32i::PREFDEC 2");
    }
    // MULV
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32i vec1(0);
        UME::SIMD::SIMD4_32i vec2;
        vec2 = vec0.mul(vec1);
        CHECK_CONDITION(vec2[0] == 0 && vec2[1] == 0 && vec2[2] == 0 && vec2[3] == 0, "SIMD4_32i::MULV 1");
    }
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32i vec1(-3);
        UME::SIMD::SIMD4_32i vec2;
        vec2 = vec0.mul(vec1);
        CHECK_CONDITION(vec2[0] == -27 && vec2[1] == -24 && vec2[2] == -21 && vec2[3] == -18, "SIMD4_32i::MULV 2");
    }
    // MULS
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, 7, 6);
        int32_t val1 = 0;
        UME::SIMD::SIMD4_32i vec2;
        vec2 = vec0.mul(val1);
        CHECK_CONDITION(vec2[0] == 0 && vec2[1] == 0 && vec2[2] == 0 && vec2[3] == 0, "SIMD4_32i::MULS 1");
    }
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, 7, 6);
        int32_t val1 = -3;
        UME::SIMD::SIMD4_32i vec2;
        vec2 = vec0.mul(val1);
        CHECK_CONDITION(vec2[0] == -27 && vec2[1] == -24 && vec2[2] == -21 && vec2[3] == -18, "SIMD4_32i::MULS 2");
    }
    // MULVA
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, 7, 6);
        UME::SIMD::SIMD4_32i vec1(-3);
        vec0.mula(vec1);
        CHECK_CONDITION(vec0[0] == -27 && vec0[1] == -24 && vec0[2] == -21 && vec0[3] == -18, "SIMD4_32i::MULVA");
    }
    // MULSA
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, 7, 6);
        int32_t val1 = -3;
        vec0.mula(val1);
        CHECK_CONDITION(vec0[0] == -27 && vec0[1] == -24 && vec0[2] == -21 && vec0[3] == -18, "SIMD4_32i::MULSA");
    }
    // DIVV
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, -7, 6);
        UME::SIMD::SIMD4_32i vec1(3);
        UME::SIMD::SIMD4_32i vec2;
        vec2 = vec0.div(vec1);
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == 2 && vec2[2] == -2 && vec2[3] == 2, "SIMD4_32i::DIVV");
    }
    // DIVS
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, -7, 6);
        int32_t val1 = -3;
        UME::SIMD::SIMD4_32i vec2;
        vec2 = vec0.div(val1);
        CHECK_CONDITION(vec2[0] == -3 && vec2[1] == -2 && vec2[2] == 2 && vec2[3] == -2, "SIMD4_32i::DIVS");
    }
    // DIVVA
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, -7, 6);
        UME::SIMD::SIMD4_32i vec1(3);
        vec0.diva(vec1);
        CHECK_CONDITION(vec0[0] == 3 && vec0[1] == 2 && vec0[2] == -2 && vec0[3] == 2, "SIMD4_32i::DIVVA");
    }
    // DIVSA
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, -7, 6);
        int32_t val1 = 3;
        vec0.diva(val1);
        CHECK_CONDITION(vec0[0] == 3 && vec0[1] == 2 && vec0[2] == -2 && vec0[3] == 2, "SIMD4_32i::DIVSA");
    }
    // RCPS
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, -7, 6);
        int32_t val1 = -18;
        UME::SIMD::SIMD4_32i vec2;
        vec2 = vec0.rcp(val1);
        CHECK_CONDITION(vec2[0] == -2 && vec2[1] == -2 && vec2[2] == 2 && vec2[3] == -3, "SIMD4_32i::RCPS");
    }
    // LSHV
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, -7, 6);
        UME::SIMD::SIMD4_32u vec1(1, 2, 3, 4);
        UME::SIMD::SIMD4_32i vec2;
        vec2 = vec0.lsh(vec1);
        CHECK_CONDITION(vec2[0] == 18 && vec2[1] == 32 && vec2[2] == -56 && vec2[3] == 96, "SIMD4_32i::LSHV");
    }
    // MLSHV
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, -7, 6);
        UME::SIMD::SIMD4_32u vec1(1, 2, 3, 4);
        UME::SIMD::SIMD4_32i vec2;
        UME::SIMD::SIMDMask4       mask(true, false, false, true);
        vec2 = vec0.lsh(mask, vec1);
        CHECK_CONDITION(vec2[0] == 18 && vec2[1] == 8 && vec2[2] == -7 && vec2[3] == 96, "SIMD4_32i::MLSHV");
    }
    // LSHS
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, -7, 6);
        uint32_t val1 = 3;
        UME::SIMD::SIMD4_32i vec2;
        vec2 = vec0.lsh(val1);
        CHECK_CONDITION(vec2[0] == 72 && vec2[1] == 64 && vec2[2] == -56 && vec2[3] == 48, "SIMD4_32i::LSHS");
    }
    // MLSHS
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, -7, 6);
        uint32_t val1 = 3;
        UME::SIMD::SIMD4_32i vec2;
        UME::SIMD::SIMDMask4       mask(true, false, false, true);
        vec2 = vec0.lsh(mask, val1);
        CHECK_CONDITION(vec2[0] == 72 && vec2[1] == 8 && vec2[2] == -7 && vec2[3] == 48, "SIMD4_32i::MLSHS");
    }
    // LSHVA
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, -7, 6);
        UME::SIMD::SIMD4_32u vec1(1, 2, 3, 4);
        vec0.lsha(vec1);
        CHECK_CONDITION(vec0[0] == 18 && vec0[1] == 32 && vec0[2] == -56 && vec0[3] == 96, "SIMD4_32i::LSHVA");
    }
    // MLSHVA
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, -7, 6);
        UME::SIMD::SIMD4_32u vec1(1, 2, 3, 4);
        UME::SIMD::SIMDMask4       mask(true, false, true, false);
        vec0.lsha(mask, vec1);
        CHECK_CONDITION(vec0[0] == 18 && vec0[1] == 8 && vec0[2] == -56 && vec0[3] == 6, "SIMD4_32i::MLSHVA");
    }
    // LSHSA
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, -7, 6);
        uint32_t val1 = 3;
        vec0.lsha(val1);
        CHECK_CONDITION(vec0[0] == 72 && vec0[1] == 64 && vec0[2] == -56 && vec0[3] == 48, "SIMD4_32i::LSHSA");
    }
    // MLSHSA
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, -7, 6);
        UME::SIMD::SIMDMask4       mask(true, false, true, false);
        uint32_t val1 = 3;
        vec0.lsha(mask, val1);
        CHECK_CONDITION(vec0[0] == 72 && vec0[1] == 8 && vec0[2] == -56 && vec0[3] == 6, "SIMD4_32i::MLSHSA");
    }
    // RSHV
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, -7, 6);
        UME::SIMD::SIMD4_32u vec1(1, 2, 3, 4);
        UME::SIMD::SIMD4_32i vec2;
        vec2 = vec0.rsh(vec1);
        CHECK_CONDITION(vec2[0] == 4 && vec2[1] == 2 && vec2[2] == -1 && vec2[3] == 0, "SIMD4_32i::RSHV");
    }
    // RSHS
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, -7, 6);
        uint32_t val1 = 3;
        UME::SIMD::SIMD4_32i vec2;
        vec2 = vec0.rsh(val1);
        CHECK_CONDITION(vec2[0] == 1 && vec2[1] == 1 && vec2[2] == -1 && vec2[3] == 0, "SIMD4_32i::RSHS");
    }
    // RSHVA
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, -7, 6);
        UME::SIMD::SIMD4_32u vec1(1, 2, 3, 4);
        vec0.rsha(vec1);
        CHECK_CONDITION(vec0[0] == 4 && vec0[1] == 2 && vec0[2] == -1 && vec0[3] == 0, "SIMD4_32i::RSHVA");
    }
    // RSHSA
    {
        UME::SIMD::SIMD4_32i vec0(9, 8, -7, 6);
        uint32_t val1 = 3;
        vec0.rsha(val1);
        CHECK_CONDITION(vec0[0] == 1 && vec0[1] == 1 && vec0[2] == -1 && vec0[3] == 0, "SIMD4_32i::RSHSA");
    }
    // MROLV
    {
        UME::SIMD::SIMD4_32i vec0(0x91111111);
        UME::SIMD::SIMD4_32u vec1(3, 5, 7, 23);
        UME::SIMD::SIMD4_32i vec2;
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        vec2 = vec0.rol(mask, vec1);
        CHECK_CONDITION(vec2[0] == 0x8888888C && vec2[1] == 0x91111111 && vec2[2] == 0x888888C8 && vec2[3] == 0x91111111, "SIMD4_32i::MROLV");
    }
    // ROLS
    {
        UME::SIMD::SIMD4_32i vec0(0x80000001, 0x8F000001, 0x8F0000F1, 0xEF0000F1);
        uint32_t val1 = 5;
        UME::SIMD::SIMD4_32i vec2;
        vec2 = vec0.rol(val1);
        CHECK_CONDITION(vec2[0] == 0x00000030 && vec2[1] == 0xE0000031 && vec2[2] == 0xE0001E31 && vec2[3] == 0xE0001E3D, "SIMD4_32i::ROLS");
    }
    // MROLS
    {
        UME::SIMD::SIMD4_32i vec0(0x80000001, 0x8F000001, 0x8F0000F1, 0xEF0000F1);
        uint32_t val1 = 5;
        UME::SIMD::SIMD4_32i vec2;
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        vec2 = vec0.rol(mask, val1);
        CHECK_CONDITION(vec2[0] == 0x00000030 && vec2[1] == 0x8F000001 && vec2[2] == 0xE0001E31 && vec2[3] == 0xEF0000F1, "SIMD4_32i::MROLS");
    }
    // ROLVA
    {
        UME::SIMD::SIMD4_32i vec0(0x91111111);
        UME::SIMD::SIMD4_32u vec1(3, 5, 7, 23);
        vec0.rola(vec1);
        CHECK_CONDITION(vec0[0] == 0x8888888C && vec0[1] == 0x22222232 && vec0[2] == 0x888888C8 && vec0[3] == 0x88C88888, "SIMD4_32i::ROLVA");
    }
    // MROLVA
    {
        UME::SIMD::SIMD4_32i vec0(0x91111111);
        UME::SIMD::SIMD4_32u vec1(3, 5, 7, 23);
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        vec0.rola(mask, vec1);
        CHECK_CONDITION(vec0[0] == 0x8888888C && vec0[1] == 0x91111111 && vec0[2] == 0x888888C8 && vec0[3] == 0x91111111, "SIMD4_32i::MROLVA");
    }
    // ROLSA
    {
        UME::SIMD::SIMD4_32i vec0(0x80000001, 0x8F000001, 0x8F0000F1, 0xEF0000F1);
        uint32_t val1 = 5;
        vec0.rola(val1);
        CHECK_CONDITION(vec0[0] == 0x00000030 && vec0[1] == 0xE0000031 && vec0[2] == 0xE0001E31 && vec0[3] == 0xE0001E3D, "SIMD4_32i::ROLSA");
    }
    
    // CMPEQV
    {
        UME::SIMD::SIMD4_32i  vec0(1, 2, -3, -5);
        UME::SIMD::SIMD4_32i  vec1(1, 9, 0, -5);
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmpeq(vec1);
        CHECK_CONDITION(mask[0] == true && mask[1] == false && mask[2] == false && mask[3] == true, "SIMD4_32i::CMPEQV");
    }
    // CMPEQS
    {
        UME::SIMD::SIMD4_32i  vec0(1, 2, -3, -5);
        int32_t val1 = -3;
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmpeq(val1);
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == true && mask[3] == false, "SIMD4_32i::CMPEQS");
    }
    // CMPNEV
    {
        UME::SIMD::SIMD4_32i  vec0(1, 2, -3, -5);
        UME::SIMD::SIMD4_32i  vec1(1, 9, 0, -5);
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmpne(vec1);
        CHECK_CONDITION(mask[0] == false && mask[1] == true && mask[2] == true && mask[3] == false, "SIMD4_32i::CMPNEV");
    }
    // CMPNES
    {
        UME::SIMD::SIMD4_32i  vec0(1, 2, -3, -5);
        int32_t val1 = -3;
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmpne(val1);
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == false && mask[3] == true, "SIMD4_32i::CMPNES");
    }
    // CMPGTV
    {
        UME::SIMD::SIMD4_32i  vec0(1, 2, 3, -5);
        UME::SIMD::SIMD4_32i  vec1(1, 9, 0, -2);
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmpgt(vec1);
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == true && mask[3] == false, "SIMD4_32i::CMPGTV");
    }
    // CMPGTS
    {
        UME::SIMD::SIMD4_32i  vec0(1, 2, -3, -5);
        int32_t val1 = -3;
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmpgt(val1);
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == false && mask[3] == false, "SIMD4_32i::CMPGTS");
    }
    // CMPLTV
    {
        UME::SIMD::SIMD4_32i  vec0(1, 2, 3, -5);
        UME::SIMD::SIMD4_32i  vec1(1, 9, 0, -2);
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmplt(vec1);
        CHECK_CONDITION(mask[0] == false && mask[1] == true && mask[2] == false && mask[3] == true, "SIMD4_32i::CMPLTV");
    }
    // CMPLTS
    {
        UME::SIMD::SIMD4_32i  vec0(1, 2, -3, -5);
        int32_t val1 = -3;
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmplt(val1);
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == false && mask[3] == true, "SIMD4_32i::CMPLTS");
    }
    // CMPGEV
    {
        UME::SIMD::SIMD4_32i  vec0(1, 2, -3, -5);
        UME::SIMD::SIMD4_32i  vec1(1, 9, -3, -2);
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmpge(vec1);
        CHECK_CONDITION(mask[0] == true && mask[1] == false && mask[2] == true && mask[3] == false, "SIMD4_32i::CMPGEV");
    }
    // CMPGES
    {
        UME::SIMD::SIMD4_32i  vec0(1, 2, -3, -5);
        int32_t val1 = -3;
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmpge(val1);
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == true && mask[3] == false, "SIMD4_32i::CMPGES");
    }
    // CMPLEV
    {
        UME::SIMD::SIMD4_32i  vec0(1, 2, -3, -5);
        UME::SIMD::SIMD4_32i  vec1(1, 9, -3, -7);
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmple(vec1);
        CHECK_CONDITION(mask[0] == true && mask[1] == true && mask[2] == true && mask[3] == false, "SIMD4_32i::CMPLEV");
    }
    // CMPLES
    {
        UME::SIMD::SIMD4_32i  vec0(1, 2, -3, -5);
        int32_t val1 = -3;
        UME::SIMD::SIMDMask4 mask;
        mask = vec0.cmple(val1);
        CHECK_CONDITION(mask[0] == false && mask[1] == false && mask[2] == true && mask[3] == true, "SIMD4_32i::CMPLES");
    }
    // ANDV
    {
        UME::SIMD::SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        UME::SIMD::SIMD4_32i vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);
        UME::SIMD::SIMD4_32i vec2;

        vec2 = vec0.andv(vec1);
        CHECK_CONDITION(
            vec2[0] == 0x01012000 && vec2[1] == 0x00000300 && vec2[2] == 0x09508060 && vec2[3] == 0x000F4020, 
            "SIMD4_32i::ANDV");
    }
    // ANDS
    {
        UME::SIMD::SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        int32_t val1 = 0x571FC5A0;
        UME::SIMD::SIMD4_32i vec2;

        vec2 = vec0.ands(val1);
        CHECK_CONDITION(
            vec2[0] == 0x53130120 && vec2[1] == 0x500F0500 && vec2[2] == 0x0710C0A0 && vec2[3] == 0x000F4020, 
            "SIMD4_32i::ANDS");
    }
    // MANDS
    {
        UME::SIMD::SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        int32_t val1 = 0x571FC5A0;
        UME::SIMD::SIMD4_32i vec2;
        UME::SIMD::SIMDMask4 mask(true, false, false, true);

        vec2 = vec0.ands(mask, val1);
        CHECK_CONDITION(
            vec2[0] == 0x53130120 && vec2[1] == 0xF00F0F10 && vec2[2] == 0x0FF0F0F0 && vec2[3] == 0x000F4020, 
            "SIMD4_32i::MANDS");
    }
    // ANDVA
    {
        UME::SIMD::SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        UME::SIMD::SIMD4_32i vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);

        vec0.anda(vec1);
        CHECK_CONDITION(
            vec0[0] == 0x01012000 && vec0[1] == 0x00000300 && vec0[2] == 0x09508060 && vec0[3] == 0x000F4020, 
            "SIMD4_32i::ANDVA");
    }
    // MANDVA
    {
        UME::SIMD::SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        UME::SIMD::SIMD4_32i vec1(0x09C9AC81, 0x01C0A301, 0x3956806F, 0x571FC5A0);
        UME::SIMD::SIMDMask4 mask(true, false, false, true);

        vec0.anda(mask, vec1);
        CHECK_CONDITION(
            vec0[0] == 0x01012000 && vec0[1] == 0xF00F0F10 && vec0[2] == 0x0FF0F0F0 && vec0[3] == 0x000F4020, 
            "SIMD4_32i::MANDVA");
    }
    // MANDSA
    {
        UME::SIMD::SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        int32_t val1 = 0x571FC5A0;
        UME::SIMD::SIMDMask4 mask(true, false, false, true);
        
        vec0.anda(mask, val1);
        CHECK_CONDITION(
            vec0[0] == 0x53130120 && vec0[1] == 0xF00F0F10 && vec0[2] == 0x0FF0F0F0 && vec0[3] == 0x000F4020, 
            "SIMD4_32i::MANDSA");
    }
    // NOT
    {
        UME::SIMD::SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        UME::SIMD::SIMD4_32i vec1;
        vec1 = vec0.notv();
        CHECK_CONDITION(
            vec1[0] == 0x0CCCCCCB && vec1[1] == 0x0FF0F0EF && vec1[2] == 0xF00F0F0F && vec1[3] == 0xFFF0BDC0, 
            "SIMD4_32i::NOT");
    }
    // MNOT
    {
        UME::SIMD::SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        UME::SIMD::SIMD4_32i vec1;
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        vec1 = vec0.notv(mask);
        CHECK_CONDITION(
            vec1[0] == 0x0CCCCCCB && vec1[1] == 0xF00F0F10 && vec1[2] == 0xF00F0F0F && vec1[3] == 0x000F423F, 
            "SIMD4_32i::MNOT");
    }
    // MNOTA
    {
        UME::SIMD::SIMD4_32i vec0(0xF3333334, 0xF00F0F10, 0x0FF0F0F0, 0x000F423F);
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        vec0.nota(mask);
        CHECK_CONDITION(
            vec0[0] == 0x0CCCCCCB && vec0[1] == 0xF00F0F10 && vec0[2] == 0xF00F0F0F && vec0[3] == 0x000F423F, 
            "SIMD4_32i::MNOTA");
    }
    // MBLENDV
    {
        UME::SIMD::SIMD4_32i vec0(3), vec1(-5);
        UME::SIMD::SIMD4_32i vec2(-2);
        UME::SIMD::SIMDMask4 mask(true, false, false, true);
        vec2 = vec0.blend(mask, vec1);
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == -5 && vec2[2] == -5 && vec2[3] == 3, "SIMD4_32i::MBLENDV");
    }
    // MBLENDS
    {
        UME::SIMD::SIMD4_32i vec0(3);
        int32_t val1 = -5;
        UME::SIMD::SIMD4_32i vec2(-2);
        UME::SIMD::SIMDMask4       mask(true, false, false, true);
        vec2 = vec0.blend(mask, val1);
        CHECK_CONDITION(vec2[0] == 3 && vec2[1] == -5 && vec2[2] == -5 && vec2[3] == 3, "SIMD4_32i::MBLENDS");
    }
    // HAND   
    {
        UME::SIMD::SIMD4_32i vec0(0xF3333304, 0xFF0F3F00, 0x0FF0F000, 0x0F0F720F);
        int32_t val1;
        val1 = vec0.hand();
        CHECK_CONDITION(val1 == 0x03003000, "SIMD4_32i::HAND");
    }
    // MHAND
    {
        UME::SIMD::SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        UME::SIMD::SIMDMask4 mask(true, false, false, true);
        int32_t val1;
        val1 = vec0.hand(mask);
        CHECK_CONDITION(val1 == 0x00030204, "SIMD4_32i::MHAND");
    }
    // HANDS
    {
        UME::SIMD::SIMD4_32i vec0(0xF3333304, 0xFF0F3F00, 0x0FF0F000, 0x0F0F720F);
        int32_t val1;
        int32_t val2 = 0x03003000;
        val1 = vec0.hand(val2);
        CHECK_CONDITION(val1 == 0x03003000, "SIMD4_32i::HANDS");
    }
    // MHANDS
    {
        UME::SIMD::SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        UME::SIMD::SIMDMask4 mask(true, false, false, true);
        int32_t val1;
        int32_t val2 = 0x00010004;
        val1 = vec0.hand(mask, val2);
        CHECK_CONDITION(val1 == 0x00010004, "SIMD4_32i::MHANDS");
    }    
    // HOR
    {
        UME::SIMD::SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        int32_t val1;
        val1 = vec0.hor();
        CHECK_CONDITION(val1 == 0xFFFFFF0F, "SIMD4_32i::HOR");
    }
    // MHOR
    {
        UME::SIMD::SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        UME::SIMD::SIMDMask4 mask(false, false, true, true);
        int32_t val1;
        val1 = vec0.hor(mask);
        CHECK_CONDITION(val1 == 0x0FFFF20F, "SIMD4_32i::MHOR");
    }
    // HORS
    {
        UME::SIMD::SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        int32_t val1;
        int32_t val2 = 0x00000030;
        val1 = vec0.hor(val2);
        CHECK_CONDITION(val1 == 0xFFFFFF3F, "SIMD4_32i::HORS");
    }
    // MHORS
    {
        UME::SIMD::SIMD4_32i vec0(0xF3333304, 0xF00F0F00, 0x0FF0F000, 0x000F420F);
        UME::SIMD::SIMDMask4 mask(false, false, true, true);
        int32_t val1;
        int32_t val2 = 0x00000030;
        val1 = vec0.hor(mask, val2);
        CHECK_CONDITION(val1 == 0x0FFFF23F, "SIMD4_32i::MHORS");
    }
    // HMUL
    {
        UME::SIMD::SIMD4_32i vec0(1, -2, 3, 4);
        int32_t val1 = 0;
        val1 = vec0.hmul();
        CHECK_CONDITION(val1 == -24, "SIMD4_32i::HMUL");
    }
    // MHMUL
    {
        UME::SIMD::SIMD4_32i vec0(1, -2, 3, 4);
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        int32_t val1 = 0;
        val1 = vec0.hmul(mask);
        CHECK_CONDITION(val1 == 3, "SIMD4_32i::MHMUL");
    }
    // HMULS
    {
        UME::SIMD::SIMD4_32i vec0(1, -2, 3, 4);
        int32_t val1 = -42;
        int32_t res = 0;
        res = vec0.hmul(val1);
        CHECK_CONDITION(res == 1008, "SIMD4_32i::HMULS");
    }
    // MHMULS
    {
        UME::SIMD::SIMD4_32i vec0(1, -2, 3, 4);
        UME::SIMD::SIMDMask4 mask(true, false, true, false);
        int32_t val1 = -42;
        int32_t res = 0;
        res = vec0.hmul(mask, val1);
        CHECK_CONDITION(res == -126, "SIMD4_32i::HMULS");
    }
    // NEG
    {
        UME::SIMD::SIMD4_32i vec0(9, -8, 7, 6);
        UME::SIMD::SIMD4_32i vec1;
        vec1 = vec0.neg();
        CHECK_CONDITION(vec1[0] == -9 && vec1[1] == 8 && vec1[2] == -7 && vec1[3] == -6, "SIMD4_32i::NEG");
    }
    return g_failCount;
}

int test_UME_SIMD4_32f(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD4_32f test";
    INIT_TEST(header, supressMessages);

    {
        UME::SIMD::SIMD4_32f vec0;
        CHECK_CONDITION(true, "SIMD4_32f()"); 
    }
    
    {
        UME::SIMD::SIMD4_32f vec0(3.14f);
        CHECK_CONDITION(vec0[3] == 3.14f, "SIMD4_32f(float f)");
    }

    {
        UME::SIMD::SIMD4_32f vec0(1.11f, 2.22f, 3.33f, 4.44f);
        CHECK_CONDITION(vec0[3] == 4.44f && vec0[1] == 2.22f, "SIMD4_32f(float f0 ... float f3)");
    }

    {
        UME::SIMD::SIMD4_32f vec0(3.14f);
        UME::SIMD::SIMD4_32f vec1(2.71f);
        vec1 = vec0;
        CHECK_CONDITION(vec1[3] == 3.14f, "SIMD4_32f::operator=");
    }

    // ADDV
    {
        UME::SIMD::SIMD4_32f vec0(1.0f);
        UME::SIMD::SIMD4_32f vec1(2.0f);
        UME::SIMD::SIMD4_32f vec2;
        vec2 = vec0.add(vec1);
        CHECK_CONDITION(vec2[0] == 3.0f, "SIMD4_32f::ADDV");
    }
    // ADDS
    {
        UME::SIMD::SIMD4_32f vec0(1.0f);
        float val1 = 2.0f;
        UME::SIMD::SIMD4_32f vec2;
        vec2 = vec0.add(val1);
        CHECK_CONDITION(vec2[3] > 2.99f && vec2[3] < 3.01f, "SIMD4_32f::ADDS");
    }
    // MULV
    {
        UME::SIMD::SIMD4_32f vec0(3.0f);
        UME::SIMD::SIMD4_32f vec1(4.0f);
        UME::SIMD::SIMD4_32f vec2;
        vec2 = vec0.mul(vec1);
        CHECK_CONDITION(vec2[1] > 11.99f && vec2[1] < 12.01f, "SIMD4_32f::MULV");
    }
    // MULS
    {
        UME::SIMD::SIMD4_32f vec0(3.0f);
        float val1 = 4.0f;
        UME::SIMD::SIMD4_32f vec2;
        vec2 = vec0.mul(val1);
        CHECK_CONDITION(vec2[1] > 11.99f && vec2[1] < 12.01f, "SIMD4_32f::MULS");
    }
    // SQR
    {
        UME::SIMD::SIMD4_32f vec0(4.0f);
        UME::SIMD::SIMD4_32f vec1;
        //vec1 = vec0.sqr()
        CHECK_CONDITION(vec1[0] > 15.99f && vec1[3] < 16.01f, "SIMD4_32f::SQR");
    }
    // SQRT
    {
        UME::SIMD::SIMD4_32f vec0(4.0f);
        UME::SIMD::SIMD4_32f vec1;
        vec1 = vec0.sqrt();
        CHECK_CONDITION(vec1[0] > 1.99f && vec1[3] < 2.01f, "SIMD4_32f::SQRT")
    }

    {
        UME::SIMD::SIMD4_32f vec0(3.8f);
        UME::SIMD::SIMD4_32f vec1;
        //vec1 = round(vec0);
        UME::SIMD::SIMD4_32f vec2;
        //vec2 = truncate(vec0);
        CHECK_CONDITION(vec1[0] > 3.99f && vec1[3] < 4.01f, "SIMD4_32f::ROUND");
        CHECK_CONDITION(vec2[0] > 2.99f && vec2[3] < 3.01f, "SIMD4_32f::TRUNC");
    }

    return g_failCount;
}

int test_UME_SIMD2_64(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD2_64 test";
    INIT_TEST(header, supressMessages);

    {
        UME::SIMD::SIMD2_64i vec6;  
        UME::SIMD::SIMD2_64u vec7;
        CHECK_CONDITION(true, "SIMD2_64()"); 
    }

    return g_failCount;
}

#endif
