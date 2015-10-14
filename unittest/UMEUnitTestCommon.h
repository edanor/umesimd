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
//  7th Framework programme Marie Curie Actions under grant PITN-GA-2012-3VEC_LEN596".
//

#ifndef UME_UNIT_TEST_COMMON_H_
#define UME_UNIT_TEST_COMMON_H_

#include <iostream>
#include "../UMEBasicTypes.h"
#include "UMEUnitTestDataSets.h"

int g_totalTests = 0;
int g_totalFailed = 0;
int g_testMaxId = 0;
int g_failCount = 0;
bool g_allSuccess = true;
bool g_supressMessages = false;
char *g_test_header_ptr = NULL;

#define INIT_TEST(test_header_ptr, supressMesages) { \
    g_test_header_ptr = (test_header_ptr); \
    g_failCount = 0; \
    g_testMaxId = 0; \
    g_allSuccess = true; \
    g_supressMessages = supressMesages;}

#define SUPRESS_MESSAGES() { g_supressMessages = true; }
#define CHECK_CONDITION(cond, msg) \
    g_totalTests++; \
    g_testMaxId++; \
    if(!(cond)){\
        if(g_supressMessages == false) std::cout << "FAIL " << g_test_header_ptr << " Id: " << g_testMaxId << " - " << (msg) << std::endl; \
        g_totalFailed++; \
        g_failCount++; \
        g_allSuccess = false; \
    } \
    else \
    { \
        if(g_supressMessages == false) std::cout << "OK   " << g_test_header_ptr << " Id: " << g_testMaxId << " - " << (msg) << std::endl;  \
    }

#define PRINT_MESSAGE(msg) if(g_supressMessages == false) std::cout << g_test_header_ptr <<  msg << std::endl;

// This PI value is used over all unit tests. Defining it here makes it is 
// possible to use different values in real codes.
#define UME_PI_D 3.141592653589793238462643383279502884197VEC_LEN93993751058209749445923078VEC_LEN4062
#define UME_2PI_D (2.0*UME_PI_D)
#define UME_PI_F float(UME_PI_D)
#define UME_2PI_F (2.0f*UME_PI_F)


bool valueInRange(float value, float expectedValue, float errMargin) {

    if(expectedValue == 0.0f)
    {
        return (errMargin >= value) & ((-errMargin) <= value);
    }
    else if(value > 0.0f)
    {
        return ((expectedValue)*(1.0f + errMargin) >= value) 
             & ((expectedValue)*(1.0f - errMargin) <= value);
    }
    else
    {
        return ((expectedValue)*(1.0f + errMargin) <= value)
             & ((expectedValue)*(1.0f - errMargin) >= value);
    }
}

bool valueInRange(double value, double expectedValue, double errMargin) {
    if(value >= 0.0f)
    {
        return ((expectedValue)*(1.0f + errMargin) >= value) 
             & ((expectedValue)*(1.0f - errMargin) <= value);
    }
    else
    {
        return ((expectedValue)*(1.0f + errMargin) <= value)
             & ((expectedValue)*(1.0f - errMargin) >= value);
    }
}

bool valueInRange(uint32_t value, uint32_t expectedValue, float errMargin) {
    return valueInRange((float)value, (float)expectedValue, errMargin);
}

bool valueInRange(int32_t value, int32_t expectedValue, float errMargin) {
    return valueInRange((float)value, (float)expectedValue, errMargin);
}

bool valuesExact(int32_t const *values, int32_t const *expectedValues, unsigned int count) 
{
    bool retval = true;
    for(unsigned int i = 0; i < count; i++) {
        if(values[i] != expectedValues[i])
        {
            retval = false;
            break;
        }
    }
    return retval;
}

bool valuesExact(uint32_t const *values, uint32_t const *expectedValues, unsigned int count) 
{
    bool retval = true;
    for(unsigned int i = 0; i < count; i++) {
        if(values[i] != expectedValues[i])
        {
            retval = false;
            break;
        }
    }
    return retval;
}

bool valuesExact(bool const *values, bool const *expectedValues, unsigned int count)
{
    bool retval = true;
    for(unsigned int i = 0; i < count; i++) {
        if(values[i] != expectedValues[i])
        {
            retval = false;
            break;
        }
    }
    return retval;
}

bool valuesInRange(float const *values, float const *expectedValues, unsigned int count, float errMargin)
{
    bool retval = true;
    for(unsigned int i = 0; i < count; i++) {
        if(!valueInRange(values[i], expectedValues[i], errMargin))
        {
            retval = false;
            break;
        }
    }
    return retval;
}

bool valuesInRange(double const *values, double const *expectedValues, unsigned int count, double errMargin)
{
    bool retval = true;
    for(unsigned int i = 0; i < count; i++) {
        if(!valueInRange(values[i], expectedValues[i], errMargin))
        {
            retval = false;
            break;
        }
    }
    return retval;
}

// This is a dirty hack to use the same testing function for both int and float types... 
bool valuesInRange(uint32_t const *values, uint32_t const *expectedValues, unsigned int count, double errMargin)
{
    return valuesExact(values, expectedValues, count);
}

// This is a dirty hack to use the same testing function for both int and float types... 
bool valuesInRange(int32_t const *values, int32_t const *expectedValues, unsigned int count, double errMargin)
{
    return valuesExact(values, expectedValues, count);
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLANDTest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        MASK_TYPE m2 = m0.land(m1);
        m2.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LAND[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LAND");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        MASK_TYPE m2 = m0 & m1;
        m2.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LAND[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LAND(operator &)");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLANDATest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        m0.landa(m1);
        m0.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LAND[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDA");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        m0 &= m1;
        m0.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LAND[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDA(operator &=)");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLORTest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        MASK_TYPE m2 = m0.lor(m1);
        m2.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LOR[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LOR");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        MASK_TYPE m2 = m0 | m1;
        m2.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LOR[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LOR(operator |)");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLORATest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        m0.lora(m1);
        m0.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LOR[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LORA");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        m0 |= m1;
        m0.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LOR[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LORA(operator |=)");
    }
}


template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLXORTest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        MASK_TYPE m2 = m0.lxor(m1);
        m2.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LXOR[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LXOR");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        MASK_TYPE m2 = m0 ^ m1;
        m2.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LXOR[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LXOR(operator ^)");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLXORATest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        m0.lxora(m1);
        m0.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LXOR[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LXORA");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        m0 ^= m1;
        m0.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LXOR[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LXORA(operator ^=)");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLNOTTest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1 = m0.lnot();
        m1.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LNOT[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LNOT");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1 = !m0;
        m1.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LNOT[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LNOT(operator !)");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLNOTATest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        m0.lnota();
        m0.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LNOT[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LNOTA");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericHLANDTest()
{
    MASK_TYPE m0(DATA_SET::inputs::maskA);
    bool value = m0.hland();
    bool expected = DATA_SET::outputs::HLAND[VEC_LEN-1];
    CHECK_CONDITION(value == expected, "HLAND");
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericHLORTest()
{
    MASK_TYPE m0(DATA_SET::inputs::maskA);
    bool value = m0.hlor();
    bool expected = DATA_SET::outputs::HLOR[VEC_LEN-1];
    CHECK_CONDITION(value == expected, "HLOR");
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericHLXORTest()
{
    MASK_TYPE m0(DATA_SET::inputs::maskA);
    bool value = m0.hlxor();
    bool expected = DATA_SET::outputs::HLXOR[VEC_LEN-1];
    CHECK_CONDITION(value == expected, "HLXOR");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericADDVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0.add(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDV, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "ADDV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0 + vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDV, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "ADDV(operator+)");
    }   
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMADDVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.add(mask, vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MADDV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MADDV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericADDSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.add(DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::ADDS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "ADDS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMADDSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.add(mask, DATA_SET::inputs::scalarA);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MADDS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MADDS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericADDVATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0.adda(vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDV, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "ADDVA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0 += vec1;
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDV, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "ADDVA(operator+=)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMADDVATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.adda(mask, vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MADDV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MADDVA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericADDSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.adda(DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::ADDS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "ADDSA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMADDSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.adda(mask, DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MADDS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MADDSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericPOSTINCTest()
{
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.postinc();
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFINC, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, DATA_SET::inputs::inputA, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "POSTINC");
    }
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0++;
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFINC, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, DATA_SET::inputs::inputA, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "POSTINC(operator++(int))");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMPOSTINCTest()
{
    SCALAR_TYPE values0[VEC_LEN];
    SCALAR_TYPE values1[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.postinc(mask);
    vec0.store(values0);
    vec1.store(values1);
    bool inRange0 = valuesInRange(values0, DATA_SET::outputs::MPOSTPREFINC, VEC_LEN, 0.01f);
    bool inRange1 = valuesInRange(values1, DATA_SET::inputs::inputA, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange0 && inRange1, "MPOSTINC");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericPREFINCTest()
{
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.prefinc();
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFINC, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, DATA_SET::outputs::POSTPREFINC, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "PREFINC");
    }
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = ++vec0;
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFINC, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, DATA_SET::outputs::POSTPREFINC, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "PREFINC(operator++())");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMPREFINCTest()
{
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.prefinc(mask);
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::MPOSTPREFINC, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, DATA_SET::outputs::MPOSTPREFINC, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "MPREFINC");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSUBVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0.sub(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SUBV, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SUBV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0 - vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SUBV, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SUBV(operator-)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSUBVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.sub(mask, vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MSUBV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSUBSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.sub(DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::SUBS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "SUBS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSUBSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.sub(mask, DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MSUBS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSUBVATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0.suba(vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SUBV, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SUBVA");
    }
    
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0 -= vec1;
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SUBV, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SUBVA(operator-=)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSUBVATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.suba(mask, vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MSUBVA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSUBSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.suba(DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::SUBS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "SUBSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSUBSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.suba(mask, DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MSUBSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSUBFROMVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    VEC_TYPE vec2 = vec0.subfrom(vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::SUBFROMV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "SUBFROMV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSUBFROMVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.subfrom(mask, vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBFROMV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MSUBFROMV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSUBFROMSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec2 = vec0.subfrom(DATA_SET::inputs::scalarA);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::SUBFROMS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "SUBFROMS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSUBFROMSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.subfrom(mask, DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBFROMS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MSUBFROMS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSUBFROMVATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    vec0.subfroma(vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::SUBFROMV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "SUBFROMVA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSUBFROMVATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.subfroma(mask, vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBFROMV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MSUBFROMVA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSUBFROMSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.subfroma(DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::SUBFROMS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "SUBFROMSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSUBFROMSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.subfroma(mask, DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBFROMS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MSUBFROMSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericPOSTDECTest()
{
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.postdec();
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFDEC, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, DATA_SET::inputs::inputA, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "POSTDEC");
    }
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0--;
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFDEC, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, DATA_SET::inputs::inputA, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "POSTDEC(operator--(int))");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMPOSTDECTest()
{
    SCALAR_TYPE values0[VEC_LEN];
    SCALAR_TYPE values1[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.postdec(mask);
    vec0.store(values0);
    vec1.store(values1);
    bool inRange0 = valuesInRange(values0, DATA_SET::outputs::MPOSTPREFDEC, VEC_LEN, 0.01f);
    bool inRange1 = valuesInRange(values1, DATA_SET::inputs::inputA, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange0 && inRange1, "MPOSTDEC");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericPREFDECTest()
{
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.prefdec();
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFDEC, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, DATA_SET::outputs::POSTPREFDEC, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "PREFDEC");
    }
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = --vec0;
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFDEC, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, DATA_SET::outputs::POSTPREFDEC, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "PREFDEC(operator--())");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMPREFDECTest()
{
    SCALAR_TYPE values0[VEC_LEN];
    SCALAR_TYPE values1[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.prefdec(mask);
    vec0.store(values0);
    vec1.store(values1);
    bool inRange0 = valuesInRange(values0, DATA_SET::outputs::MPOSTPREFDEC, VEC_LEN, 0.01f);
    bool inRange1 = valuesInRange(values1, DATA_SET::outputs::MPOSTPREFDEC, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange0 && inRange1, "MPREFDEC");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMULVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0.mul(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULV, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MULV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0 * vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULV, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MULV(operator*)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMULVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.mul(mask, vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMULV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MMULV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMULSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.mul(DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MULS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MULS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMULSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.mul(mask, DATA_SET::inputs::scalarA);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMULS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MMULS");
}
        
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMULVATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0.mula(vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULV, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MULVA");
    }
    
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0 *= vec1;
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULV, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MULVA(operator*)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMULVATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.mula(mask, vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMULV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MMULVA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMULSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.mula(DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MULS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MULSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMULSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.mula(mask, DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMULS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MMULSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericDIVVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0.div(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::DIVV, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "DIVV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0 / vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::DIVV, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "DIVV(operator/)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMDIVVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.div(mask, vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MDIVV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MDIVV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericDIVSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.div(DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::DIVS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "DIVS");
}

    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMDIVSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.div(mask, DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MDIVS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MDIVS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericDIVVATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0.diva(vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::DIVV, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "DIVVA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0 /= vec1;
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::DIVV, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "DIVVA(operator/)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMDIVVATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.diva(mask, vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MDIVV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MDIVVA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericDIVSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.diva(DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::DIVS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "DIVSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMDIVSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.diva(mask, DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MDIVS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MDIVSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRCPTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.rcp();
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::RCP, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "RCP");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRCPTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.rcp(mask);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MRCP, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MRCP");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRCPSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.rcp(DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::RCPS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "RCPS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRCPSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.rcp(mask, DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MRCPS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MRCPS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRCPATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.rcpa();
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::RCP, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "RCPA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRCPATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.rcpa(mask);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MRCP, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MRCPA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRCPSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.rcpa(DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::RCPS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "RCPSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRCPSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.rcpa(mask, DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MRCPS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MRCPSA");
}
    
template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPEQVTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(true);
    mask = vec0.cmpeq(vec1);
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::CMPEQV, VEC_LEN);
    CHECK_CONDITION(inRange, "CMPEQV");
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPEQSTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(true);
    mask = vec0.cmpeq(DATA_SET::inputs::scalarA);
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::CMPEQS, VEC_LEN);
    CHECK_CONDITION(inRange, "CMPEQS");
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPNEVTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(true);
    mask = vec0.cmpne(vec1);
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::CMPNEV, VEC_LEN);
    CHECK_CONDITION(inRange, "CMPNEV");
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPNESTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(true);
    mask = vec0.cmpne(DATA_SET::inputs::scalarA);
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::CMPNES, VEC_LEN);
    CHECK_CONDITION(inRange, "CMPNES");
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPGTVTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(true);
    mask = vec0.cmpgt(vec1);
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::CMPGTV, VEC_LEN);
    CHECK_CONDITION(inRange, "CMPGTV");
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPGTSTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(true);
    mask = vec0.cmpgt(DATA_SET::inputs::scalarA);
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::CMPGTS, VEC_LEN);
    CHECK_CONDITION(inRange, "CMPGTS");
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPLTVTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(true);
    mask = vec0.cmplt(vec1);
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::CMPLTV, VEC_LEN);
    CHECK_CONDITION(inRange, "CMPLTV");
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPLTSTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(true);
    mask = vec0.cmplt(DATA_SET::inputs::scalarA);
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::CMPLTS, VEC_LEN);
    CHECK_CONDITION(inRange, "CMPLTS");
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPGEVTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(true);
    mask = vec0.cmpge(vec1);
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::CMPGEV, VEC_LEN);
    CHECK_CONDITION(inRange, "CMPGEV");
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPGESTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(true);
    mask = vec0.cmpge(DATA_SET::inputs::scalarA);
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::CMPGES, VEC_LEN);
    CHECK_CONDITION(inRange, "CMPGES");
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPLEVTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(true);
    mask = vec0.cmple(vec1);
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::CMPLEV, VEC_LEN);
    CHECK_CONDITION(inRange, "CMPLEV");
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPLESTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(true);
    mask = vec0.cmple(DATA_SET::inputs::scalarA);
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::CMPLES, VEC_LEN);
    CHECK_CONDITION(inRange, "CMPLES");
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPEVTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    bool value = vec0.cmpe(vec1);
    CHECK_CONDITION(value == DATA_SET::outputs::CMPEV, "CMPEV");
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPESTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    bool value = vec0.cmpe(DATA_SET::inputs::scalarA);
    CHECK_CONDITION(value == DATA_SET::outputs::CMPES, "CMPES");
}
 
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBANDVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    VEC_TYPE vec2 = vec0.band(vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::BANDV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "BANDV");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBANDVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.band(mask, vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBANDV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MBANDV");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBANDSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec2 = vec0.band(DATA_SET::inputs::scalarA);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::BANDS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "BANDS");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBANDSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.band(mask, DATA_SET::inputs::scalarA);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBANDS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MBANDS");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBANDVATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    vec0.banda(vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::BANDV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "BANDVA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBANDVATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.banda(mask, vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBANDV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MBANDVA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBANDSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.banda(DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::BANDS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "BANDSA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBANDSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.banda(mask, DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBANDS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MBANDSA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBORVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    VEC_TYPE vec2 = vec0.bor(vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::BORV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "BORV");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBORVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.bor(mask, vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBORV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MBORV");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBORSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec2 = vec0.bor(DATA_SET::inputs::scalarA);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::BORS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "BORS");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBORSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.bor(mask, DATA_SET::inputs::scalarA);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBORS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MBORS");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBORVATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    vec0.bora(vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::BORV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "BORVA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBORVATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.bora(mask, vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBORV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MBORVA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBORSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.bora(DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::BORS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "BORSA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBORSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.bora(mask, DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBORS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MBORSA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBXORVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    VEC_TYPE vec2 = vec0.bxor(vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::BXORV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "BXORV");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBXORVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.bxor(mask, vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBXORV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MBXORV");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBXORSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec2 = vec0.bxor(DATA_SET::inputs::scalarA);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::BXORS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "BXORS");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBXORSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.bxor(mask, DATA_SET::inputs::scalarA);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBXORS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MBXORS");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBXORVATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    vec0.bxora(vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::BXORV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "BXORVA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBXORVATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.bxora(mask, vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBXORV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MBXORVA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBXORSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.bxora(DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::BXORS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "BXORSA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBXORSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.bxora(mask, DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBXORS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MBXORSA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBNOTTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.bnot();
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::BNOT, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "BNOT");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBNOTTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.bnot(mask);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBNOT, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MBNOT");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBNOTATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.bnota();
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::BNOT, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "BNOTA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBNOTATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.bnota(mask);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBNOT, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MBNOTA");
}
 
        // (Pack/Unpack operations - not available for SIMD1)
        // PACK     - assign vector with two half-length vectors
        // PACKLO   - assign lower half of a vector with a half-length vector
        // PACKHI   - assign upper half of a vector with a half-length vector
        // UNPACK   - Unpack lower and upper halfs to half-length vectors.
        // UNPACKLO - Unpack lower half and return as a half-length vector.
        // UNPACKHI - Unpack upper half and return as a half-length vector.
 
        //(Blend/Swizzle operations)
        // BLENDV   - Blend (mix) two vectors
        // BLENDS   - Blend (mix) vector with scalar (promoted to vector)
        // BLENDVA  - Blend (mix) two vectors and assign
        // BLENDSA  - Blend (mix) vector with scalar (promoted to vector) and
        // assign
        // SWIZZLE  - Swizzle (reorder/permute) vector elements
        // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign
 
        //(Reduction to scalar operations)
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHADDTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    SCALAR_TYPE value = vec0.hadd();
    bool inRange = valueInRange(value, DATA_SET::outputs::HADD[VEC_LEN-1], 0.01f);
    CHECK_CONDITION(inRange, "HADD");
}
        // MHADD - Masked add elements of a vector (horizontal add)
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHMULTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    SCALAR_TYPE value = vec0.hmul();
    bool inRange = valueInRange(value, DATA_SET::outputs::HMUL[VEC_LEN-1], 0.01f);
    CHECK_CONDITION(inRange, "HMUL");
}
        // MHMUL - Masked multiply elements of a vector (horizontal mul)

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHANDTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    SCALAR_TYPE value = vec0.hand();
    bool inRange = valueInRange(value, DATA_SET::outputs::HAND[VEC_LEN-1], 0.01f);
    CHECK_CONDITION(inRange, "HAND");
}
        // MHAND - Masked AND of elements of a vector (horizontal AND)
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHORTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    SCALAR_TYPE value = vec0.hor();
    bool inRange = valueInRange(value, DATA_SET::outputs::HOR[VEC_LEN-1], 0.01f);
    CHECK_CONDITION(inRange, "HOR");
}
        // MHOR  - Masked OR of elements of a vector (horizontal OR)
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHXORTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    SCALAR_TYPE value = vec0.hxor();
    bool inRange = valueInRange(value, DATA_SET::outputs::HXOR[VEC_LEN-1], 0.01f);
    CHECK_CONDITION(inRange, "HXOR");
}
        // MHXOR - Masked XOR of elements of a vector (horizontal XOR)
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericFMULADDVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    VEC_TYPE vec2(DATA_SET::inputs::inputC);
    VEC_TYPE vec3 = vec0.fmuladd(vec1, vec2);
    vec3.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::FMULADDV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "FMULADDV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMFMULADDVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    VEC_TYPE vec2(DATA_SET::inputs::inputC);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec3 = vec0.fmuladd(mask, vec1, vec2);
    vec3.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MFMULADD, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MFMULADDV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericFMULSUBVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    VEC_TYPE vec2(DATA_SET::inputs::inputC);
    VEC_TYPE vec3 = vec0.fmulsub(vec1, vec2);
    vec3.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::FMULSUBV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "FMULSUBV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMFMULSUBVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    VEC_TYPE vec2(DATA_SET::inputs::inputC);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec3 = vec0.fmulsub(mask, vec1, vec2);
    vec3.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MFMULSUBV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MFMULSUBV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericFADDMULVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    VEC_TYPE vec2(DATA_SET::inputs::inputC);
    VEC_TYPE vec3 = vec0.faddmul(vec1, vec2);
    vec3.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::FADDMULV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "FADDMULV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMFADDMULVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    VEC_TYPE vec2(DATA_SET::inputs::inputC);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec3 = vec0.faddmul(mask, vec1, vec2);
    vec3.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MFADDMULV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MFADDMULV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericFSUBMULVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    VEC_TYPE vec2(DATA_SET::inputs::inputC);
    VEC_TYPE vec3 = vec0.fsubmul(vec1, vec2);
    vec3.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::FSUBMULV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "FSUBMULV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMFSUBMULVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    VEC_TYPE vec2(DATA_SET::inputs::inputC);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec3 = vec0.fsubmul(mask, vec1, vec2);
    vec3.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MFSUBMULV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MFSUBMULV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMAXVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    VEC_TYPE vec2 = vec0.max(vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MAXV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MAXV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMAXVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.max(mask, vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMAXV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MMAXV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMAXSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.max(DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MAXS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MAXS");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMAXSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.max(mask, DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMAXS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MMAXS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMAXVATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    vec0.maxa(vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MAXV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MAXVA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMAXVATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.maxa(mask, vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMAXV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MMAXVA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMAXSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.maxa(DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MAXS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MAXSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMAXSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.maxa(mask, DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMAXS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MMAXSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMINVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    VEC_TYPE vec2 = vec0.min(vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MINV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MINV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMINVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.min(mask, vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMINV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MMINV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMINSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.min(DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MINS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MINS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMINSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.min(mask, DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMINS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MMINS");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMINVATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    vec0.mina(vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MINV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MINVA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMINVATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.mina(mask, vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMINV, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MMINVA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMINSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.mina(DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MINS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MINSA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMINSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.mina(mask, DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMINS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MMINSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHMAXTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    SCALAR_TYPE value = vec0.hmax();
    SCALAR_TYPE expected = DATA_SET::outputs::HMAX[VEC_LEN-1];
    bool inRange = valueInRange(value, expected, 0.01f);
    CHECK_CONDITION(inRange, "HMAX");
}

// MHMAX
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHMINTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    SCALAR_TYPE value = vec0.hmin();
    SCALAR_TYPE expected = DATA_SET::outputs::HMIN[VEC_LEN-1];
    bool inRange = valueInRange(value, expected, 0.01f);
    CHECK_CONDITION(inRange, "HMIN");
}

// MHMIN

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericNEGTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.neg();
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::NEG, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "NEG");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 =  -vec0;
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::NEG, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "NEG(operator-)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMNEGTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.neg(mask);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MNEG, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MNEG");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericNEGATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.nega();
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::NEG, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "NEGA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMNEGATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.nega(mask);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MNEG, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MNEGA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericABSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.abs();
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::ABS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "ABS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMABSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.abs(mask);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MABS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MABS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericABSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.absa();
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::ABS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "ABSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMABSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.absa(mask);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MABS, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MABSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSQRTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.sqr();
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::SQR, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "SQR");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSQRTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.sqr(mask);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSQR, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MSQR");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSQRATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.sqra();
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::SQR, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "SQRA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSQRATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.sqra(mask);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSQR, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MSQRA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericROUNDTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.round();
    vec1.store(values);
    //bool inRange = valuesInRange(values, DATA_SET::outputs::ROUND, VEC_LEN, 0.01f);
    CHECK_CONDITION(false, "ROUND");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename VEC_INT_TYPE, int VEC_LEN, typename DATA_SET>
void genericTRUNCTest()
{
    int32_t values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_INT_TYPE vec1 = vec0.trunc();
    vec1.store(values);
    bool exact = valuesExact(values, DATA_SET::outputs::TRUNC, VEC_LEN);
    CHECK_CONDITION(exact, "TRUNC");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename VEC_INT_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMTRUNCTest()
{
    int32_t values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_INT_TYPE vec1 = vec0.trunc(mask); 
    vec1.store(values);
    bool exact = valuesExact(values, DATA_SET::outputs::MTRUNC, VEC_LEN);
    CHECK_CONDITION(exact, "MTRUNC");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericFLOORTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.floor();
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::FLOOR, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "FLOOR");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMFLOORTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.floor(mask);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MFLOOR, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "MFLOOR");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericCEILTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.ceil();
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::CEIL, VEC_LEN, 0.05f);
    CHECK_CONDITION(inRange, "CEIL");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMCEILTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.ceil(mask);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MCEIL, VEC_LEN, 0.05f);
    CHECK_CONDITION(inRange, "MCEIL");
}
  
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericISFINTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask = vec0.isfin();
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::ISFIN, VEC_LEN);
    CHECK_CONDITION(inRange, "ISFIN");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericISINFTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask = vec0.isinf();
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::ISINF, VEC_LEN);
    CHECK_CONDITION(inRange, "ISINF");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericISANTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask = vec0.isan();
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::ISAN, VEC_LEN);
    CHECK_CONDITION(inRange, "ISAN");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericISNANTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask = vec0.isnan();
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::ISNAN, VEC_LEN);
    CHECK_CONDITION(inRange, "ISNAN");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericISNORMTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask = vec0.isnorm();
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::ISNORM, VEC_LEN);
    CHECK_CONDITION(inRange, "ISNORM");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericISSUBTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask = vec0.issub();
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::ISSUB, VEC_LEN);
    CHECK_CONDITION(inRange, "ISSUB");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericISZEROTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask = vec0.iszero();
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::ISZERO, VEC_LEN);
    CHECK_CONDITION(inRange, "ISZERO");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericISZEROSUBTest()
{
    bool values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask = vec0.iszerosub();
    mask.store(values);
    bool inRange = valuesExact(values, DATA_SET::outputs::ISZEROSUB, VEC_LEN);
    CHECK_CONDITION(inRange, "ISZEROSUB");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSINTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.sin();
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::SIN, VEC_LEN, 0.1f);
    CHECK_CONDITION(inRange, "SIN");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSINTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.sin(mask);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSIN, VEC_LEN, 0.1f);
    CHECK_CONDITION(inRange, "MSIN");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericCOSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.cos();
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::COS, VEC_LEN, 0.1f);
    CHECK_CONDITION(inRange, "COS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMCOSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.cos(mask);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MCOS, VEC_LEN, 0.1f);
    CHECK_CONDITION(inRange, "MCOS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericTANTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.tan();
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::TAN, VEC_LEN, 0.1f);
    CHECK_CONDITION(inRange, "TAN");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMTANTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.tan(mask);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MTAN, VEC_LEN, 0.1f);
    CHECK_CONDITION(inRange, "MTAN");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericCTANTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.ctan();
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::CTAN, VEC_LEN, 0.1f);
    CHECK_CONDITION(inRange, "CTAN");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMCTANTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.ctan(mask);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MCTAN, VEC_LEN, 0.1f);
    CHECK_CONDITION(inRange, "MCTAN");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericBaseInterfaceTest()
{   
    // ASSIGNV
    // MASSIGNV
    // ASSIGNS
    // MASSIGNS
    // PREFETCH0
    // PREFETCH1
    // PREFETCH2
    // LOAD
    // MLOAD
    // LOADA
    // MLOADA
    // STORE
    // MSTORE
    // STOREA
    // MSTOREA
    // SWIZZLE
    // SWIZZLEA
    genericADDVTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMADDVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericADDSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMADDSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericADDVATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMADDVATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericADDSATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMADDSATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    // SADDV
    // MSADDV
    // SADDS
    // MSADDS
    // SADDVA
    // MSADDVA
    // SADDSA
    // MSADDSA
    genericPOSTINCTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMPOSTINCTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericPREFINCTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMPREFINCTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericSUBVTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMSUBVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericSUBSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMSUBSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericSUBVATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMSUBVATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericSUBSATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMSUBSATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    // SSUBV
    // MSSUBV
    // SSUBS
    // MSSUBS
    // SSUBVA
    // MSSUBVA
    // SSUBSA
    // MSSUBSA
    genericSUBFROMVTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMSUBFROMVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericSUBFROMSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMSUBFROMSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericSUBFROMVATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMSUBFROMVATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericSUBFROMSATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMSUBFROMSATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericPOSTDECTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMPOSTDECTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericPREFDECTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMPREFDECTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericMULVTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMMULVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericMULSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMMULSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericMULVATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMMULVATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericMULSATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMMULSATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericDIVVTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMDIVVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericDIVSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMDIVSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericDIVVATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMDIVVATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericDIVSATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMDIVSATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericRCPTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMRCPTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericRCPSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMRCPSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericRCPATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMRCPATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericRCPSATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMRCPSATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();

    genericCMPEQVTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCMPEQSTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCMPNEVTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCMPNESTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCMPGTVTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCMPGTSTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCMPLTVTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCMPLTSTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCMPGEVTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCMPGESTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCMPLEVTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCMPLESTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCMPEVTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCMPESTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();

    // BLENDV
    // BLENDS
    // BLENDVA
    // BLENDSA

    
    genericHADDTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    // MHADD
    // HADDS
    // MHADDS
    genericHMULTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    // MHMUL
    // HMULS
    // MHMULS

    genericFMULADDVTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMFMULADDVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericFMULSUBVTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMFMULSUBVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericFADDMULVTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMFADDMULVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericFSUBMULVTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMFSUBMULVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();

    genericMAXVTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMMAXVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericMAXSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMMAXSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericMAXVATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMMAXVATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericMAXSATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMMAXSATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericMINVTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMMINVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericMINSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMMINSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericMINVATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMMINVATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericMINSATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMMINSATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericHMAXTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    // MHMAX
    // IMAX
    // MIMAX
    genericHMINTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    // MHMIN
    // IMIN
    // MIMIN

    // POWV
    // MPOWV
    // POWS
    // MPOWS

}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericBitwiseInterfaceTest()
{
    genericBANDVTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMBANDVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericBANDSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMBANDSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericBANDVATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMBANDVATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericBANDSATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMBANDSATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericBORVTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMBORVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericBORSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMBORSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericBORVATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMBORVATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericBORSATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMBORSATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericBXORVTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMBXORVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericBXORSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMBXORSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericBXORVATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMBXORVATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericBXORSATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMBXORSATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericBNOTTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMBNOTTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericBNOTATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMBNOTATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();

    genericHANDTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    // MHAND
    // HANDS
    // MHANDS
    genericHORTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    // MHOR
    // HORS
    // MHORS
    genericHXORTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    // MHXOR
    // HXORS
    // MHXORS

}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericGatherScatterInterfaceTest()
{
    
    // GATHER
    // MGATHER
    // MGATHERV
    // SCATTER
    // MSCATTER
    // SCATTERV
    // MSCATTERV
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericShiftRotateInterfaceTest()
{
    // LSHV
    // MLSHV
    // LSHS
    // MLSHS
    // LSHVA
    // MLSHVA
    // LSHSA
    // MLSHSA
    // RSHV
    // MRSHV
    // RSHS
    // MRSHS
    // RSHVA
    // MRSHVA
    // RSHSA
    // MRSHSA
    // ROLV
    // MROLV
    // ROLS
    // MROLS
    // ROLVA
    // MROLVA
    // ROLSA
    // MROLSA
    // RORV
    // MRORV
    // RORS
    // MRORS
    // RORVA
    // MRORVA
    // RORSA
    // MRORSA
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericSignInterfaceTest()
{
    genericNEGTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMNEGTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericNEGATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMNEGATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericABSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMABSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericABSATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMABSATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename VEC_INT_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericFloatInterfaceTest()
{
    genericROUNDTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    // MROUND
    genericTRUNCTest<VEC_TYPE, SCALAR_TYPE, VEC_INT_TYPE, VEC_LEN, DATA_SET>();
    genericMTRUNCTest<VEC_TYPE, SCALAR_TYPE, VEC_INT_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericFLOORTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMFLOORTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCEILTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMCEILTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericISFINTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericISINFTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericISANTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericISNANTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericISNORMTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericISSUBTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericISZEROTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericISZEROSUBTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    
    genericSQRTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMSQRTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericSQRATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMSQRATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    /*genericSQRTTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMSQRTTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericSQRTATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMSQRTATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    
    genericSINTest<VEC_TYPE, VEC_LEN, DATA_SET>();
    genericMSINTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCOSTest<VEC_TYPE, VEC_LEN, DATA_SET>();
    genericMCOSTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericTANTest<VEC_TYPE, VEC_LEN, DATA_SET>();
    genericMTANTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCTANTest<VEC_TYPE, VEC_LEN, DATA_SET>();
    genericMCTANTest<VEC_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();*/
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMaskTest() {
    genericLANDTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLANDATest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLORTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLORATest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLXORTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLXORATest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLNOTTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLNOTATest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericHLANDTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericHLORTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericHLXORTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericUintTest() {
    genericBaseInterfaceTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericBitwiseInterfaceTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericGatherScatterInterfaceTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericShiftRotateInterfaceTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>(); 
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericIntTest() {
    genericBaseInterfaceTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericBitwiseInterfaceTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericGatherScatterInterfaceTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericShiftRotateInterfaceTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    //genericSignInterfaceTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename VEC_INT_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericFloatTest() {
    genericBaseInterfaceTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericGatherScatterInterfaceTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericSignInterfaceTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericFloatInterfaceTest<VEC_TYPE, SCALAR_TYPE, VEC_INT_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
}

#endif
