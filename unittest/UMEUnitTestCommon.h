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
#include "UMEUnitTestDataSets8.h"
#include "UMEUnitTestDataSets16.h"
#include "UMEUnitTestDataSets32.h"
#include "UMEUnitTestDataSets64.h"

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

bool valueInRange(uint64_t value, uint64_t expectedValue, float errMargin) {
    return valueInRange((float)value, (float)expectedValue, errMargin);
}

bool valueInRange(int64_t value, int64_t expectedValue, float errMargin) {
    return valueInRange((float)value, (float)expectedValue, errMargin);
}

bool valuesExact(uint8_t const *values, uint8_t const *expectedValues, unsigned int count)
{
    bool retval = true;
    for (unsigned int i = 0; i < count; i++) {
        if (values[i] != expectedValues[i])
        {
            retval = false;
            break;
        }
    }
    return retval;
}

bool valuesExact(int8_t const *values, int8_t const *expectedValues, unsigned int count)
{
    bool retval = true;
    for (unsigned int i = 0; i < count; i++) {
        if (values[i] != expectedValues[i])
        {
            retval = false;
            break;
        }
    }
    return retval;
}

bool valuesExact(uint16_t const *values, uint16_t const *expectedValues, unsigned int count)
{
    bool retval = true;
    for (unsigned int i = 0; i < count; i++) {
        if (values[i] != expectedValues[i])
        {
            retval = false;
            break;
        }
    }
    return retval;
}

bool valuesExact(int16_t const *values, int16_t const *expectedValues, unsigned int count)
{
    bool retval = true;
    for (unsigned int i = 0; i < count; i++) {
        if (values[i] != expectedValues[i])
        {
            retval = false;
            break;
        }
    }
    return retval;
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

bool valuesExact(int64_t const *values, int64_t const *expectedValues, unsigned int count)
{
    bool retval = true;
    for (unsigned int i = 0; i < count; i++) {
        if (values[i] != expectedValues[i])
        {
            retval = false;
            break;
        }
    }
    return retval;
}

bool valuesExact(uint64_t const *values, uint64_t const *expectedValues, unsigned int count)
{
    bool retval = true;
    for (unsigned int i = 0; i < count; i++) {
        if (values[i] != expectedValues[i])
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
bool valuesInRange(uint8_t const *values, uint8_t const *expectedValues, unsigned int count, double errMargin)
{
    return valuesExact(values, expectedValues, count);
}

bool valuesInRange(int8_t const *values, int8_t const *expectedValues, unsigned int count, double errMargin)
{
    return valuesExact(values, expectedValues, count);
}

bool valuesInRange(uint16_t const *values, uint16_t const *expectedValues, unsigned int count, double errMargin)
{
    return valuesExact(values, expectedValues, count);
}

bool valuesInRange(int16_t const *values, int16_t const *expectedValues, unsigned int count, double errMargin)
{
    return valuesExact(values, expectedValues, count);
}

bool valuesInRange(uint32_t const *values, uint32_t const *expectedValues, unsigned int count, double errMargin)
{
    return valuesExact(values, expectedValues, count);
}

bool valuesInRange(uint64_t const *values, uint64_t const *expectedValues, unsigned int count, double errMargin)
{
    return valuesExact(values, expectedValues, count);
}

bool valuesInRange(int32_t const *values, int32_t const *expectedValues, unsigned int count, double errMargin)
{
    return valuesExact(values, expectedValues, count);
}

bool valuesInRange(int64_t const *values, int64_t const *expectedValues, unsigned int count, double errMargin)
{
    return valuesExact(values, expectedValues, count);
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLANDVTest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        MASK_TYPE m2 = m0.land(m1);
        m2.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LANDV[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDV");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        MASK_TYPE m2 = m0 & m1;
        m2.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LANDV[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDV(operator &)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        MASK_TYPE m2 = m0 && m1;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LANDV[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDV(operator &&)");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLANDSTest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s1 = DATA_SET::inputs::scalarA;
        MASK_TYPE m2 = m0.land(s1);
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LANDS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDS");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s1 = DATA_SET::inputs::scalarB;
        MASK_TYPE m2 = m0.land(s1);
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LANDS_B[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDS");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s1 = DATA_SET::inputs::scalarA;
        MASK_TYPE m2 = m0 & s1;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LANDS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDS(operator & RHS scalar)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s1 = DATA_SET::inputs::scalarB;
        MASK_TYPE m2 = m0 & s1;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LANDS_B[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDS(operator & RHS scalar)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s1 = DATA_SET::inputs::scalarA;
        MASK_TYPE m2 = m0 && s1;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LANDS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDS(operator && RHS scalar)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s1 = DATA_SET::inputs::scalarB;
        MASK_TYPE m2 = m0 && s1;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LANDS_B[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDS(operator && RHS scalar)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s1 = DATA_SET::inputs::scalarA;
        MASK_TYPE m2 = s1 & m0;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LANDS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDS(operator & LHS scalar)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s1 = DATA_SET::inputs::scalarB;
        MASK_TYPE m2 = s1 & m0;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LANDS_B[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDS(operator & LHS scalar)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s1 = DATA_SET::inputs::scalarA;
        MASK_TYPE m2 = s1 && m0;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LANDS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDS(operator && LHS scalar)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s1 = DATA_SET::inputs::scalarB;
        MASK_TYPE m2 = s1 && m0;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LANDS_B[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDS(operator && LHS scalar)");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLANDVATest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        m0.landa(m1);
        m0.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LANDV[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDVA");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        m0 &= m1;
        m0.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LANDV[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDVA(operator &=)");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLANDSATest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s1 = DATA_SET::inputs::scalarA;
        m0.landa(s1);
        m0.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LANDS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDSA");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s1 = DATA_SET::inputs::scalarA;
        m0 &= s1;
        m0.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LANDS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LANDSA(operator &=)");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLORVTest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        MASK_TYPE m2 = m0.lor(m1);
        m2.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LORV[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LORV");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        MASK_TYPE m2 = m0 | m1;
        m2.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LORV[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LORV(operator |)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        MASK_TYPE m2 = m0 || m1;
        m2.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LORV[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LORV(operator ||)");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLORSTest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s0 = DATA_SET::inputs::scalarA;
        MASK_TYPE m2 = m0.lor(s0);
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LORS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LORS");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s0 = DATA_SET::inputs::scalarB;
        MASK_TYPE m2 = m0.lor(s0);
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LORS_B[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LORS");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s0 = DATA_SET::inputs::scalarA;
        MASK_TYPE m2 = m0 | s0;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LORS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LORS (operator| RHS scalar)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s0 = DATA_SET::inputs::scalarB;
        MASK_TYPE m2 = m0 | s0;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LORS_B[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LORS (operator| RHS scalar)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s0 = DATA_SET::inputs::scalarA;
        MASK_TYPE m2 = m0 || s0;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LORS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LORS (operator|| RHS scalar)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s0 = DATA_SET::inputs::scalarB;
        MASK_TYPE m2 = m0 || s0;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LORS_B[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LORS (operator|| RHS scalar)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s0 = DATA_SET::inputs::scalarA;
        MASK_TYPE m2 = s0 | m0;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LORS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LORS (operator| LHS scalar)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s0 = DATA_SET::inputs::scalarB;
        MASK_TYPE m2 = s0 | m0;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LORS_B[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LORS (operator| LHS scalar)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s0 = DATA_SET::inputs::scalarA;
        MASK_TYPE m2 = s0 || m0;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LORS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LORS (operator|| LHS scalar)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s0 = DATA_SET::inputs::scalarB;
        MASK_TYPE m2 = s0 || m0;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LORS_B[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LORS (operator|| LHS scalar)");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLORVATest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        m0.lora(m1);
        m0.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LORV[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LORVA");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        m0 |= m1;
        m0.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LORV[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LORVA(operator |=)");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLORSATest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s1 = DATA_SET::inputs::scalarA;
        m0.lora(s1);
        m0.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LORS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LORSA");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s1 = DATA_SET::inputs::scalarA;
        m0 |= s1;
        m0.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LORS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LORSA(operator |=)");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLXORVTest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        MASK_TYPE m2 = m0.lxor(m1);
        m2.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LXORV[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LXORV");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        MASK_TYPE m2 = m0 ^ m1;
        m2.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LXORV[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LXORV(operator^)");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLXORSTest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s0 = DATA_SET::inputs::scalarA;
        MASK_TYPE m2 = m0.lxor(s0);
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LXORS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LXORS");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s0 = DATA_SET::inputs::scalarB;
        MASK_TYPE m2 = m0.lxor(s0);
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LXORS_B[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LXORS");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s0 = DATA_SET::inputs::scalarA;
        MASK_TYPE m2 = m0 ^ s0;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LXORS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LXORS (operator^ RHS scalar)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s0 = DATA_SET::inputs::scalarB;
        MASK_TYPE m2 = m0 ^ s0;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LXORS_B[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LXORS (operator^ RHS scalar)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s0 = DATA_SET::inputs::scalarA;
        MASK_TYPE m2 = s0 ^ m0;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LXORS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LXORS (operator^ LHS scalar)");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s0 = DATA_SET::inputs::scalarB;
        MASK_TYPE m2 = s0 ^ m0;
        m2.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LXORS_B[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LXORS (operator^ LHS scalar)");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLXORVATest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        m0.lxora(m1);
        m0.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LXORV[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LXORVA");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        MASK_TYPE m1(DATA_SET::inputs::maskB);
        m0 ^= m1;
        m0.store(values);
        bool exact = true;
        for(int i = 0; i < VEC_LEN; i++) {
            if(values[i] != DATA_SET::outputs::LXORV[i]) {
                exact = false; 
                break;
            }
        }
        CHECK_CONDITION(exact, "LXORVA(operator^=)");
    }
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericLXORSATest()
{
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s1 = DATA_SET::inputs::scalarA;
        m0.lxora(s1);
        m0.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LXORS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LXORSA");
    }
    {
        bool values[VEC_LEN];
        MASK_TYPE m0(DATA_SET::inputs::maskA);
        bool s1 = DATA_SET::inputs::scalarA;
        m0 ^= s1;
        m0.store(values);
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            if (values[i] != DATA_SET::outputs::LXORS_A[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "LXORSA(operator^=)");
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
void genericEXTRACTTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value;
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            value = vec0.extract(i);
            if (value != DATA_SET::inputs::inputA[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "EXTRACT");
    }
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value;
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            value = vec0[i];
            if (value != DATA_SET::inputs::inputA[i]) {
                exact = false;
                break;
            }
        }
        CHECK_CONDITION(exact, "EXTRACT(operator[])");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericINSERTTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0;
        
        bool exact = true;
        for (int i = 0; i < VEC_LEN; i++) {
            vec0.insert(i, DATA_SET::inputs::inputA[i]);
        }
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "INSERT");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0;

        bool exact = true;
        for (uint32_t i = 0; i < VEC_LEN; i++) {
            vec0[i] = DATA_SET::inputs::inputA[i];
        }
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "INSERT(operator[] =)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void generiASSIGNVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0.assign(vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::inputs::inputB, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "ASSIGNV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0 = vec1;
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::inputs::inputB, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "ASSIGNV(operator=)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void generiMASSIGNVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        SCALAR_TYPE expected[VEC_LEN];
        for (int i = 0; i < VEC_LEN; i++) {
            expected[i] = DATA_SET::inputs::maskA[i] ? 
                          DATA_SET::inputs::inputB[i] : 
                          DATA_SET::inputs::inputA[i];
        }
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        vec0.assign(mask, vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, expected, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MASSIGNV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        SCALAR_TYPE expected[VEC_LEN];
        for (int i = 0; i < VEC_LEN; i++) {
            expected[i] = DATA_SET::inputs::maskA[i] ?
                DATA_SET::inputs::inputB[i] :
                DATA_SET::inputs::inputA[i];
        }
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
#if !defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        vec0[mask] = vec1;
#else
        vec0(mask) = vec1;
#endif
        vec0.store(values);
        bool inRange = valuesInRange(values, expected, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MASSIGNV(vec[mask] = ) ");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void generiASSIGNSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        SCALAR_TYPE expected[VEC_LEN];
        for (int i = 0; i < VEC_LEN; i++) expected[i] = DATA_SET::inputs::scalarA;
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        vec0.assign(DATA_SET::inputs::scalarA);
        vec0.store(values);
        bool inRange = valuesInRange(values, expected, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "ASSIGNS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        SCALAR_TYPE expected[VEC_LEN];
        for (int i = 0; i < VEC_LEN; i++) expected[i] = DATA_SET::inputs::scalarA;
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        vec0 = DATA_SET::inputs::scalarA;
        vec0.store(values);
        bool inRange = valuesInRange(values, expected, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "ASSIGNS(operator=)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void generiMASSIGNSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        SCALAR_TYPE expected[VEC_LEN];
        for (int i = 0; i < VEC_LEN; i++) {
            expected[i] = DATA_SET::inputs::maskA[i] ?  
                          DATA_SET::inputs::scalarA :
                          DATA_SET::inputs::inputA[i];
        }
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        vec0.assign(mask, DATA_SET::inputs::scalarA);
        vec0.store(values);
        bool inRange = valuesInRange(values, expected, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MASSIGNS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        SCALAR_TYPE expected[VEC_LEN];
        for (int i = 0; i < VEC_LEN; i++) {
            expected[i] = DATA_SET::inputs::maskA[i] ?
                          DATA_SET::inputs::scalarA :
                          DATA_SET::inputs::inputA[i];
        }
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        vec0.assign(mask, DATA_SET::inputs::scalarA);
#if !defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        vec0[mask] = DATA_SET::inputs::scalarA;
#else
        vec0(mask) = DATA_SET::inputs::scalarA;
#endif
        vec0.store(values);
        bool inRange = valuesInRange(values, expected, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MASSIGNS");
    }
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
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "ADDV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0 + vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "ADDV(operator+)");
    }   
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMADDVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.add(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MADDV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MADDV");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericADDSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.add(DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "ADDS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0 + DATA_SET::inputs::scalarA;
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "ADDS(operator+ RHS scalar");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = DATA_SET::inputs::scalarA + vec0;
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "ADDS(operator+ LHS scalar");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMADDSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.add(mask, DATA_SET::inputs::scalarA);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MADDS, VEC_LEN, SCALAR_TYPE(0.01f));
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
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "ADDVA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0 += vec1;
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "ADDVA(operator+=)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMADDVATest()
    {
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        vec0.adda(mask, vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MADDV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MADDVA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        vec0(mask) += vec1;
#else
        vec0[mask] += vec1;
#endif
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MADDV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MADDVA(vec[mask] +=)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericADDSATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        vec0.adda(DATA_SET::inputs::scalarA);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "ADDSA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        vec0 += DATA_SET::inputs::scalarA;
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "ADDSA(operator+=)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMADDSATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        vec0.adda(mask, DATA_SET::inputs::scalarA);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MADDS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MADDSA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
#if !defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        vec0[mask] += DATA_SET::inputs::scalarA;
#else
        vec0(mask) += DATA_SET::inputs::scalarA;
#endif
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MADDS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MADDSA(vec[mask] +=)");
    }
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
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFINC, VEC_LEN, SCALAR_TYPE(0.01f));
        bool inRange1 = valuesInRange(values1, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange0 && inRange1, "POSTINC");
    }
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0++;
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFINC, VEC_LEN, SCALAR_TYPE(0.01f));
        bool inRange1 = valuesInRange(values1, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange0 = valuesInRange(values0, DATA_SET::outputs::MPOSTPREFINC, VEC_LEN, SCALAR_TYPE(0.01f));
    bool inRange1 = valuesInRange(values1, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
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
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFINC, VEC_LEN, SCALAR_TYPE(0.01f));
        bool inRange1 = valuesInRange(values1, DATA_SET::outputs::POSTPREFINC, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange0 && inRange1, "PREFINC");
    }
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = ++vec0;
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFINC, VEC_LEN, SCALAR_TYPE(0.01f));
        bool inRange1 = valuesInRange(values1, DATA_SET::outputs::POSTPREFINC, VEC_LEN, SCALAR_TYPE(0.01f));
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
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::MPOSTPREFINC, VEC_LEN, SCALAR_TYPE(0.01f));
        bool inRange1 = valuesInRange(values1, DATA_SET::outputs::MPOSTPREFINC, VEC_LEN, SCALAR_TYPE(0.01f));
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
        bool inRange = valuesInRange(values, DATA_SET::outputs::SUBV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "SUBV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0 - vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SUBV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MSUBV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSUBSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.sub(DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SUBS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "SUBS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0 - DATA_SET::inputs::scalarA;
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SUBS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "SUBS(operator- RHS scalar)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSUBSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.sub(mask, DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBS, VEC_LEN, SCALAR_TYPE(0.01f));
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
        bool inRange = valuesInRange(values, DATA_SET::outputs::SUBV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "SUBVA");
    }
    
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0 -= vec1;
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SUBV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "SUBVA(operator-=)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSUBVATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        vec0.suba(mask, vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MSUBVA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        vec0(mask) -= vec1;
#else
        vec0[mask] -= vec1;
#endif
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MSUBVA(vec[mask] -=)");
    }
}
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSUBSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.suba(DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::SUBS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "SUBSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSUBSATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        vec0.suba(mask, DATA_SET::inputs::scalarA);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MSUBSA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
#if !defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        vec0[mask] -= DATA_SET::inputs::scalarA;
#else
        vec0(mask) -= DATA_SET::inputs::scalarA;
#endif
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MSUBSA(vec[mask] -=)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSUBFROMVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    VEC_TYPE vec2 = vec0.subfrom(vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::SUBFROMV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBFROMV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MSUBFROMV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSUBFROMSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = vec0.subfrom(DATA_SET::inputs::scalarA);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SUBFROMS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "SUBFROMS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = DATA_SET::inputs::scalarA - vec0;
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SUBFROMS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "SUBFROMS(operator- LHS scalar)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSUBFROMSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.subfrom(mask, DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBFROMS, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::SUBFROMV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBFROMV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MSUBFROMVA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSUBFROMSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.subfroma(DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::SUBFROMS, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBFROMS, VEC_LEN, SCALAR_TYPE(0.01f));
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
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFDEC, VEC_LEN, SCALAR_TYPE(0.01f));
        bool inRange1 = valuesInRange(values1, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange0 && inRange1, "POSTDEC");
    }
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0--;
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFDEC, VEC_LEN, SCALAR_TYPE(0.01f));
        bool inRange1 = valuesInRange(values1, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange0 = valuesInRange(values0, DATA_SET::outputs::MPOSTPREFDEC, VEC_LEN, SCALAR_TYPE(0.01f));
    bool inRange1 = valuesInRange(values1, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
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
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFDEC, VEC_LEN, SCALAR_TYPE(0.01f));
        bool inRange1 = valuesInRange(values1, DATA_SET::outputs::POSTPREFDEC, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange0 && inRange1, "PREFDEC");
    }
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = --vec0;
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFDEC, VEC_LEN, SCALAR_TYPE(0.01f));
        bool inRange1 = valuesInRange(values1, DATA_SET::outputs::POSTPREFDEC, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange0 = valuesInRange(values0, DATA_SET::outputs::MPOSTPREFDEC, VEC_LEN, SCALAR_TYPE(0.01f));
    bool inRange1 = valuesInRange(values1, DATA_SET::outputs::MPOSTPREFDEC, VEC_LEN, SCALAR_TYPE(0.01f));
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
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MULV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0 * vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMULV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MMULV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMULSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.mul(DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MULS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0 * DATA_SET::inputs::scalarA;
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MULS(operator* RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = DATA_SET::inputs::scalarA * vec0;
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MULS(operator* LHS scalar)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMULSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.mul(mask, DATA_SET::inputs::scalarA);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMULS, VEC_LEN, SCALAR_TYPE(0.01f));
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
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MULVA");
    }
    
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0 *= vec1;
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MULVA(operator*)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMULVATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        vec0.mula(mask, vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MMULV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MMULVA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        vec0(mask) *= vec1;
#else
        vec0[mask] *= vec1;
#endif
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MMULV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MMULVA(vec[mask] /=)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMULSATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        vec0.mula(DATA_SET::inputs::scalarA);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MULSA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        vec0 *= DATA_SET::inputs::scalarA;
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MULSA(operator*=)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMULSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.mula(mask, DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMULS, VEC_LEN, SCALAR_TYPE(0.01f));
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
        bool inRange = valuesInRange(values, DATA_SET::outputs::DIVV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "DIVV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0 / vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::DIVV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MDIVV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MDIVV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericDIVSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.div(DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::DIVS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "DIVS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0 / DATA_SET::inputs::scalarA;
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::DIVS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "DIVS(operator/ RHS scalar)");
    }
}

    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMDIVSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.div(mask, DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MDIVS, VEC_LEN, SCALAR_TYPE(0.01f));
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
        bool inRange = valuesInRange(values, DATA_SET::outputs::DIVV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "DIVVA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0 /= vec1;
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::DIVV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "DIVVA(operator/)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMDIVVATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        vec0.diva(mask, vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MDIVV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MDIVVA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        vec0(mask) /= vec1;
#else
        vec0[mask] /= vec1;
#endif
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MDIVV, VEC_LEN, SCALAR_TYPE(0.01f));

#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        CHECK_CONDITION(inRange, "MDIVVA(vec(mask) /=)");
#else
        CHECK_CONDITION(inRange, "MDIVVA(vec[mask] /=)");
#endif
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericDIVSATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        vec0.diva(DATA_SET::inputs::scalarA);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::DIVS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "DIVSA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        vec0 /= DATA_SET::inputs::scalarA;
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::DIVS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "DIVSA(operator/=)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMDIVSATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        vec0.diva(mask, DATA_SET::inputs::scalarA);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MDIVS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MDIVSA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
#if !defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        vec0[mask] /= DATA_SET::inputs::scalarA;
#else
        vec0(mask) /= DATA_SET::inputs::scalarA;
#endif
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MDIVS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MDIVSA(vec[mask] \\=)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRCPTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.rcp();
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::RCP, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MRCP, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MRCP");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRCPSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.rcp(DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::RCPS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "RCPS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.rcp(DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::RCPS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "RCPS(operator/ LHS scalar)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRCPSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.rcp(mask, DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MRCPS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MRCPS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRCPATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.rcpa();
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::RCP, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MRCP, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MRCPA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRCPSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.rcpa(DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::RCPS, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MRCPS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MRCPSA");
}
    
template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPEQVTest()
{
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(true);
        mask = vec0 == vec1;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPEQV, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPEQV(operator==)");
    }
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPEQSTest()
{
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = vec0.cmpeq(DATA_SET::inputs::scalarA);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPEQS, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPEQS");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = vec0 == DATA_SET::inputs::scalarA;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPEQS, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPEQS(operator== RHS scalar)");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = DATA_SET::inputs::scalarA == vec0;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPEQS, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPEQS(operator== LHS scalar)");
    }
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPNEVTest()
{
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(true);
        mask = vec0 != vec1;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPNEV, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPNEV(operator!=)");
    }
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPNESTest()
{
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = vec0.cmpne(DATA_SET::inputs::scalarA);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPNES, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPNES");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = vec0 != DATA_SET::inputs::scalarA;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPNES, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPNES(operator!= RHS scalar)");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = DATA_SET::inputs::scalarA != vec0;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPNES, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPNES(operator!= LHS scalar)");
    }
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPGTVTest()
{
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(true);
        mask = vec0 > vec1;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPGTV, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPGTV(operator>)");
    }

}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPGTSTest()
{
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = vec0.cmpgt(DATA_SET::inputs::scalarA);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPGTS, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPGTS");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = vec0 > DATA_SET::inputs::scalarA;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPGTS, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPGTS(operator> RHS scalar)");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = DATA_SET::inputs::scalarA > vec0;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPLTS, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPGTS(operator> LHS scalar)");
    }
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPLTVTest()
{
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(true);
        mask = vec0 < vec1;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPLTV, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPLTV(operator<)");
    }
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPLTSTest()
{
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = vec0.cmplt(DATA_SET::inputs::scalarA);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPLTS, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPLTS");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = vec0 < DATA_SET::inputs::scalarA;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPLTS, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPLTS(operator< RHS scalar)");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = DATA_SET::inputs::scalarA < vec0;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPGTS, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPLTS(operator< LHS scalar)");
    }
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPGEVTest()
{
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(true);
        mask = vec0 >= vec1;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPGEV, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPGEV(operator>=)");
    }
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPGESTest()
{
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = vec0.cmpge(DATA_SET::inputs::scalarA);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPGES, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPGES");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = vec0 >= DATA_SET::inputs::scalarA;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPGES, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPGES(operator>= RHS scalar)");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = DATA_SET::inputs::scalarA >= vec0;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPLES, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPGES(operator>= LHS scalar)");
    }
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPLEVTest()
{
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(true);
        mask = vec0 <= vec1;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPLEV, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPLEV(operator<=)");
    }
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPLESTest()
{
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = vec0.cmple(DATA_SET::inputs::scalarA);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPLES, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPLES");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = vec0 <= DATA_SET::inputs::scalarA;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPLES, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPLES(operator<= LHS scalar)");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = DATA_SET::inputs::scalarA <= vec0;
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPGES, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPLES(operator<= RHS scalar)");
    }
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
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0.band(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BANDV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BANDV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0 & vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BANDV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BANDV(operator&)");
    }
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBANDV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MBANDV");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBANDSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = vec0.band(DATA_SET::inputs::scalarA);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BANDS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BANDS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = vec0 & DATA_SET::inputs::scalarA;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BANDS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BANDS(operator & RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = DATA_SET::inputs::scalarA & vec0;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BANDS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BANDS(operator & LHS scalar)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBANDSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.band(mask, DATA_SET::inputs::scalarA);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBANDS, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::BANDV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "BANDVA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBANDVATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        vec0.banda(mask, vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBANDV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MBANDVA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        vec0(mask) &= vec1;
#else
        vec0[mask] &= vec1;
#endif
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBANDV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MBANDVA(vec[mask] &=)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBANDSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.banda(DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::BANDS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "BANDSA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBANDSATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        vec0.banda(mask, DATA_SET::inputs::scalarA);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBANDS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MBANDSA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
#if !defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        vec0[mask] &= DATA_SET::inputs::scalarA;
#else
        vec0(mask) &= DATA_SET::inputs::scalarA;
#endif
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBANDS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MBANDSA(vec[mask] &=)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBORVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    VEC_TYPE vec2 = vec0.bor(vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::BORV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBORV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MBORV");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBORSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = vec0.bor(DATA_SET::inputs::scalarA);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BORS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BORS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = vec0 | DATA_SET::inputs::scalarA;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BORS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BORS(operator| RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = DATA_SET::inputs::scalarA | vec0;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BORS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BORS(operator| LHS scalar)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBORSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.bor(mask, DATA_SET::inputs::scalarA);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBORS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MBORS");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBORVATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0.bora(vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BORV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BORVA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0 |= vec1;
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BORV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BORVA(operator|=)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBORVATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        vec0.bora(mask, vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBORV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MBORVA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        vec0(mask) |= vec1;
#else
        vec0[mask] |= vec1;
#endif
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBORV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MBORVA(vec[mask] |=)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBORSATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        vec0.bora(DATA_SET::inputs::scalarA);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BORS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BORSA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        vec0 |= DATA_SET::inputs::scalarA;
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BORS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BORSA(operator|=)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBORSATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        vec0.bora(mask, DATA_SET::inputs::scalarA);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBORS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MBORSA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
#if !defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        vec0[mask] |= DATA_SET::inputs::scalarA;
#else
        vec0(mask) |= DATA_SET::inputs::scalarA;
#endif
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBORS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MBORSA(vec[mask]|=");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBXORVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0.bxor(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BXORV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BXORV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0 ^ vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BXORV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BXORV(operator^");
    }
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBXORV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MBXORV");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBXORSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = vec0.bxor(DATA_SET::inputs::scalarA);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BXORS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BXORS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = vec0 ^ DATA_SET::inputs::scalarA;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BXORS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BXORS(operator^ RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = DATA_SET::inputs::scalarA ^ vec0;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BXORS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BXORS(operator ^ LHS scalar)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBXORSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.bxor(mask, DATA_SET::inputs::scalarA);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBXORS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MBXORS");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBXORVATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0.bxora(vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BXORV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BXORVA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        vec0 ^= vec1;
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BXORV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BXORVA(operator^=)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBXORVATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        vec0.bxora(mask, vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBXORV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MBXORVA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
#if !defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        vec0[mask] ^= vec1;
#else
        vec0(mask) ^= vec1;
#endif
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBXORV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MBXORVA(vec[mask] ^=)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBXORSATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        vec0.bxora(DATA_SET::inputs::scalarA);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BXORS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BXORSA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        vec0 ^= DATA_SET::inputs::scalarA;
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BXORS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BXORSA(operator^=)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBXORSATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        vec0.bxora(mask, DATA_SET::inputs::scalarA);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBXORS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MBXORSA");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
#if !defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        vec0[mask] ^= DATA_SET::inputs::scalarA;
#else
        vec0(mask) ^= DATA_SET::inputs::scalarA;
#endif
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBXORS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MBXORSA(vec[mask] ^=)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBNOTTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.bnot();
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BNOT, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BNOT");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = ~vec0;
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BNOT, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "BNOT(operator!)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBNOTTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.bnot(mask);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBNOT, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MBNOT");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericBNOTATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.bnota();
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::BNOT, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MBNOT, VEC_LEN, SCALAR_TYPE(0.01f));
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
        // assign
        // SWIZZLE  - Swizzle (reorder/permute) vector elements
        // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign
 
        //(Reduction to scalar operations)
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHADDTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    SCALAR_TYPE value = vec0.hadd();
    bool inRange = valueInRange(value, DATA_SET::outputs::HADD[VEC_LEN-1], SCALAR_TYPE(SCALAR_TYPE(0.01f)));
    CHECK_CONDITION(inRange, "HADD");
}
        // MHADD - Masked add elements of a vector (horizontal add)
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHMULTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    SCALAR_TYPE value = vec0.hmul();
    bool inRange = valueInRange(value, DATA_SET::outputs::HMUL[VEC_LEN-1], SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "HMUL");
}
        // MHMUL - Masked multiply elements of a vector (horizontal mul)

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHBANDTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    SCALAR_TYPE value = vec0.hband();
    bool inRange = valueInRange(value, DATA_SET::outputs::HBAND[VEC_LEN-1], SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "HBAND");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMHBANDTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    SCALAR_TYPE value = vec0.hband(mask0);
    bool inRange = valueInRange(value, DATA_SET::outputs::MHBAND[VEC_LEN-1], SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MHBAND");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHBANDSTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    SCALAR_TYPE value = vec0.hband(DATA_SET::inputs::scalarA);
    bool inRange = valueInRange(value, DATA_SET::outputs::HBANDS[VEC_LEN-1], SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "HBANDS");
}
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMHBANDSTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    SCALAR_TYPE value = vec0.hband(mask0, DATA_SET::inputs::scalarA);
    bool inRange = valueInRange(value, DATA_SET::outputs::MHBANDS[VEC_LEN-1], SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MHBANDS");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHBORTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    SCALAR_TYPE value = vec0.hbor();
    bool inRange = valueInRange(value, DATA_SET::outputs::HBOR[VEC_LEN-1], SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "HBOR");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMHBORTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    SCALAR_TYPE value = vec0.hbor(mask0);
    bool inRange = valueInRange(value, DATA_SET::outputs::MHBOR[VEC_LEN-1], SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MHBOR");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHBORSTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    SCALAR_TYPE value = vec0.hbor(DATA_SET::inputs::scalarA);
    bool inRange = valueInRange(value, DATA_SET::outputs::HBORS[VEC_LEN-1], SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "HBORS");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMHBORSTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    SCALAR_TYPE value = vec0.hbor(mask0, DATA_SET::inputs::scalarA);
    bool inRange = valueInRange(value, DATA_SET::outputs::MHBORS[VEC_LEN-1], SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MHBORS");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHBXORTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    SCALAR_TYPE value = vec0.hbxor();
    bool inRange = valueInRange(value, DATA_SET::outputs::HBXOR[VEC_LEN-1], SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "HBXOR");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMHBXORTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    SCALAR_TYPE value = vec0.hbxor(mask0);
    bool inRange = valueInRange(value, DATA_SET::outputs::MHBXOR[VEC_LEN-1], SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MHBXOR");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHBXORSTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    SCALAR_TYPE value = vec0.hbxor(DATA_SET::inputs::scalarA);
    bool inRange = valueInRange(value, DATA_SET::outputs::HBXORS[VEC_LEN-1], SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "HBXORS");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMHBXORSTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    SCALAR_TYPE value = vec0.hbxor(mask0, DATA_SET::inputs::scalarA);
    bool inRange = valueInRange(value, DATA_SET::outputs::MHBXORS[VEC_LEN-1], SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MHBXORS");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericFMULADDVTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1(DATA_SET::inputs::inputB);
    VEC_TYPE vec2(DATA_SET::inputs::inputC);
    VEC_TYPE vec3 = vec0.fmuladd(vec1, vec2);
    vec3.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::FMULADDV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MFMULADDV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::FMULSUBV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MFMULSUBV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::FADDMULV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MFADDMULV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::FSUBMULV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MFSUBMULV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MAXV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMAXV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MMAXV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMAXSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.max(DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MAXS, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMAXS, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MAXV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMAXV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MMAXVA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMAXSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.maxa(DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MAXS, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMAXS, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MINV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMINV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MMINV");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMINSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.min(DATA_SET::inputs::scalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MINS, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMINS, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MINV, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMINV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MMINVA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMINSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.mina(DATA_SET::inputs::scalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MINS, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MMINS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MMINSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHMAXTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    SCALAR_TYPE value = vec0.hmax();
    SCALAR_TYPE expected = DATA_SET::outputs::HMAX[VEC_LEN-1];
    bool inRange = valueInRange(value, expected, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "HMAX");
}

// MHMAX
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHMINTest()
{
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    SCALAR_TYPE value = vec0.hmin();
    SCALAR_TYPE expected = DATA_SET::outputs::HMIN[VEC_LEN-1];
    bool inRange = valueInRange(value, expected, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "HMIN");
}

// MHMIN

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericLSHVTest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
    VEC_TYPE vec2 = vec0.lsh(vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::LSHV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "LSHV");
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMLSHVTest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.lsh(mask0, vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MLSHV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MLSHV");
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericLSHSTest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.lsh(DATA_SET::inputs::inputShiftScalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::LSHS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "LSHS");
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMLSHSTest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.lsh(mask0, DATA_SET::inputs::inputShiftScalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MLSHS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MLSHS");
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericLSHVATest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
    vec0.lsha(vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::LSHV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "LSHVA");
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMLSHVATest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    vec0.lsha(mask0, vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MLSHV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MLSHVA");
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericLSHSATest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.lsha(DATA_SET::inputs::inputShiftScalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::LSHS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "LSHSA");
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMLSHSATest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    vec0.lsha(mask0, DATA_SET::inputs::inputShiftScalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MLSHS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MLSHSA");
} 

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRSHVTest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
    VEC_TYPE vec2 = vec0.rsh(vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RSHV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "RSHV");
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRSHVTest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.rsh(mask0, vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRSHV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MRSHV");
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRSHSTest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.rsh(DATA_SET::inputs::inputShiftScalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RSHS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "RSHS");
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRSHSTest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.rsh(mask0, DATA_SET::inputs::inputShiftScalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRSHS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MRSHS");
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRSHVATest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
    vec0.rsha(vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RSHV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "RSHVA");
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRSHVATest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    vec0.rsha(mask0, vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRSHV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MRSHVA");
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRSHSATest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.rsha(DATA_SET::inputs::inputShiftScalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RSHS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "RSHSA");
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRSHSATest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    vec0.rsha(mask0, DATA_SET::inputs::inputShiftScalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRSHS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MRSHSA");
} 

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericROLVTest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
    VEC_TYPE vec2 = vec0.rol(vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::ROLV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "ROLV");
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMROLVTest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.rol(mask0, vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MROLV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MROLV");
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericROLSTest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.rol(DATA_SET::inputs::inputShiftScalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::ROLS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "ROLS");
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMROLSTest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.rol(mask0, DATA_SET::inputs::inputShiftScalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MROLS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MROLS");
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericROLVATest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
    vec0.rola(vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::ROLV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "ROLVA");
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMROLVATest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    vec0.rola(mask0, vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MROLV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MROLVA");
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericROLSATest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.rola(DATA_SET::inputs::inputShiftScalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::ROLS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "ROLSA");
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMROLSATest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    vec0.rola(mask0, DATA_SET::inputs::inputShiftScalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MROLS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MROLSA");
} 

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRORVTest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
    VEC_TYPE vec2 = vec0.ror(vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RORV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "RORV");
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRORVTest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    VEC_TYPE vec2 = vec0.ror(mask0, vec1);
    vec2.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRORV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MRORV");
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRORSTest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.ror(DATA_SET::inputs::inputShiftScalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RORS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "RORS");
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRORSTest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.ror(mask0, DATA_SET::inputs::inputShiftScalarA);
    vec1.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRORS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MRORS");
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRORVATest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
    vec0.rora(vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RORV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "RORVA");
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRORVATest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    vec0.rora(mask0, vec1);
    vec0.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRORV, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MRORVA");
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRORSATest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.rora(DATA_SET::inputs::inputShiftScalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RORS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "RORSA");
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRORSATest() 
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask0(DATA_SET::inputs::maskA);
    vec0.rora(mask0, DATA_SET::inputs::inputShiftScalarA);
    vec0.store(values);
    bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRORS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MRORSA");
} 
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericNEGTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.neg();
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::NEG, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "NEG");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 =  -vec0;
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::NEG, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MNEG, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MNEG");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericNEGATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.nega();
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::NEG, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MNEG, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MNEGA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericABSTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.abs();
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::ABS, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MABS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MABS");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericABSATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.absa();
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::ABS, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MABS, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MABSA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSQRTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.sqr();
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::SQR, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSQR, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MSQR");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSQRATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.sqra();
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::SQR, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSQR, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MSQRA");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSQRTTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.abs();
    VEC_TYPE vec2 = vec1.sqrt();
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::SQRT, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "SQRT");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSQRTTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.abs(mask);
    VEC_TYPE vec2 = vec1.sqrt(mask);
    vec2.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSQRT, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MSQRT");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSQRTATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    vec0.absa();
    vec0.sqrta();
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::SQRT, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "SQRTA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSQRTATest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    vec0.absa(mask);
    vec0.sqrta(mask);
    vec0.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MSQRT, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MSQRTA");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericROUNDTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_TYPE vec1 = vec0.round();
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::ROUND, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "ROUND");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMROUNDTest()
{
    SCALAR_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    MASK_TYPE mask(DATA_SET::inputs::maskA);
    VEC_TYPE vec1 = vec0.round(mask); 
    vec1.store(values);
    bool inRange = valuesInRange(values, DATA_SET::outputs::MROUND, VEC_LEN, SCALAR_TYPE(0.01f));
    CHECK_CONDITION(inRange, "MROUND");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename VEC_INT_TYPE, typename SCALAR_INT_TYPE, int VEC_LEN, typename DATA_SET>
void genericTRUNCTest()
{
    SCALAR_INT_TYPE values[VEC_LEN];
    VEC_TYPE vec0(DATA_SET::inputs::inputA);
    VEC_INT_TYPE vec1 = vec0.trunc();
    vec1.store(values);
    bool exact = valuesExact(values, DATA_SET::outputs::TRUNC, VEC_LEN);
    CHECK_CONDITION(exact, "TRUNC");
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename VEC_INT_TYPE, typename SCALAR_INT_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMTRUNCTest()
{
    SCALAR_INT_TYPE values[VEC_LEN];
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::FLOOR, VEC_LEN, SCALAR_TYPE(0.01f));
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
    bool inRange = valuesInRange(values, DATA_SET::outputs::MFLOOR, VEC_LEN, SCALAR_TYPE(0.01f));
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

template<typename UINT_VEC_TYPE, typename INT_VEC_TYPE, typename INT_SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericUTOITest()
{
    INT_SCALAR_TYPE intValues[VEC_LEN];
    UINT_VEC_TYPE vec0(DATA_SET::inputs::inputA);
    INT_VEC_TYPE vec1;
    vec1 = INT_VEC_TYPE(vec0);
    vec1.store(intValues);
    bool inRange = valuesInRange(intValues, DATA_SET::outputs::UTOI, VEC_LEN, 0.1f);
    CHECK_CONDITION(inRange, "UTOI");
}

template<typename UINT_VEC_TYPE, typename FLOAT_VEC_TYPE, typename FLOAT_SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericUTOFTest()
{
    FLOAT_SCALAR_TYPE floatValues[VEC_LEN];
    UINT_VEC_TYPE vec0(DATA_SET::inputs::inputA);
    FLOAT_VEC_TYPE vec1;
    vec1 = FLOAT_VEC_TYPE(vec0);
    vec1.store(floatValues);
    bool inRange = valuesInRange(floatValues, DATA_SET::outputs::UTOF, VEC_LEN, 0.1f);
    CHECK_CONDITION(inRange, "UTOF");
}

template<typename INT_VEC_TYPE, typename UINT_VEC_TYPE, typename UINT_SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericITOUTest()
{
    UINT_SCALAR_TYPE uintValues[VEC_LEN];
    INT_VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1;
    vec1 = UINT_VEC_TYPE(vec0);
    vec1.store(uintValues);
    bool inRange = valuesInRange(uintValues, DATA_SET::outputs::ITOU, VEC_LEN, 0.1f);
    CHECK_CONDITION(inRange, "ITOU");
}

template<typename INT_VEC_TYPE, typename FLOAT_VEC_TYPE, typename FLOAT_SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericITOFTest()
{
    FLOAT_SCALAR_TYPE floatValues[VEC_LEN];
    INT_VEC_TYPE vec0(DATA_SET::inputs::inputA);
    FLOAT_VEC_TYPE vec1;
    vec1 = FLOAT_VEC_TYPE(vec0);
    vec1.store(floatValues);
    bool inRange = valuesInRange(floatValues, DATA_SET::outputs::ITOF, VEC_LEN, 0.1f);
    CHECK_CONDITION(inRange, "ITOF");
}

template<typename FLOAT_VEC_TYPE, typename UINT_VEC_TYPE, typename UINT_SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericFTOUTest()
{
    UINT_SCALAR_TYPE uintValues[VEC_LEN];
    FLOAT_VEC_TYPE vec0(DATA_SET::inputs::inputA);
    UINT_VEC_TYPE vec1;
    vec1 = UINT_VEC_TYPE(vec0);
    vec1.store(uintValues);
    bool inRange = valuesInRange(uintValues, DATA_SET::outputs::FTOU, VEC_LEN, 0.1f);
    CHECK_CONDITION(inRange, "FTOU");
}

template<typename FLOAT_VEC_TYPE, typename INT_VEC_TYPE, typename INT_SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericFTOITest()
{
    INT_SCALAR_TYPE intValues[VEC_LEN];
    FLOAT_VEC_TYPE vec0(DATA_SET::inputs::inputA);
    INT_VEC_TYPE vec1;
    vec1 = INT_VEC_TYPE(vec0);
    vec1.store(intValues);
    bool inRange = valuesInRange(intValues, DATA_SET::outputs::FTOI, VEC_LEN, 0.1f);
    CHECK_CONDITION(inRange, "FTOI");
}

template<typename VEC_TYPE_X, typename SCALAR_TYPE_X, typename VEC_TYPE_Y, typename SCALAR_TYPE_Y, int VEC_LEN, typename DATA_SET>
void genericPROMOTETest()
{
    SCALAR_TYPE_X input[VEC_LEN];
    SCALAR_TYPE_Y expected[VEC_LEN];
    SCALAR_TYPE_Y output[VEC_LEN];

    // Easier to generate output data here than to pre-compute it. There
    // are too many conversions to provide ready data sets.
    for (int i = 0; i < VEC_LEN; i++) {
        input[i] = SCALAR_TYPE_X(DATA_SET::inputs::inputA[i]);
        expected[i] = SCALAR_TYPE_Y(DATA_SET::inputs::inputA[i]);
    }

    VEC_TYPE_X vec0(input);
    VEC_TYPE_Y vec1;

    vec1 = VEC_TYPE_Y(vec0);
    vec1.store(output);
    bool inRange = valuesInRange(output, expected, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "PROMOTE");
}


template<typename VEC_TYPE_X, typename SCALAR_TYPE_X, typename VEC_TYPE_Y, typename SCALAR_TYPE_Y, int VEC_LEN, typename DATA_SET>
void genericDEMOTETest()
{
    SCALAR_TYPE_X input[VEC_LEN];
    SCALAR_TYPE_Y expected[VEC_LEN];
    SCALAR_TYPE_Y output[VEC_LEN];

    // Easier to generate output data here than to pre-compute it. There
    // are too many conversions to provide ready data sets.
    for (int i = 0; i < VEC_LEN; i++) {
        input[i] = SCALAR_TYPE_X(DATA_SET::inputs::inputA[i]);
        expected[i] = SCALAR_TYPE_Y(DATA_SET::inputs::inputA[i]);
    }

    VEC_TYPE_X vec0(input);
    VEC_TYPE_Y vec1;

    vec1 = VEC_TYPE_Y(vec0);
    vec1.store(output);
    bool inRange = valuesInRange(output, expected, VEC_LEN, 0.01f);
    CHECK_CONDITION(inRange, "DEMOTE");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericBaseInterfaceTest()
{   
    genericINSERTTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericEXTRACTTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    generiASSIGNVTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    generiMASSIGNVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    generiASSIGNSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    generiMASSIGNSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
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

    genericHBANDTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMHBANDTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericHBANDSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMHBANDSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericHBORTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMHBORTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericHBORSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMHBORSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericHBXORTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMHBXORTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericHBXORSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMHBXORSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
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

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericShiftRotateInterfaceTest()
{
    genericLSHVTest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMLSHVTest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericLSHSTest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMLSHSTest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericLSHVATest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMLSHVATest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericLSHSATest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMLSHSATest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericRSHVTest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMRSHVTest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericRSHSTest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMRSHSTest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericRSHVATest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMRSHVATest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericRSHSATest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMRSHSATest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericROLVTest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMROLVTest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericROLSTest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMROLSTest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericROLVATest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMROLVATest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericROLSATest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMROLSATest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericRORVTest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMRORVTest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericRORSTest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMRORSTest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericRORVATest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMRORVATest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericRORSATest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMRORSATest<VEC_TYPE, UINT_VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
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

template<typename VEC_TYPE, typename SCALAR_TYPE, typename VEC_INT_TYPE, typename SCALAR_INT_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericFloatInterfaceTest()
{
    genericROUNDTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMROUNDTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericTRUNCTest<VEC_TYPE, SCALAR_TYPE, VEC_INT_TYPE, SCALAR_INT_TYPE, VEC_LEN, DATA_SET>();
    genericMTRUNCTest<VEC_TYPE, SCALAR_TYPE, VEC_INT_TYPE, SCALAR_INT_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
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
    genericSQRTTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMSQRTTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericSQRTATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMSQRTATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    
    genericSINTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMSINTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCOSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMCOSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericTANTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMTANTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericCTANTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMCTANTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMaskTest() {
    genericLANDVTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLANDSTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLANDVATest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLANDSATest<MASK_TYPE, VEC_LEN, DATA_SET>();
    genericLORVTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLORSTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLORVATest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLORSATest<MASK_TYPE, VEC_LEN, DATA_SET>();
    genericLXORVTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLXORSTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLXORVATest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLXORSATest<MASK_TYPE, VEC_LEN, DATA_SET>();
    genericLNOTTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLNOTATest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericHLANDTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericHLORTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericHLXORTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
}

template<
        typename UINT_VEC_TYPE,
        typename UINT_SCALAR_TYPE,
        typename INT_VEC_TYPE,
        typename INT_SCALAR_TYPE,
        typename FLOAT_VEC_TYPE,
        typename FLOAT_SCALAR_TYPE,
        typename MASK_TYPE,
        int VEC_LEN,
        typename DATA_SET>
void genericUintTest() {
    genericBaseInterfaceTest<UINT_VEC_TYPE, UINT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericBitwiseInterfaceTest<UINT_VEC_TYPE, UINT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericGatherScatterInterfaceTest<UINT_VEC_TYPE, UINT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericShiftRotateInterfaceTest<UINT_VEC_TYPE, UINT_VEC_TYPE, UINT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericUTOITest<UINT_VEC_TYPE, INT_VEC_TYPE, INT_SCALAR_TYPE, VEC_LEN, DATA_SET> ();
    genericUTOFTest<UINT_VEC_TYPE, FLOAT_VEC_TYPE, FLOAT_SCALAR_TYPE, VEC_LEN, DATA_SET> ();
}

// Special version for scalars that don't have corresponding float scalar (e.g. uint8_t)
template<
    typename UINT_VEC_TYPE,
    typename UINT_SCALAR_TYPE,
    typename INT_VEC_TYPE,
    typename INT_SCALAR_TYPE,
    typename MASK_TYPE,
    int VEC_LEN,
    typename DATA_SET>
    void genericUintTest() {
    genericBaseInterfaceTest<UINT_VEC_TYPE, UINT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericBitwiseInterfaceTest<UINT_VEC_TYPE, UINT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericGatherScatterInterfaceTest<UINT_VEC_TYPE, UINT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericShiftRotateInterfaceTest<UINT_VEC_TYPE, UINT_VEC_TYPE, UINT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericUTOITest<UINT_VEC_TYPE, INT_VEC_TYPE, INT_SCALAR_TYPE, VEC_LEN, DATA_SET>();
}

template<
        typename INT_VEC_TYPE,
        typename INT_SCALAR_TYPE,
        typename UINT_VEC_TYPE,
        typename UINT_SCALAR_TYPE,
        typename FLOAT_VEC_TYPE,
        typename FLOAT_SCALAR_TYPE,
        typename MASK_TYPE,
        int VEC_LEN,
        typename DATA_SET>
void genericIntTest() {
    genericBaseInterfaceTest<INT_VEC_TYPE, INT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericBitwiseInterfaceTest<INT_VEC_TYPE, INT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericGatherScatterInterfaceTest<INT_VEC_TYPE, INT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericShiftRotateInterfaceTest<INT_VEC_TYPE, UINT_VEC_TYPE, INT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericSignInterfaceTest<INT_VEC_TYPE, INT_SCALAR_TYPE  , MASK_TYPE, VEC_LEN, DATA_SET>();
    genericITOUTest<INT_VEC_TYPE, UINT_VEC_TYPE, UINT_SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericITOFTest<INT_VEC_TYPE, FLOAT_VEC_TYPE, FLOAT_SCALAR_TYPE, VEC_LEN, DATA_SET>();
}

template<
    typename INT_VEC_TYPE,
    typename INT_SCALAR_TYPE,
    typename UINT_VEC_TYPE,
    typename UINT_SCALAR_TYPE,
    typename MASK_TYPE,
    int VEC_LEN,
    typename DATA_SET>
void genericIntTest() {
    genericBaseInterfaceTest<INT_VEC_TYPE, INT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericBitwiseInterfaceTest<INT_VEC_TYPE, INT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericGatherScatterInterfaceTest<INT_VEC_TYPE, INT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericShiftRotateInterfaceTest<INT_VEC_TYPE, UINT_VEC_TYPE, INT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericSignInterfaceTest<INT_VEC_TYPE, INT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericITOUTest<INT_VEC_TYPE, UINT_VEC_TYPE, UINT_SCALAR_TYPE, VEC_LEN, DATA_SET>();
}

template<
        typename FLOAT_VEC_TYPE, 
        typename FLOAT_SCALAR_TYPE,
        typename UINT_VEC_TYPE,
        typename UINT_SCALAR_TYPE,
        typename INT_VEC_TYPE,
        typename INT_SCALAR_TYPE,
        typename MASK_TYPE,
        int VEC_LEN,
        typename DATA_SET>
void genericFloatTest() {
    genericBaseInterfaceTest<FLOAT_VEC_TYPE, FLOAT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericGatherScatterInterfaceTest<FLOAT_VEC_TYPE, FLOAT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericSignInterfaceTest<FLOAT_VEC_TYPE, FLOAT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericFloatInterfaceTest<FLOAT_VEC_TYPE, FLOAT_SCALAR_TYPE, INT_VEC_TYPE, INT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericFTOUTest<FLOAT_VEC_TYPE, UINT_VEC_TYPE, UINT_SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericFTOITest<FLOAT_VEC_TYPE, INT_VEC_TYPE, INT_SCALAR_TYPE, VEC_LEN, DATA_SET>();
}

#endif
