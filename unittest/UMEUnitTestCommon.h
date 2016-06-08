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
#include "../UMESimd.h"
#include "UMEUnitTestDataSets8.h"
#include "UMEUnitTestDataSets16.h"
#include "UMEUnitTestDataSets32.h"
#include "UMEUnitTestDataSets64.h"

#include <random>

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
    if (expectedValue == 0.0)
    {
        return (errMargin >= value) & ((-errMargin) <= value);
    }
    else if (value > 0.0f)
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

// Randomization routines for random-generated tests.
template<typename SCALAR_TYPE>
SCALAR_TYPE randomValue(std::mt19937 & generator) {
    std::uniform_int_distribution<SCALAR_TYPE> dist(std::numeric_limits<SCALAR_TYPE>::min(), std::numeric_limits<SCALAR_TYPE>::max());
    return dist(generator);
}

template<>
uint8_t randomValue<uint8_t>(std::mt19937 & generator) {
    std::uniform_int_distribution<uint16_t> dist(0, 255);
    return uint8_t(dist(generator));
}

template<>
int8_t randomValue<int8_t>(std::mt19937 & generator) {
    int16_t min = int16_t(std::numeric_limits<int8_t>::min());
    int16_t max = int16_t(std::numeric_limits<int8_t>::max());

    std::uniform_int_distribution<int16_t> dist(min, max);
    int16_t t0 = dist(generator);
    return int8_t(t0);
}

template<>
float randomValue<float>(std::mt19937 & generator) {
    std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    return dist(generator);
}

template<>
double randomValue<double>(std::mt19937 & generator) {
    std::uniform_real_distribution<double> dist(std::numeric_limits<double>::min(), std::numeric_limits<double>::max());
    return dist(generator);
}

template<>
bool randomValue<bool>(std::mt19937 & generator) {
    std::uniform_int_distribution<int32_t> dist(std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
    int32_t t0 = dist(generator);
    return t0 > 0;
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

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN>
void genericLANDVTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    {
        bool values[VEC_LEN];
        bool outputs[VEC_LEN];
        SCALAR_TYPE inputA[VEC_LEN];
        SCALAR_TYPE inputB[VEC_LEN];

        for (int i = 0; i < VEC_LEN; i++)
        {
            inputA[i] = randomValue<SCALAR_TYPE>(gen);
            inputB[i] = randomValue<SCALAR_TYPE>(gen);
            outputs[i] = inputA[i] && inputB[i];
        }
        VEC_TYPE t0(inputA);
        VEC_TYPE t1(inputB);
        // Return type of LAND should be a mask, regardless of VEC_TYPE.
        auto t2 = t0.land(t1); 
        t2.store(values);
        CHECK_CONDITION(valuesExact(values, outputs, VEC_LEN), "LANDV gen");
    }
    {
        bool values[VEC_LEN];
        bool outputs[VEC_LEN];
        SCALAR_TYPE inputA[VEC_LEN];
        SCALAR_TYPE inputB[VEC_LEN];

        for (int i = 0; i < VEC_LEN; i++)
        {
            inputA[i] = randomValue<SCALAR_TYPE>(gen);
            inputB[i] = randomValue<SCALAR_TYPE>(gen);
            outputs[i] = inputA[i] && inputB[i];
        }
        VEC_TYPE t0(inputA);
        VEC_TYPE t1(inputB);
        // Return type of LAND should be a mask, regardless of VEC_TYPE.
        auto t2 = t0 && t1;
        t2.store(values);
        CHECK_CONDITION(valuesExact(values, outputs, VEC_LEN), "LANDV gen (operator &&)");
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

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN>
void genericLORVTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    {
        bool values[VEC_LEN];
        bool outputs[VEC_LEN];
        SCALAR_TYPE inputA[VEC_LEN];
        SCALAR_TYPE inputB[VEC_LEN];

        for (int i = 0; i < VEC_LEN; i++)
        {
            inputA[i] = randomValue<SCALAR_TYPE>(gen);
            inputB[i] = randomValue<SCALAR_TYPE>(gen);
            outputs[i] = inputA[i] || inputB[i];
        }
        VEC_TYPE t0(inputA);
        VEC_TYPE t1(inputB);
        // Return type of LAND should be a mask, regardless of VEC_TYPE.
        auto t2 = t0.lor(t1);
        t2.store(values);
        CHECK_CONDITION(valuesExact(values, outputs, VEC_LEN), "LORV gen");
    }
    {
        bool values[VEC_LEN];
        bool outputs[VEC_LEN];
        SCALAR_TYPE inputA[VEC_LEN];
        SCALAR_TYPE inputB[VEC_LEN];

        for (int i = 0; i < VEC_LEN; i++)
        {
            inputA[i] = randomValue<SCALAR_TYPE>(gen);
            inputB[i] = randomValue<SCALAR_TYPE>(gen);
            outputs[i] = inputA[i] || inputB[i];
        }
        VEC_TYPE t0(inputA);
        VEC_TYPE t1(inputB);
        // Return type of LAND should be a mask, regardless of VEC_TYPE.
        auto t2 = t0 || t1;
        t2.store(values);
        CHECK_CONDITION(valuesExact(values, outputs, VEC_LEN), "LORV gen (operator ||)");
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

        for (unsigned int i = 0; i < VEC_LEN; i++) {
            vec0.insert(i, DATA_SET::inputs::inputA[i]);
        }
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "INSERT");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0;

        for (uint32_t i = 0; i < VEC_LEN; i++) {
            vec0[i] = DATA_SET::inputs::inputA[i];
        }
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "INSERT(operator[] =)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericASSIGNVTest()
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
void genericMASSIGNVTest()
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
void genericASSIGNSTest()
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
void genericMASSIGNSTest()
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
void genericLOAD_STORETest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0;
        vec0.load(DATA_SET::inputs::inputA);
        vec0.store(values);
        bool inRange = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange), "LOAD/STORE");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN>
void genericMLOADTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE inputB[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];
    bool inputMask[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        inputB[i] = randomValue<SCALAR_TYPE>(gen);
        inputMask[i] = randomValue<bool>(gen);

        output[i] = inputMask[i] ? inputB[i] : inputA[i];
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        MASK_TYPE mask(inputMask);

        vec0.load(mask, inputB);
        vec0.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange), "MLOAD");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN>
void genericMSTORETest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE inputB[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];
    bool inputMask[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        inputB[i] = randomValue<SCALAR_TYPE>(gen);
        inputMask[i] = randomValue<bool>(gen);

        output[i] = inputMask[i] ? inputB[i] : 0;
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        MASK_TYPE mask(inputMask);

        memset(values, 0, sizeof(SCALAR_TYPE)*VEC_LEN);

        vec0.load(inputB);
        vec0.store(mask, values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange), "MSTORE");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericLOADA_STOREATest()
{
    {
        alignas(VEC_TYPE::alignment()) SCALAR_TYPE aligned_in[VEC_LEN];
        alignas(VEC_TYPE::alignment()) SCALAR_TYPE values[VEC_LEN];

        for (int i = 0; i < VEC_LEN; i++) aligned_in[i] = DATA_SET::inputs::inputA[i];

        VEC_TYPE vec0;
        vec0.loada(aligned_in);
        vec0.storea(values);
        bool inRange = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange), "LOADA/STOREA");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN>
void genericMLOADATest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    alignas(VEC_TYPE::alignment()) SCALAR_TYPE inputA[VEC_LEN];
    alignas(VEC_TYPE::alignment()) SCALAR_TYPE inputB[VEC_LEN];
    alignas(VEC_TYPE::alignment()) SCALAR_TYPE output[VEC_LEN];
    alignas(MASK_TYPE::alignment()) bool inputMask[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        inputB[i] = randomValue<SCALAR_TYPE>(gen);
        inputMask[i] = randomValue<bool>(gen);

        output[i] = inputMask[i] ? inputB[i] : inputA[i];
    }

    {
        alignas(VEC_TYPE::alignment()) SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        MASK_TYPE mask(inputMask);

        vec0.loada(mask, inputB);
        vec0.storea(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange), "MLOADA");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN>
void genericMSTOREATest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    alignas(VEC_TYPE::alignment()) SCALAR_TYPE inputA[VEC_LEN];
    alignas(VEC_TYPE::alignment()) SCALAR_TYPE inputB[VEC_LEN];
    alignas(VEC_TYPE::alignment()) SCALAR_TYPE output[VEC_LEN];
    alignas(MASK_TYPE::alignment()) bool inputMask[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        inputB[i] = randomValue<SCALAR_TYPE>(gen);
        inputMask[i] = randomValue<bool>(gen);

        output[i] = inputMask[i] ? inputB[i] : 0;
    }

    {
        alignas(VEC_TYPE::alignment()) SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        MASK_TYPE mask(inputMask);

        memset(values, 0, sizeof(SCALAR_TYPE)*VEC_LEN);

        vec0.loada(inputB);
        vec0.storea(mask, values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange), "MSTOREA");
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "ADDV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0 + vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "ADDV(operator+)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2;
        vec2 = UME::SIMD::FUNCTIONS::add(vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "ADDV(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN>
void genericADDVTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE inputB[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        inputB[i] = randomValue<SCALAR_TYPE>(gen);
        output[i] = inputA[i] + inputB[i];
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        VEC_TYPE vec2 = vec0.add(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "ADDV gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        VEC_TYPE vec2 = vec0 + vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "ADDV(operator+) gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        VEC_TYPE vec2;
        vec2 = UME::SIMD::FUNCTIONS::add(vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "ADDV(function) gen");
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "MADDV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::add(mask, vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MADDV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "MADDV(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN>
void genericMADDVTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE inputB[VEC_LEN];
    bool inputMask[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        inputB[i] = randomValue<SCALAR_TYPE>(gen);
        inputMask[i] = randomValue<bool>(gen);
        output[i] = inputMask[i] ? inputA[i] + inputB[i] : inputA[i];
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        MASK_TYPE mask(inputMask);
        VEC_TYPE vec2 = vec0.add(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "MADDV gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        MASK_TYPE mask(inputMask);
        VEC_TYPE vec2;
        vec2 = UME::SIMD::FUNCTIONS::add(mask, vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "MADDV(function) gen");
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "ADDS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0 + DATA_SET::inputs::scalarA;
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "ADDS(operator+ RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1;
        vec1 = UME::SIMD::FUNCTIONS::add(vec0, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "ADDS(function - RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = DATA_SET::inputs::scalarA + vec0;
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "ADDS(operator+ LHS scalar");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1;
        vec1 = UME::SIMD::FUNCTIONS::add(DATA_SET::inputs::scalarA, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ADDS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "ADDS(function - LHS scalar");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN>
void genericADDSTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE inputB;
    SCALAR_TYPE output[VEC_LEN];

    inputB = randomValue<SCALAR_TYPE>(gen);
    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        output[i] = inputA[i] + inputB;
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1 = vec0.add(inputB);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "ADDS gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1 = vec0 + inputB;
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "ADDS(operator+ RHS scalar) gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1 = inputB + vec0;
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "ADDS(operator+ LHS scalar) gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1;
        vec1 = UME::SIMD::FUNCTIONS::add(vec0, inputB);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "ADDV(function RHS scalar) gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1;
        vec1 = UME::SIMD::FUNCTIONS::add(inputB, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "ADDV(function LHS scalar) gen");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMADDSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.add(mask, DATA_SET::inputs::scalarA);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MADDS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "MADDS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::add(mask, vec0, DATA_SET::inputs::scalarA);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MADDS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "MADDS (function - RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::add(mask, DATA_SET::inputs::scalarA, vec0);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MADDS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        //CHECK_CONDITION((inRange & isUnmodified), "MADDS (function - LHS scalar)");
        // TODO: MADDS with LHS requires separate test output data.
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN>
void genericMADDSTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE inputB;
    SCALAR_TYPE output[VEC_LEN];
    SCALAR_TYPE outputLHS[VEC_LEN];
    bool inputMask[VEC_LEN];

    inputB = randomValue<SCALAR_TYPE>(gen);
    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        inputMask[i] = randomValue<bool>(gen);
        output[i] = inputMask[i] ? inputA[i] + inputB : inputA[i];
        outputLHS[i] = inputMask[i] ? inputA[i] + inputB : inputB;
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        MASK_TYPE mask(inputMask);
        VEC_TYPE vec1 = vec0.add(mask, inputB);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "MADDS gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1;
        MASK_TYPE mask(inputMask);
        vec1 = UME::SIMD::FUNCTIONS::add(mask, vec0, inputB);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "MADDS(function RHS scalar) gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1;
        MASK_TYPE mask(inputMask);
        vec1 = UME::SIMD::FUNCTIONS::add(mask, inputB, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, outputLHS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "MADDS(function LHS scalar) gen");
    }
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

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN>
void genericSADDVTest_random() {

    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE inputB[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];

    SCALAR_TYPE maxVal = std::numeric_limits<SCALAR_TYPE>::max();
    SCALAR_TYPE minVal = std::numeric_limits<SCALAR_TYPE>::min();

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        inputB[i] = randomValue<SCALAR_TYPE>(gen);

        if (inputA[i] > 0 && inputB[i] > 0) {
            output[i] = ((maxVal - inputA[i]) < inputB[i]) ?  maxVal : inputA[i] + inputB[i];
        }
        else if (inputA[i] < 0 && inputB[i] < 0) {
            output[i] = ((minVal - inputA[i]) > inputB[i]) ? minVal : inputA[i] + inputB[i];
        }
        else {
            output[i] = inputA[i] + inputB[i];
        }
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        VEC_TYPE vec2 = vec0.sadd(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "SADDV gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        VEC_TYPE vec2;
        vec2 = UME::SIMD::FUNCTIONS::sadd(vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "SADDV(function) gen");
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
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::postinc(vec0);
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFINC, VEC_LEN, SCALAR_TYPE(0.01f));
        bool inRange1 = valuesInRange(values1, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange0 && inRange1, "POSTINC(function)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMPOSTINCTest()
{
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
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::postinc(mask, vec0);
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::MPOSTPREFINC, VEC_LEN, SCALAR_TYPE(0.01f));
        bool inRange1 = valuesInRange(values1, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange0 && inRange1, "MPOSTINC(function)");
    }
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
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::prefinc(vec0);
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFINC, VEC_LEN, SCALAR_TYPE(0.01f));
        bool inRange1 = valuesInRange(values1, DATA_SET::outputs::POSTPREFINC, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange0 && inRange1, "PREFINC(function)");
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
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::prefinc(mask, vec0);
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::MPOSTPREFINC, VEC_LEN, SCALAR_TYPE(0.01f));
        bool inRange1 = valuesInRange(values1, DATA_SET::outputs::MPOSTPREFINC, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange0 && inRange1, "MPREFINC(function)");
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "SUBV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0 - vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SUBV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "SUBV(operator-)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::sub(vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SUBV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "SUBV(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN>
void genericSUBVTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE inputB[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        inputB[i] = randomValue<SCALAR_TYPE>(gen);
        output[i] = inputA[i] - inputB[i];
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        VEC_TYPE vec2 = vec0.sub(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "SUBV gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        VEC_TYPE vec2 = vec0 - vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "SUBV(operator-) gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        VEC_TYPE vec2;
        vec2 = UME::SIMD::FUNCTIONS::sub(vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "SUBV(function) gen");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSUBVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.sub(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MSUBV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::sub(mask, vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MSUBV(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN>
void genericMSUBVTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE inputB[VEC_LEN];
    bool inputMask[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        inputB[i] = randomValue<SCALAR_TYPE>(gen);
        inputMask[i] = randomValue<bool>(gen);
        output[i] = inputMask[i] ? inputA[i] - inputB[i] : inputA[i];
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        MASK_TYPE mask(inputMask);
        VEC_TYPE vec2 = vec0.sub(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "MSUBV gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        MASK_TYPE mask(inputMask);
        VEC_TYPE vec2;
        vec2 = UME::SIMD::FUNCTIONS::sub(mask, vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "MSUBV(function) gen");
    }
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "SUBS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0 - DATA_SET::inputs::scalarA;
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SUBS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "SUBS(operator- RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::sub(vec0, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SUBS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "SUBS(function RHS scalar)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSUBSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.sub(mask, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MSUBS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::sub(mask, vec0, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MSUBS(function -RHS scalar)");
    }
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
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0.subfrom(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SUBFROMV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "SUBFROMV");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSUBFROMVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.subfrom(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBFROMV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MSUBFROMV");
    }
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "SUBFROMS");
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
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.subfrom(mask, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBFROMS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MSUBFROMS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::sub(mask, DATA_SET::inputs::scalarA, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MSUBFROMS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MSUBFROMS (function LHS scalar)");
    }
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
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::postdec(vec0);
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFDEC, VEC_LEN, SCALAR_TYPE(0.01f));
        bool inRange1 = valuesInRange(values1, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange0 && inRange1, "POSTDEC(function)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMPOSTDECTest()
{
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
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::postdec(mask, vec0);
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::MPOSTPREFDEC, VEC_LEN, SCALAR_TYPE(0.01f));
        bool inRange1 = valuesInRange(values1, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange0 && inRange1, "MPOSTDEC(function)");
    }
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
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::prefdec(vec0);
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::POSTPREFDEC, VEC_LEN, SCALAR_TYPE(0.01f));
        bool inRange1 = valuesInRange(values1, DATA_SET::outputs::POSTPREFDEC, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange0 && inRange1, "PREFDEC(function)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMPREFDECTest()
{
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
    {
        SCALAR_TYPE values0[VEC_LEN];
        SCALAR_TYPE values1[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::prefdec(mask, vec0);
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, DATA_SET::outputs::MPOSTPREFDEC, VEC_LEN, SCALAR_TYPE(0.01f));
        bool inRange1 = valuesInRange(values1, DATA_SET::outputs::MPOSTPREFDEC, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange0 && inRange1, "MPREFDEC(function)");
    }
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MULV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0 * vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MULV(operator*)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::mul(vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MULV(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMULVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.mul(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MMULV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MMULV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::mul(mask, vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MMULV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MMULV(function)");
    }
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MULS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0 * DATA_SET::inputs::scalarA;
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MULS(operator* RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::mul(vec0, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MULS(function - RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::mul(DATA_SET::inputs::scalarA, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MULS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MULS(function LHS scalar)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMULSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.mul(mask, DATA_SET::inputs::scalarA);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MMULS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MMULS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::mul(mask, vec0, DATA_SET::inputs::scalarA);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MMULS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MMULS(function - RHS scalar)");
    }
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "DIVV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0 / vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::DIVV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "DIVV(operator/)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::div(vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::DIVV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "DIVV(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMDIVVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.div(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MDIVV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MDIVV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::div(mask, vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MDIVV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MDIVV(function)");
    }
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "DIVS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0 / DATA_SET::inputs::scalarA;
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::DIVS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "DIVS(operator/ RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::div(vec0, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::DIVS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "DIVS(function - RHS scalar)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMDIVSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.div(mask, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MDIVS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MDIVS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::div(mask, vec0, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MDIVS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MDIVS(function - RHS scalar)");
    }
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

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN>
void genericREMVTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE inputB[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        // C++ standard says:
        // "The binary / operator yields the quotient, and the binary % operator 
        //  yields the remainder from the division of the first expression by the
        //  second. If the second operand of / or % is zero the behavior is undefined. 
        //  For integral operands the / operator yields the algebraic quotient with any 
        //  fractional part discarded; if the quotient a/b is representable in the type 
        //  of the result, (a/b)*b + a%b is equal to a."
        // And also:
        // "If both operands are nonnegative then the remainder is nonnegative; 
        //  if not, the sign of the remainder is implementation-defined."
        // This means we can only check this operation for non-negative left operand
        // and positive, non-zero right operand.
        do {
            inputA[i] = randomValue<SCALAR_TYPE>(gen);
        } while (inputA[i] < 0);
        do {
            inputB[i] = randomValue<SCALAR_TYPE>(gen);
        } while (inputB[i] <= 0);
        output[i] = inputA[i] % inputB[i];
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        VEC_TYPE vec2 = vec0.rem(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "REMV gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        VEC_TYPE vec2 = vec0 % vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "REMV(operator%) gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        VEC_TYPE vec2;
        vec2 = UME::SIMD::FUNCTIONS::rem(vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "REMV(function) gen");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN>
void genericMREMVTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE inputB[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];
    bool inputMask[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        // C++ standard says:
        // "The binary / operator yields the quotient, and the binary % operator 
        //  yields the remainder from the division of the first expression by the
        //  second. If the second operand of / or % is zero the behavior is undefined. 
        //  For integral operands the / operator yields the algebraic quotient with any 
        //  fractional part discarded; if the quotient a/b is representable in the type 
        //  of the result, (a/b)*b + a%b is equal to a."
        // And also:
        // "If both operands are nonnegative then the remainder is nonnegative; 
        //  if not, the sign of the remainder is implementation-defined."
        // This means we can only check this operation for non-negative left operand
        // and positive, non-zero right operand.
        do {
            inputA[i] = randomValue<SCALAR_TYPE>(gen);
        }
        while(inputA[i] < 0);
        do {
            inputB[i] = randomValue<SCALAR_TYPE>(gen);
        } while (inputB[i] <= 0);
        inputMask[i] = randomValue<bool>(gen);
        output[i] = inputMask[i] ? (inputA[i] % inputB[i]) : inputA[i];
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        MASK_TYPE mask(inputMask);
        VEC_TYPE vec2 = vec0.rem(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "MREMV gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        MASK_TYPE mask(inputMask);
        VEC_TYPE vec2;
        vec2 = UME::SIMD::FUNCTIONS::rem(mask, vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "MREMV(function) gen");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN>
void genericREMSTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE inputB;
    SCALAR_TYPE output[VEC_LEN];

    do {
        inputB = randomValue<SCALAR_TYPE>(gen);
    } while (inputB <= 0);

    for (int i = 0; i < VEC_LEN; i++) {
        do {
            inputA[i] = randomValue<SCALAR_TYPE>(gen);
        } while (inputA[i] < 0);
        output[i] = inputA[i] % inputB;
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1 = vec0.rem(inputB);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "REMS gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1 = vec0 % inputB;
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "REMS(operator% RHS scalar) gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1;
        vec1 = UME::SIMD::FUNCTIONS::rem(vec0, inputB);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "REMS(function RHS scalar) gen");
    }

    do {
        inputB = randomValue<SCALAR_TYPE>(gen);
    } while (inputB < 0);

    for (int i = 0; i < VEC_LEN; i++) {
        do {
            inputA[i] = randomValue<SCALAR_TYPE>(gen);
        } while (inputA[i] <= 0);
        output[i] = inputB % inputA[i];
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1;
        vec1 = UME::SIMD::FUNCTIONS::rem(inputB, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "REMS(function LHS scalar) gen");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRCPTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.rcp();
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::RCP, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "RCP");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::rcp(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::RCP, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "RCP(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRCPTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.rcp(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MRCP, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MRCP");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::rcp(mask, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MRCP, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MRCP(function)");
    }
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "RCPS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.rcp(DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::RCPS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "RCPS(operator/ LHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::rcp(vec0, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::RCPS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "RCPS(function)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRCPSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.rcp(mask, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MRCPS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MRCPS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::rcp(mask, vec0, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MRCPS, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MRCPS (function)");
    }
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(true);
        mask = UME::SIMD::FUNCTIONS::cmpeq(vec0, vec1);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPEQV, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPEQV(function)");
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
        mask = UME::SIMD::FUNCTIONS::cmpeq(vec0, DATA_SET::inputs::scalarA);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPEQS, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPEQS(function - RHS scalar)");
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = UME::SIMD::FUNCTIONS::cmpeq(DATA_SET::inputs::scalarA, vec0);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPEQS, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPEQS(function - LHS scalar)");
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(true);
        mask = UME::SIMD::FUNCTIONS::cmpne(vec0, vec1);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPNEV, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPNEV(function)");
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
        mask = UME::SIMD::FUNCTIONS::cmpne(vec0, DATA_SET::inputs::scalarA);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPNES, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPNES(function - RHS scalar)");
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = UME::SIMD::FUNCTIONS::cmpne(DATA_SET::inputs::scalarA, vec0);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPNES, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPNES(function - LHS scalar)");
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(true);
        mask = UME::SIMD::FUNCTIONS::cmpgt(vec0, vec1);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPGTV, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPGTV(function)");
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
        mask = UME::SIMD::FUNCTIONS::cmpgt(vec0, DATA_SET::inputs::scalarA);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPGTS, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPGTS(function - RHS scalar)");
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = UME::SIMD::FUNCTIONS::cmpgt(DATA_SET::inputs::scalarA, vec0);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPLTS, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPGTS(function - LHS scalar)");
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(true);
        mask = UME::SIMD::FUNCTIONS::cmplt(vec0, vec1);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPLTV, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPLTV(function)");
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
        mask = UME::SIMD::FUNCTIONS::cmplt(vec0, DATA_SET::inputs::scalarA);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPLTS, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPLTS(function - RHS scalar)");
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = UME::SIMD::FUNCTIONS::cmplt(DATA_SET::inputs::scalarA, vec0);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPGTS, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPLTS(function - LHS scalar)");
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(true);
        mask = UME::SIMD::FUNCTIONS::cmpge(vec0, vec1);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPGEV, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPGEV(function)");
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
        mask = UME::SIMD::FUNCTIONS::cmpge(vec0, DATA_SET::inputs::scalarA);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPGES, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPGES(function RHS scalar)");
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = UME::SIMD::FUNCTIONS::cmpge(DATA_SET::inputs::scalarA, vec0);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPLES, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPGES(function - LHS scalar)");
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(true);
        mask = UME::SIMD::FUNCTIONS::cmple(vec0, vec1);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPLEV, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPLEV(function)");
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
        mask = UME::SIMD::FUNCTIONS::cmple(vec0, DATA_SET::inputs::scalarA);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPLES, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPLES(function - LHS scalar)");
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
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(true);
        mask = UME::SIMD::FUNCTIONS::cmple(DATA_SET::inputs::scalarA, vec0);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::CMPGES, VEC_LEN);
        CHECK_CONDITION(inRange, "CMPLES(function - RHS scalar)");
    }
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPEVTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        bool value = vec0.cmpe(vec1);
        CHECK_CONDITION(value == DATA_SET::outputs::CMPEV, "CMPEV");
    }
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        bool value = UME::SIMD::FUNCTIONS::cmpe(vec0, vec1);
        CHECK_CONDITION(value == DATA_SET::outputs::CMPEV, "CMPEV(function)");
    }
}

template<typename VEC_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericCMPESTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        bool value = vec0.cmpe(DATA_SET::inputs::scalarA);
        CHECK_CONDITION(value == DATA_SET::outputs::CMPES, "CMPES");
    }
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        bool value = UME::SIMD::FUNCTIONS::cmpe(vec0, DATA_SET::inputs::scalarA);
        CHECK_CONDITION(value == DATA_SET::outputs::CMPES, "CMPES(function - RHS scalar)");
    }
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        bool value = UME::SIMD::FUNCTIONS::cmpe(DATA_SET::inputs::scalarA, vec0);
        CHECK_CONDITION(value == DATA_SET::outputs::CMPES, "CMPES(function - LHS scalar)");
    }
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BANDV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0 & vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BANDV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BANDV(operator&)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::band(vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BANDV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BANDV(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBANDVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.band(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBANDV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MBANDV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::band(mask, vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBANDV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MBANDV");
    }
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BANDS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = vec0 & DATA_SET::inputs::scalarA;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BANDS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BANDS(operator & RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::band(vec0, DATA_SET::inputs::scalarA);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BANDS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BANDS(function - RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = DATA_SET::inputs::scalarA & vec0;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BANDS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BANDS(operator & LHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::band(DATA_SET::inputs::scalarA, vec0);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BANDS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BANDS(function - LHS scalar)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBANDSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.band(mask, DATA_SET::inputs::scalarA);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBANDS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MBANDS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::band(mask, vec0, DATA_SET::inputs::scalarA);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBANDS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MBANDS(function - RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::band(mask, DATA_SET::inputs::scalarA, vec0);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBANDS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        //CHECK_CONDITION((inRange && isUnmodified), "MBANDS(function - LHS scalar)");
        // TODO: this function requires separate output data
    }
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
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0.bor(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BORV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BORV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0 | vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BORV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BORV(operator |)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::bor(vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BORV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BORV(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBORVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.bor(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBORV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MBORV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::bor(mask, vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBORV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MBORV(function)");
    }
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BORS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = vec0 | DATA_SET::inputs::scalarA;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BORS(operator| RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::bor(vec0, DATA_SET::inputs::scalarA);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BORS(function - RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = DATA_SET::inputs::scalarA | vec0;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BORS(operator| LHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::bor(DATA_SET::inputs::scalarA, vec0);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BORS(function - LHS scalar)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBORSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.bor(mask, DATA_SET::inputs::scalarA);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MBORS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::bor(mask, vec0, DATA_SET::inputs::scalarA);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MBORS(function - RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::bor(mask, DATA_SET::inputs::scalarA, vec0);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        //CHECK_CONDITION((inRange && isUnmodified), "MBORS(function - RHS scalar)");
        // TODO: this test requires separate output data
    }
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BXORV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0 ^ vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BXORV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BXORV(operator^");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::bxor(vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BXORV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BXORV(function");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBXORVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.bxor(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBXORV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MBXORV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::bxor(mask, vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBXORV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MBXORV(function)");
    }
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BXORS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = vec0 ^ DATA_SET::inputs::scalarA;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BXORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BXORS(operator^ RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::bxor(vec0, DATA_SET::inputs::scalarA);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BXORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BXORS(function - RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = DATA_SET::inputs::scalarA ^ vec0;
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BXORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BXORS(operator ^ LHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::bxor(DATA_SET::inputs::scalarA, vec0);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BXORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BXORS(function - LHS scalar)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBXORSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.bxor(mask, DATA_SET::inputs::scalarA);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBXORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MBXORS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::bxor(mask, vec0, DATA_SET::inputs::scalarA);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBXORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MBXORS(function - RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::bxor(mask, DATA_SET::inputs::scalarA, vec0);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBXORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        // CHECK_CONDITION((inRange && isUnmodified), "MBXORS(function - LHS scalar)");
        // TODO: this test requires separate output data
    }
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BNOT");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = ~vec0;
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BNOT, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BNOT(operator!)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::bnot(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::BNOT, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "BNOT(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMBNOTTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.bnot(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBNOT, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MBNOT");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::bnot(mask, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MBNOT, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MBNOT(function)");
    }
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

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN>
void genericBLENDVTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE inputB[VEC_LEN];
    bool inputMask[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        inputB[i] = randomValue<SCALAR_TYPE>(gen);
        inputMask[i] = randomValue<bool>(gen);

        output[i] = inputMask[i] ? inputB[i] : inputA[i];
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1(inputB);
        MASK_TYPE mask0(inputMask);

        VEC_TYPE vec2 = vec0.blend(mask0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        vec1.store(values);
        isUnmodified &= valuesInRange(values, inputB, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "BLENDV gen");
    }
}
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN>
void genericBLENDSTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE inputB = randomValue<SCALAR_TYPE>(gen);
    bool inputMask[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];


    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        inputMask[i] = randomValue<bool>(gen);

        output[i] = inputMask[i] ? inputB : inputA[i];
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        MASK_TYPE mask0(inputMask);

        VEC_TYPE vec2 = vec0.blend(mask0, inputB);
        vec2.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "BLENDS gen");
    }
}
        // assign
        // SWIZZLE  - Swizzle (reorder/permute) vector elements
        // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign
 
        //(Reduction to scalar operations)
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHADDTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = vec0.hadd();
        bool inRange = valueInRange(value, DATA_SET::outputs::HADD[VEC_LEN - 1], SCALAR_TYPE(SCALAR_TYPE(0.01f)));
        CHECK_CONDITION(inRange, "HADD");
    }
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = UME::SIMD::FUNCTIONS::hadd(vec0);
        bool inRange = valueInRange(value, DATA_SET::outputs::HADD[VEC_LEN - 1], SCALAR_TYPE(SCALAR_TYPE(0.01f)));
        CHECK_CONDITION(inRange, "HADD(function)");
    }
}

// MHADD - Masked add elements of a vector (horizontal add)
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHMULTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = vec0.hmul();
        bool inRange = valueInRange(value, DATA_SET::outputs::HMUL[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "HMUL");
    }
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = UME::SIMD::FUNCTIONS::hmul(vec0);
        bool inRange = valueInRange(value, DATA_SET::outputs::HMUL[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "HMUL(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN>
void genericHMULTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE output;

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
    }

    output = inputA[0];
    for (int i = 1; i < VEC_LEN; i++) {
        output *= inputA[i];
    }

    {
        VEC_TYPE vec0(inputA);
        SCALAR_TYPE value = vec0.hmul();
        bool inRange = valueInRange(value, output, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange), "HMUL gen");
    }
    {
        VEC_TYPE vec0(inputA);
        SCALAR_TYPE value = UME::SIMD::FUNCTIONS::hmul(vec0);
        bool inRange = valueInRange(value, output, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange), "HMUL(function) gen");
    }
}

        // MHMUL - Masked multiply elements of a vector (horizontal mul)
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHBANDTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = vec0.hband();
        bool inRange = valueInRange(value, DATA_SET::outputs::HBAND[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange , "HBAND");
    }
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = UME::SIMD::FUNCTIONS::hband(vec0);
        bool inRange = valueInRange(value, DATA_SET::outputs::HBAND[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange , "HBAND");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMHBANDTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        SCALAR_TYPE value = vec0.hband(mask0);
        bool inRange = valueInRange(value, DATA_SET::outputs::MHBAND[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MHBAND");
    }
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        SCALAR_TYPE value = UME::SIMD::FUNCTIONS::hband(mask0, vec0);
        bool inRange = valueInRange(value, DATA_SET::outputs::MHBAND[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MHBAND(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHBANDSTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = vec0.hband(DATA_SET::inputs::scalarA);
        bool inRange = valueInRange(value, DATA_SET::outputs::HBANDS[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "HBANDS");
    }/*
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = UME::SIMD::FUNCTIONS::hband(vec0, DATA_SET::inputs::scalarA);
        bool inRange = valueInRange(value, DATA_SET::outputs::HBANDS[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "HBANDS(function - RHS scalar)");
    }*/
}
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMHBANDSTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        SCALAR_TYPE value = vec0.hband(mask0, DATA_SET::inputs::scalarA);
        bool inRange = valueInRange(value, DATA_SET::outputs::MHBANDS[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MHBANDS");
    }
    /*{
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        SCALAR_TYPE value = UME::SIMD::FUNCTIONS::hband(mask0, vec0, DATA_SET::inputs::scalarA);
        bool inRange = valueInRange(value, DATA_SET::outputs::MHBANDS[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MHBANDS(function - RHS scalar)");
    }*/
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHBORTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = vec0.hbor();
        bool inRange = valueInRange(value, DATA_SET::outputs::HBOR[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "HBOR");
    }
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = UME::SIMD::FUNCTIONS::hbor(vec0);
        bool inRange = valueInRange(value, DATA_SET::outputs::HBOR[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "HBOR(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMHBORTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        SCALAR_TYPE value = vec0.hbor(mask0);
        bool inRange = valueInRange(value, DATA_SET::outputs::MHBOR[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MHBOR");
    }
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        SCALAR_TYPE value = UME::SIMD::FUNCTIONS::hbor(mask0, vec0);
        bool inRange = valueInRange(value, DATA_SET::outputs::MHBOR[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MHBOR(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHBORSTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = vec0.hbor(DATA_SET::inputs::scalarA);
        bool inRange = valueInRange(value, DATA_SET::outputs::HBORS[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "HBORS");
    }/*
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = UME::SIMD::FUNCTIONS::hbor(vec0, DATA_SET::inputs::scalarA);
        bool inRange = valueInRange(value, DATA_SET::outputs::HBORS[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "HBORS(function - RHS scalar)");
    }*/
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMHBORSTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        SCALAR_TYPE value = vec0.hbor(mask0, DATA_SET::inputs::scalarA);
        bool inRange = valueInRange(value, DATA_SET::outputs::MHBORS[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MHBORS");
    }/*
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        SCALAR_TYPE value = UME::SIMD::FUNCTIONS::hbor(mask0, vec0, DATA_SET::inputs::scalarA);
        bool inRange = valueInRange(value, DATA_SET::outputs::MHBORS[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MHBORS(function - RHS scalar)");
    }*/
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHBXORTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = vec0.hbxor();
        bool inRange = valueInRange(value, DATA_SET::outputs::HBXOR[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "HBXOR");
    }
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = UME::SIMD::FUNCTIONS::hbxor(vec0);
        bool inRange = valueInRange(value, DATA_SET::outputs::HBXOR[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "HBXOR(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMHBXORTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        SCALAR_TYPE value = vec0.hbxor(mask0);
        bool inRange = valueInRange(value, DATA_SET::outputs::MHBXOR[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MHBXOR");
    }
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        SCALAR_TYPE value = UME::SIMD::FUNCTIONS::hbxor(mask0, vec0);
        bool inRange = valueInRange(value, DATA_SET::outputs::MHBXOR[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MHBXOR");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHBXORSTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = vec0.hbxor(DATA_SET::inputs::scalarA);
        bool inRange = valueInRange(value, DATA_SET::outputs::HBXORS[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "HBXORS");
    }
    /*{
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = UME::SIMD::FUNCTIONS::hbxor(vec0, DATA_SET::inputs::scalarA);
        bool inRange = valueInRange(value, DATA_SET::outputs::HBXORS[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "HBXORS(function - RHS scalar)");
    }*/
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMHBXORSTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        SCALAR_TYPE value = vec0.hbxor(mask0, DATA_SET::inputs::scalarA);
        bool inRange = valueInRange(value, DATA_SET::outputs::MHBXORS[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MHBXORS");
    }
    /*{
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        SCALAR_TYPE value = UME::SIMD::FUNCTIONS::hbxor(mask0, vec0, DATA_SET::inputs::scalarA);
        bool inRange = valueInRange(value, DATA_SET::outputs::MHBXORS[VEC_LEN - 1], SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MHBXORS(function - RHS scalar)");
    }*/
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericFMULADDVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2(DATA_SET::inputs::inputC);
        VEC_TYPE vec3 = vec0.fmuladd(vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::FMULADDV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "FMULADDV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2(DATA_SET::inputs::inputC);
        VEC_TYPE vec3 = UME::SIMD::FUNCTIONS::fmuladd(vec0, vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::FMULADDV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "FMULADDV(function)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMFMULADDVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2(DATA_SET::inputs::inputC);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec3 = vec0.fmuladd(mask, vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MFMULADDV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MFMULADDV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2(DATA_SET::inputs::inputC);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec3 = UME::SIMD::FUNCTIONS::fmuladd(mask, vec0, vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MFMULADDV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MFMULADDV(function)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericFMULSUBVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2(DATA_SET::inputs::inputC);
        VEC_TYPE vec3 = vec0.fmulsub(vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::FMULSUBV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "FMULSUBV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2(DATA_SET::inputs::inputC);
        VEC_TYPE vec3 = UME::SIMD::FUNCTIONS::fmulsub(vec0, vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::FMULSUBV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "FMULSUBV(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMFMULSUBVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2(DATA_SET::inputs::inputC);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec3 = vec0.fmulsub(mask, vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MFMULSUBV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MFMULSUBV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2(DATA_SET::inputs::inputC);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec3 = UME::SIMD::FUNCTIONS::fmulsub(mask, vec0, vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MFMULSUBV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MFMULSUBV(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericFADDMULVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2(DATA_SET::inputs::inputC);
        VEC_TYPE vec3 = vec0.faddmul(vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::FADDMULV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "FADDMULV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2(DATA_SET::inputs::inputC);
        VEC_TYPE vec3 = UME::SIMD::FUNCTIONS::faddmul(vec0, vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::FADDMULV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "FADDMULV(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMFADDMULVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2(DATA_SET::inputs::inputC);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec3 = vec0.faddmul(mask, vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MFADDMULV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MFADDMULV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2(DATA_SET::inputs::inputC);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec3 = UME::SIMD::FUNCTIONS::faddmul(mask, vec0, vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MFADDMULV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MFADDMULV(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericFSUBMULVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2(DATA_SET::inputs::inputC);
        VEC_TYPE vec3 = vec0.fsubmul(vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::FSUBMULV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "FSUBMULV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2(DATA_SET::inputs::inputC);
        VEC_TYPE vec3 = UME::SIMD::FUNCTIONS::fsubmul(vec0, vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::FSUBMULV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "FSUBMULV(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMFSUBMULVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2(DATA_SET::inputs::inputC);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec3 = vec0.fsubmul(mask, vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MFSUBMULV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MFSUBMULV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2(DATA_SET::inputs::inputC);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec3 = UME::SIMD::FUNCTIONS::fsubmul(mask, vec0, vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MFSUBMULV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MFSUBMULV(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMAXVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0.max(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MAXV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MAXV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::max(vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MAXV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MAXV(function)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMAXVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.max(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MMAXV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MMAXV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::max(mask, vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MMAXV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MMAXV(function)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMAXSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.max(DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MAXS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MAXS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::max(vec0, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MAXS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MAXS(function - RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::max(DATA_SET::inputs::scalarA, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MAXS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MAXS(function - LHS scalar)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMAXSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.max(mask, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MMAXS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MMAXS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::max(mask, vec0, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MMAXS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MMAXS(function - RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::max(mask, DATA_SET::inputs::scalarA, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MMAXS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        //CHECK_CONDITION((inRange && isUnmodified), "MMAXS(function - LHS scalar)");
        // TODO: this test requires separate output data
    }
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
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = vec0.min(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MINV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MINV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::min(vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MINV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MINV(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMINVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.min(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MMINV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MMINV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1(DATA_SET::inputs::inputB);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::min(mask, vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MMINV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MMINV(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericMINSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.min(DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MINS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MINS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::min(vec0, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MINS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MINS(function - RHS scalar");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::min(DATA_SET::inputs::scalarA, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MINS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MINS(function - LHS scalar");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMMINSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.min(mask, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MMINS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MMINS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::min(mask, vec0, DATA_SET::inputs::scalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MMINS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MMINS(function - RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::min(mask, DATA_SET::inputs::scalarA, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MMINS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        //CHECK_CONDITION((inRange && isUnmodified), "MMINS(function - LHS scalar)");
        // TODO: this test requires separate output data
    }
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
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = vec0.hmax();
        SCALAR_TYPE expected = DATA_SET::outputs::HMAX[VEC_LEN - 1];
        bool inRange = valueInRange(value, expected, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "HMAX");
    }
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = UME::SIMD::FUNCTIONS::hmax(vec0);
        SCALAR_TYPE expected = DATA_SET::outputs::HMAX[VEC_LEN - 1];
        bool inRange = valueInRange(value, expected, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "HMAX(function)");
    }
}

// MHMAX
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericHMINTest()
{
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = vec0.hmin();
        SCALAR_TYPE expected = DATA_SET::outputs::HMIN[VEC_LEN - 1];
        bool inRange = valueInRange(value, expected, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "HMIN");
    }
    {
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        SCALAR_TYPE value = UME::SIMD::FUNCTIONS::hmin(vec0);
        SCALAR_TYPE expected = DATA_SET::outputs::HMIN[VEC_LEN - 1];
        bool inRange = valueInRange(value, expected, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "HMIN(function)");
    }
}

// MHMIN
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericLSHVTest() 
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        VEC_TYPE vec2 = vec0.lsh(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::LSHV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "LSHV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::lsh(vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::LSHV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "LSHV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        VEC_TYPE vec2 = vec0 << vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::LSHV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "LSHV (operator<<)");
    }
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMLSHVTest() 
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.lsh(mask0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MLSHV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MLSHV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::lsh(mask0, vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MLSHV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MLSHV");
    }
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericLSHSTest() 
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.lsh(DATA_SET::inputs::inputShiftScalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::LSHS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "LSHS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::lsh(vec0, DATA_SET::inputs::inputShiftScalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::LSHS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "LSHS(function - RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0 << DATA_SET::inputs::inputShiftScalarA;
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::LSHS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "LSHS(operator<< RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::lsh(DATA_SET::inputs::inputShiftScalarA, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::LSHS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        //CHECK_CONDITION((inRange && isUnmodified), "LSHS(function - LHS scalar)");
        // TODO: this test requires separate output data
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = DATA_SET::inputs::inputShiftScalarA << vec0;
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::LSHS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        //CHECK_CONDITION((inRange && isUnmodified), "LSHS(operator<< - LHS scalar)");
        // TODO: this test requires separate output data
    }
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMLSHSTest() 
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.lsh(mask0, DATA_SET::inputs::inputShiftScalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MLSHS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MLSHS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::lsh(mask0, vec0, DATA_SET::inputs::inputShiftScalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MLSHS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MLSHS(function - RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::lsh(mask0, DATA_SET::inputs::inputShiftScalarA, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MLSHS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        //CHECK_CONDITION((inRange && isUnmodified), "MLSHS(function - LHS scalar)");
        // TOOO: This test requires separate output data
    }
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
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        VEC_TYPE vec2 = vec0.rsh(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RSHV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "RSHV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::rsh(vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RSHV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "RSHV(function)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        VEC_TYPE vec2 = vec0 >> vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RSHV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "RSHV(operator>>)");
    }
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRSHVTest() 
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.rsh(mask0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRSHV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MRSHV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::rsh(mask0, vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRSHV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MRSHV(function)");
    }
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRSHSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.rsh(DATA_SET::inputs::inputShiftScalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RSHS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "RSHS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::rsh(vec0, DATA_SET::inputs::inputShiftScalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RSHS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "RSHS(function - RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0 >> DATA_SET::inputs::inputShiftScalarA;
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RSHS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "RSHS(operator>> - RHS scalar)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::rsh(DATA_SET::inputs::inputShiftScalarA, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RSHS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        // CHECK_CONDITION((inRange && isUnmodified), "RSHS(function - LHS scalar)");
        // TODO: this test requires separate output data
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = DATA_SET::inputs::inputShiftScalarA >> vec0;
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RSHS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        // CHECK_CONDITION((inRange && isUnmodified), "RSHS(function - LHS scalar)");
        // TODO: this test requires separate output data
    }
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRSHSTest() 
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.rsh(mask0, DATA_SET::inputs::inputShiftScalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRSHS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MRSHS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::rsh(mask0, vec0, DATA_SET::inputs::inputShiftScalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRSHS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MRSHS(function - RHS scalar)");
    }
    {/*
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::rsh(mask0, DATA_SET::inputs::inputShiftScalarA, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRSHS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));*/
        // CHECK_CONDITION((inRange && isUnmodified), "MRSHS(function - LHS scalar)");
        // TODO: this test requires separate output data
    }
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRSHVATest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        vec0.rsha(vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RSHV, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "RSHVA");
    }
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
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        VEC_TYPE vec2 = vec0.rol(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::ROLV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "ROLV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::rol(vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::ROLV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "ROLV(function)");
    }
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMROLVTest() 
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.rol(mask0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MROLV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MROLV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::rol(mask0, vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MROLV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MROLV(function)");
    }
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericROLSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.rol(DATA_SET::inputs::inputShiftScalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::ROLS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "ROLS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::rol(vec0, DATA_SET::inputs::inputShiftScalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::ROLS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "ROLS(function - RHS scalar)");
    }
    {
      /* SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::rol(DATA_SET::inputs::inputShiftScalarA, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::ROLS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f)); */
        //CHECK_CONDITION((inRange && isUnmodified), "ROLS(function - LHS scalar)");
        // TODO: this test requires separate output data
    }
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMROLSTest() 
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.rol(mask0, DATA_SET::inputs::inputShiftScalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MROLS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MROLS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::rol(mask0, vec0, DATA_SET::inputs::inputShiftScalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MROLS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MROLS(function - RHS scalar)");
    }
    {
        /*SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::rol(mask0, DATA_SET::inputs::inputShiftScalarA, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MROLS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));*/
        //CHECK_CONDITION((inRange && isUnmodified), "MROLS(function - LHS scalar)");
        // TODO: this test requires separate output data
    }
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
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        VEC_TYPE vec2 = vec0.ror(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RORV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "RORV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::ror(vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RORV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "RORV(function)");
    }
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRORVTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = vec0.ror(mask0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRORV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MRORV");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        UINT_VEC_TYPE vec1(DATA_SET::inputs::inputShiftA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::ror(mask0, vec0, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRORV, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MRORV(function)");
    }
}

template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericRORSTest() 
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.ror(DATA_SET::inputs::inputShiftScalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "RORS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::ror(vec0, DATA_SET::inputs::inputShiftScalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "RORS(function - RHS scalar)");
    }
    {
        /*SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::ror(DATA_SET::inputs::inputShiftScalarA, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::RORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));*/
        //CHECK_CONDITION((inRange && isUnmodified), "RORS(function - LHS scalar)");
        // TODO: this test requires separate output data
    }
}
    
template<typename VEC_TYPE, typename UINT_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMRORSTest() 
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.ror(mask0, DATA_SET::inputs::inputShiftScalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MRORS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::ror(mask0, vec0, DATA_SET::inputs::inputShiftScalarA);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MRORS(function - RHS scalar)");
    }
    {
        /*SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask0(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::ror(mask0, DATA_SET::inputs::inputShiftScalarA, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, (SCALAR_TYPE*)DATA_SET::outputs::MRORS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));*/
        //CHECK_CONDITION(inRange, "MRORS(function - LHS scalar)");
        // TODO: this test requires separate output data
    }
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
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "NEG");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 =  -vec0;
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::NEG, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "NEG(operator-)");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::neg(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::NEG, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "NEG(function)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMNEGTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.neg(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MNEG, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MNEG");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::neg(mask, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MNEG, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MNEG(function)");
    }
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
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.abs();
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ABS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "ABS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::abs(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ABS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "ABS(function)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMABSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.abs(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MABS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MABS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::abs(mask, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MABS, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MABS(function)");
    }
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
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.sqr();
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SQR, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "SQR");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::sqr(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SQR, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "SQR(function)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSQRTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.sqr(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MSQR, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MSQR");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::sqr(mask, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MSQR, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "MSQR(function)");
    }
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
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.abs();
        VEC_TYPE vec2 = vec1.sqrt();
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SQRT, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "SQRT");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.abs();
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::sqrt(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SQRT, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "SQRT(function)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSQRTTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.abs(mask);
        VEC_TYPE vec2 = vec1.sqrt(mask);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MSQRT, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MSQRT");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.abs(mask);
        VEC_TYPE vec2 = UME::SIMD::FUNCTIONS::sqrt(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MSQRT, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MSQRT");
    }
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
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.round();
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ROUND, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "ROUND");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::round(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::ROUND, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "ROUND(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMROUNDTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.round(mask); 
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MROUND, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MROUND");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::round(mask, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MROUND, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MROUND(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename VEC_INT_TYPE, typename SCALAR_INT_TYPE, int VEC_LEN, typename DATA_SET>
void genericTRUNCTest()
{
    {
        SCALAR_INT_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_INT_TYPE vec1 = vec0.trunc();
        vec1.store(values);
        bool exact = valuesExact(values, DATA_SET::outputs::TRUNC, VEC_LEN);
        CHECK_CONDITION(exact, "TRUNC");
    }
    {
        SCALAR_INT_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_INT_TYPE vec1 = UME::SIMD::FUNCTIONS::trunc(vec0);
        vec1.store(values);
        bool exact = valuesExact(values, DATA_SET::outputs::TRUNC, VEC_LEN);
        CHECK_CONDITION(exact, "TRUNC(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename VEC_INT_TYPE, typename SCALAR_INT_TYPE, int VEC_LEN>
void genericTRUNCTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_INT_TYPE output[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        output[i] = SCALAR_INT_TYPE(inputA[i]);
    }
    {
        SCALAR_INT_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_INT_TYPE vec1 = vec0.trunc();
        vec1.store(values);
        bool exact = valuesExact(values, output, VEC_LEN);
        CHECK_CONDITION(exact, "TRUNC gen");
    }
    {
        SCALAR_INT_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_INT_TYPE vec1 = UME::SIMD::FUNCTIONS::trunc(vec0);
        vec1.store(values);
        bool exact = valuesExact(values, output, VEC_LEN);
        CHECK_CONDITION(exact, "TRUNC(function) gen");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename VEC_INT_TYPE, typename SCALAR_INT_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMTRUNCTest()
{
    {
        SCALAR_INT_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_INT_TYPE vec1 = vec0.trunc(mask);
        vec1.store(values);
        bool exact = valuesExact(values, DATA_SET::outputs::MTRUNC, VEC_LEN);
        CHECK_CONDITION(exact, "MTRUNC");
    }
    {
        SCALAR_INT_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_INT_TYPE vec1 = UME::SIMD::FUNCTIONS::trunc(mask, vec0);
        vec1.store(values);
        bool exact = valuesExact(values, DATA_SET::outputs::MTRUNC, VEC_LEN);
        CHECK_CONDITION(exact, "MTRUNC(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename VEC_INT_TYPE, typename SCALAR_INT_TYPE, typename MASK_TYPE, int VEC_LEN>
void genericMTRUNCTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_INT_TYPE output[VEC_LEN];
    bool maskInput[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        maskInput[i] = randomValue<bool>(gen);

        output[i] = maskInput[i] ? SCALAR_INT_TYPE(inputA[i]) : SCALAR_INT_TYPE(0);
    }

    {
        SCALAR_INT_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        MASK_TYPE mask(maskInput);
        VEC_INT_TYPE vec1 = vec0.trunc(mask);
        vec1.store(values);
        bool exact = valuesExact(values, output, VEC_LEN);
        CHECK_CONDITION(exact, "MTRUNC gen");
    }
    {
        SCALAR_INT_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        MASK_TYPE mask(maskInput);
        VEC_INT_TYPE vec1 = UME::SIMD::FUNCTIONS::trunc(mask, vec0);
        vec1.store(values);
        bool exact = valuesExact(values, output, VEC_LEN);
        CHECK_CONDITION(exact, "MTRUNC(function) gen");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericFLOORTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.floor();
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::FLOOR, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "FLOOR");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::floor(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::FLOOR, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "FLOOR(function)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMFLOORTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.floor(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MFLOOR, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MFLOOR");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::floor(mask, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MFLOOR, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MFLOOR(function)");
    }
}
    
template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericCEILTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.ceil();
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::CEIL, VEC_LEN, 0.05f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "CEIL");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::ceil(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::CEIL, VEC_LEN, 0.05f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "CEIL(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMCEILTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.ceil(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MCEIL, VEC_LEN, 0.05f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MCEIL");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::ceil(mask, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MCEIL, VEC_LEN, 0.05f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MCEIL(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericISFINTest()
{
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask = vec0.isfin();
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::ISFIN, VEC_LEN);
        CHECK_CONDITION(inRange, "ISFIN");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask = UME::SIMD::FUNCTIONS::isfin(vec0);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::ISFIN, VEC_LEN);
        CHECK_CONDITION(inRange, "ISFIN(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericISINFTest()
{
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask = vec0.isinf();
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::ISINF, VEC_LEN);
        CHECK_CONDITION(inRange, "ISINF");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask = UME::SIMD::FUNCTIONS::isinf(vec0);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::ISINF, VEC_LEN);
        CHECK_CONDITION(inRange, "ISINF(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericISANTest()
{
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask = vec0.isan();
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::ISAN, VEC_LEN);
        CHECK_CONDITION(inRange, "ISAN");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask = UME::SIMD::FUNCTIONS::isan(vec0);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::ISAN, VEC_LEN);
        CHECK_CONDITION(inRange, "ISAN(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericISNANTest()
{
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask = vec0.isnan();
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::ISNAN, VEC_LEN);
        CHECK_CONDITION(inRange, "ISNAN");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask = UME::SIMD::FUNCTIONS::isnan(vec0);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::ISNAN, VEC_LEN);
        CHECK_CONDITION(inRange, "ISNAN(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericISNORMTest()
{
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask = vec0.isnorm();
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::ISNORM, VEC_LEN);
        CHECK_CONDITION(inRange, "ISNORM");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask = UME::SIMD::FUNCTIONS::isnorm(vec0);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::ISNORM, VEC_LEN);
        CHECK_CONDITION(inRange, "ISNORM(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericISSUBTest()
{
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask = vec0.issub();
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::ISSUB, VEC_LEN);
        CHECK_CONDITION(inRange, "ISSUB");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask = UME::SIMD::FUNCTIONS::issub(vec0);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::ISSUB, VEC_LEN);
        CHECK_CONDITION(inRange, "ISSUB(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericISZEROTest()
{
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask = vec0.iszero();
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::ISZERO, VEC_LEN);
        CHECK_CONDITION(inRange, "ISZERO");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask = UME::SIMD::FUNCTIONS::iszero(vec0);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::ISZERO, VEC_LEN);
        CHECK_CONDITION(inRange, "ISZERO(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericISZEROSUBTest()
{
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask = vec0.iszerosub();
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::ISZEROSUB, VEC_LEN);
        CHECK_CONDITION(inRange, "ISZEROSUB");
    }
    {
        bool values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask = UME::SIMD::FUNCTIONS::iszerosub(vec0);
        mask.store(values);
        bool inRange = valuesExact(values, DATA_SET::outputs::ISZEROSUB, VEC_LEN);
        CHECK_CONDITION(inRange, "ISZEROSUB(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN>
void genericEXPTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        output[i] = std::exp(inputA[i]);
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1 = vec0.exp();
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "EXP gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1;
        vec1 = UME::SIMD::FUNCTIONS::exp(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "EXP(function) gen");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN>
void genericMEXPTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];
    bool inputMask[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        inputMask[i] = randomValue<bool>(gen);
        output[i] = inputMask[i] ? std::exp(inputA[i]) : inputA[i];
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        MASK_TYPE mask(inputMask);
        VEC_TYPE vec1 = vec0.exp(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "MEXP gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        MASK_TYPE mask(inputMask);
        VEC_TYPE vec1;
        vec1 = UME::SIMD::FUNCTIONS::exp(mask, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "MEXP(function) gen");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN>
void genericLOGTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        output[i] = std::log(inputA[i]);
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1 = vec0.log();
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "LOG gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1;
        vec1 = UME::SIMD::FUNCTIONS::log(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "LOG(function) gen");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN>
void genericLOG2Test_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        output[i] = std::log2(inputA[i]);
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1 = vec0.log2();
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "LOG2 gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1;
        vec1 = UME::SIMD::FUNCTIONS::log2(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "LOG2(function) gen");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN>
void genericLOG10Test_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        output[i] = std::log10(inputA[i]);
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1 = vec0.log10();
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "LOG10 gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1;
        vec1 = UME::SIMD::FUNCTIONS::log10(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "LOG10(function) gen");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericSINTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.sin();
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SIN, VEC_LEN, 0.1f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "SIN");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::sin(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::SIN, VEC_LEN, 0.1f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "SIN(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN>
void genericSINTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        output[i] = std::sin(inputA[i]);
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1 = vec0.sin();
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "SIN gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1;
        vec1 = UME::SIMD::FUNCTIONS::sin(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "SIN(function) gen");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMSINTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.sin(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MSIN, VEC_LEN, 0.1f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MSIN");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::sin(mask, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MSIN, VEC_LEN, 0.1f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MSIN(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericCOSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.cos();
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::COS, VEC_LEN, 0.1f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "COS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::cos(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::COS, VEC_LEN, 0.1f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "COS(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN>
void genericCOSTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        output[i] = std::cos(inputA[i]);
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1 = vec0.cos();
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "COS gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1;
        vec1 = UME::SIMD::FUNCTIONS::cos(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "COS(function) gen");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMCOSTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.cos(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MCOS, VEC_LEN, 0.1f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MCOS");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::cos(mask, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MCOS, VEC_LEN, 0.1f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MCOS(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericTANTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.tan();
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::TAN, VEC_LEN, 0.1f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "TAN");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::tan(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::TAN, VEC_LEN, 0.1f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION(inRange, "TAN(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN>
void genericTANTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        output[i] = std::tan(inputA[i]);
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1 = vec0.tan();
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "TAN gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1;
        vec1 = UME::SIMD::FUNCTIONS::tan(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "TAN(function) gen");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMTANTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.tan(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MTAN, VEC_LEN, 0.1f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MTAN");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::tan(mask, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MTAN, VEC_LEN, 0.1f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MTAN(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN, typename DATA_SET>
void genericCTANTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = vec0.ctan();
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::CTAN, VEC_LEN, 0.1f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "CTAN");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::ctan(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::CTAN, VEC_LEN, 0.1f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "CTAN(function)");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, int VEC_LEN>
void genericCTANTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    SCALAR_TYPE inputA[VEC_LEN];
    SCALAR_TYPE output[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<SCALAR_TYPE>(gen);
        output[i] = SCALAR_TYPE(1.0f)/std::tan(inputA[i]);
    }

    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1 = vec0.ctan();
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "CTAN gen");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(inputA);
        VEC_TYPE vec1;
        vec1 = UME::SIMD::FUNCTIONS::ctan(vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, output, VEC_LEN, SCALAR_TYPE(0.01f));
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange & isUnmodified), "CTAN(function) gen");
    }
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMCTANTest()
{
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = vec0.ctan(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MCTAN, VEC_LEN, 0.1f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MCTAN");
    }
    {
        SCALAR_TYPE values[VEC_LEN];
        VEC_TYPE vec0(DATA_SET::inputs::inputA);
        MASK_TYPE mask(DATA_SET::inputs::maskA);
        VEC_TYPE vec1 = UME::SIMD::FUNCTIONS::ctan(mask, vec0);
        vec1.store(values);
        bool inRange = valuesInRange(values, DATA_SET::outputs::MCTAN, VEC_LEN, 0.1f);
        vec0.store(values);
        bool isUnmodified = valuesInRange(values, DATA_SET::inputs::inputA, VEC_LEN, SCALAR_TYPE(0.01f));
        CHECK_CONDITION((inRange && isUnmodified), "MCTAN(function)");
    }
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

template<typename FLOAT_VEC_TYPE, typename FLOAT_SCALAR_TYPE, typename UINT_VEC_TYPE, typename UINT_SCALAR_TYPE, int VEC_LEN>
void genericFTOUTest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    FLOAT_SCALAR_TYPE inputA[VEC_LEN];
    UINT_SCALAR_TYPE output[VEC_LEN];
    
    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<FLOAT_SCALAR_TYPE>(gen);
        output[i] = UINT_SCALAR_TYPE(inputA[i]);
    }
    {
        UINT_SCALAR_TYPE values[VEC_LEN];
        FLOAT_VEC_TYPE vec0(inputA);
        UINT_VEC_TYPE vec1 = UINT_VEC_TYPE(vec0);
        vec1.store(values);
        bool exact = valuesExact(values, output, VEC_LEN);
        CHECK_CONDITION(exact, "FTOU gen");
    }
    //EXTEND THIS TEST WITH FIXED RANGE THAT CAN BE REPRESENTED AS UNSINGED
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

template<typename FLOAT_VEC_TYPE, typename FLOAT_SCALAR_TYPE, typename INT_VEC_TYPE, typename INT_SCALAR_TYPE, int VEC_LEN>
void genericFTOITest_random()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    FLOAT_SCALAR_TYPE inputA[VEC_LEN];
    INT_SCALAR_TYPE output[VEC_LEN];

    for (int i = 0; i < VEC_LEN; i++) {
        inputA[i] = randomValue<FLOAT_SCALAR_TYPE>(gen);
        output[i] = INT_SCALAR_TYPE(inputA[i]);
    }
    {
        INT_SCALAR_TYPE values[VEC_LEN];
        FLOAT_VEC_TYPE vec0(inputA);
        INT_VEC_TYPE vec1 = INT_VEC_TYPE(vec0);
        vec1.store(values);
        bool exact = valuesExact(values, output, VEC_LEN);
        CHECK_CONDITION(exact, "FTOI gen");
    }
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
void genericDEGRADETest()
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
    CHECK_CONDITION(inRange, "DEGRADE");
}

template<typename VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericBaseInterfaceTest()
{   
    genericINSERTTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericEXTRACTTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericASSIGNVTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMASSIGNVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericASSIGNSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMASSIGNSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    // PREFETCH0
    // PREFETCH1
    // PREFETCH2
    genericLOAD_STORETest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMLOADTest_random<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN>();
    genericMSTORETest_random<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN>();
    genericLOADA_STOREATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMLOADATest_random<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN>();
    genericMSTOREATest_random<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN>();
    // SWIZZLE
    // SWIZZLEA
    genericBLENDVTest_random<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN>();
    genericADDVTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericADDVTest_random<VEC_TYPE, SCALAR_TYPE, VEC_LEN>();
    genericMADDVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericMADDVTest_random<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN>();
    genericADDSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericADDSTest_random<VEC_TYPE, SCALAR_TYPE, VEC_LEN>();
    genericMADDSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericMADDSTest_random<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN>();
    genericADDVATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMADDVATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericADDSATest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericMADDSATest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericSADDVTest_random<VEC_TYPE, SCALAR_TYPE, VEC_LEN>();
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
    genericSUBVTest_random<VEC_TYPE, SCALAR_TYPE, VEC_LEN>();
    genericMSUBVTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericMSUBVTest_random<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN>();
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
    //genericHMULTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericHMULTest_random<VEC_TYPE, SCALAR_TYPE, VEC_LEN>();
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
void genericIntegerInterfaceTest()
{
    genericREMVTest_random<VEC_TYPE, SCALAR_TYPE, VEC_LEN>();
    genericMREMVTest_random<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN>();
    genericREMSTest_random<VEC_TYPE, SCALAR_TYPE, VEC_LEN>();

    genericLANDVTest_random<VEC_TYPE, SCALAR_TYPE, VEC_LEN>();
    genericLORVTest_random<VEC_TYPE, SCALAR_TYPE, VEC_LEN>();

    // Bitwise interface tests
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
    //genericTRUNCTest<VEC_TYPE, SCALAR_TYPE, VEC_INT_TYPE, SCALAR_INT_TYPE, VEC_LEN, DATA_SET>();
    genericTRUNCTest_random<VEC_TYPE, SCALAR_TYPE, VEC_INT_TYPE, SCALAR_INT_TYPE, VEC_LEN>();
    //genericMTRUNCTest<VEC_TYPE, SCALAR_TYPE, VEC_INT_TYPE, SCALAR_INT_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    genericMTRUNCTest_random<VEC_TYPE, SCALAR_TYPE, VEC_INT_TYPE, SCALAR_INT_TYPE, MASK_TYPE, VEC_LEN>();
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
    
    genericEXPTest_random<VEC_TYPE, SCALAR_TYPE, VEC_LEN>();
    genericMEXPTest_random<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN>();
    genericLOGTest_random<VEC_TYPE, SCALAR_TYPE, VEC_LEN>();
    genericLOG2Test_random<VEC_TYPE, SCALAR_TYPE, VEC_LEN>();
    genericLOG10Test_random<VEC_TYPE, SCALAR_TYPE, VEC_LEN>();

    //genericSINTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericSINTest_random<VEC_TYPE, SCALAR_TYPE, VEC_LEN>();
    //genericMSINTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    //genericCOSTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericCOSTest_random<VEC_TYPE, SCALAR_TYPE, VEC_LEN>();
    //genericMCOSTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    //genericTANTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericTANTest_random<VEC_TYPE, SCALAR_TYPE, VEC_LEN>();
    //genericMTANTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
    //genericCTANTest<VEC_TYPE, SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericCTANTest_random<VEC_TYPE, SCALAR_TYPE, VEC_LEN>();
    //genericMCTANTest<VEC_TYPE, SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
}

template<typename MASK_TYPE, int VEC_LEN, typename DATA_SET>
void genericMaskTest() {
    genericLANDVTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLANDVTest_random<MASK_TYPE, bool, VEC_LEN>();
    genericLANDSTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLANDVATest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLANDSATest<MASK_TYPE, VEC_LEN, DATA_SET>();
    genericLORVTest<MASK_TYPE, VEC_LEN, DATA_SET> ();
    genericLORVTest_random<MASK_TYPE, bool, VEC_LEN>();
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
    genericIntegerInterfaceTest<UINT_VEC_TYPE, UINT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
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
    genericIntegerInterfaceTest<UINT_VEC_TYPE, UINT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
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
    genericIntegerInterfaceTest<INT_VEC_TYPE, INT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET> ();
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
    genericIntegerInterfaceTest<INT_VEC_TYPE, INT_SCALAR_TYPE, MASK_TYPE, VEC_LEN, DATA_SET>();
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
    //genericFTOUTest<FLOAT_VEC_TYPE, UINT_VEC_TYPE, UINT_SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericFTOUTest_random<FLOAT_VEC_TYPE, FLOAT_SCALAR_TYPE, UINT_VEC_TYPE, UINT_SCALAR_TYPE, VEC_LEN>();
    //genericFTOITest<FLOAT_VEC_TYPE, INT_VEC_TYPE, INT_SCALAR_TYPE, VEC_LEN, DATA_SET>();
    genericFTOITest_random<FLOAT_VEC_TYPE, FLOAT_SCALAR_TYPE, INT_VEC_TYPE, INT_SCALAR_TYPE, VEC_LEN>();
}

#endif
