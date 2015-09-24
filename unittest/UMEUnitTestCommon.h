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

#ifndef UME_UNIT_TEST_COMMON_H_
#define UME_UNIT_TEST_COMMON_H_

#include <iostream>

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
#define UME_PI_D 3.1415926535897932384626433832795028841971693993751058209749445923078164062
#define UME_2PI_D (2.0*UME_PI_D)
#define UME_PI_F float(UME_PI_D)
#define UME_2PI_F (2.0f*UME_PI_F)

bool valueInRange(float value, float expectedValue, float errMargin) {
    if(value > 0.0f)
    {
        return ((expectedValue)*(1.0f + errMargin) > value) 
             & ((expectedValue)*(1.0f - errMargin) < value);
    }
    else
    {
        return ((expectedValue)*(1.0f + errMargin) < value)
             & ((expectedValue)*(1.0f - errMargin) > value);
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

bool valuesInRange(float *values, float *expectedValues, unsigned int count, float errMargin)
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

bool valuesInRange(double *values, double *expectedValues, unsigned int count, double errMargin)
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


#endif
