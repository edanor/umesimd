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
    if(value >= 0.0f)
    {
        return ((expectedValue)*(1.0f + errMargin) >= value) 
             & ((expectedValue)*(1.0f - errMargin) <= value);
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

float g_Init1_l[32] = { 
    1.0f,       1243.0f,        -19.123f,       -581.98f, 
    14141.87f,  10948.74f,      0.4187f,        0.0013f, 
    3.12f,      -12387.84f,     122.23f,        99.2981f, 
    3.09f,      89.123f,        84.44f,         -74.12f, 
    1948.91f,   908.000023f,    91.76f,         91236.132f, 
    99913.0f,   913.9184f,      -134918.1319f,  -813981.197f, 
    0.1841636f, 4191.941f,      9814.1947f,     8146666.817f, 
    61809.0f,   989613.32f,     -94871.2f,      99999.99999f};

float g_Init1_r[32] = { 
    90871.1837f,    193.12f,    -87591.32f, 8712366.953f, 
    7381.183f,      95651.13f, -964.12f,    98713.86513f, 
    111.94f,        9995.61f,   975.912f,   75839.1863f, 
    94.98f,         8741.94f,   9999.44f,   10000.00001f, 
    0.000001f,      194.437f,   4917.91f,   97512.91487f, 
    888131.123f,    471.1847f,  9851214.0f, -165387.147f, 
    33667.132f,     71598.231f, -9851.99f,  9913.512f, 
    94.1381f,       9841.12f,   981.31f,    78132.947f};
                               
// Pre-computed results of add operation of g_Init1_l and g_Init1_r
float g_addRes1[32] = {
    90872.1837f,    1436.12f,       -87610.443f,    8711784.973f, 
    21523.053f,     106599.87f,     -963.7013f,     98713.86643f, 
    115.06f,        -2392.23f,      1098.142f,      75938.4844f, 
    98.07f,         8831.063f,      10083.88f,      9925.88001f, 
    1948.910001f,   1102.437023f,   5009.67f,       188749.0469f, 
    988044.123f,    1385.1031f,     9716295.868f,   -979368.344f, 
    33667.31616f,   75790.172f,     -37.7953f,      8156580.329f, 
    61903.1381f,    999454.44f,     -93889.89f,     178132.947f}; 

float g_subRes1[32] = {
    -90870.1837f,   1049.88f,       87572.197f,     -8712948.933f, 
    6760.687f,      -84702.39f,     964.5387f,      -98713.86383f, 
    -108.82f,       -22383.45f,     -853.682f,      -75739.8882f, 
    -91.89f,        -8652.817f,     -9915.0f,       -10074.12001f, 
    1948.909999f,   713.563023f,    -4826.15f,      -6276.78287f, 
    -788218.123f,   442.7337f,      -9986132.132f,  -648594.05f, 
    -33666.94784f,  -67406.29f,     19666.1847f,    8136753.305f, 
    61714.8619f,    979772.2f,      -95852.51f,     21867.05299f };

float g_mulRes1[32] = {
    90871.18f,          240048.16f,     1675008.81f,        -5070423319.31f, 
    104383730.43f,      1047259353.08f, -403.68f,           128.33f, 
    349.25f,            -123824017.38f, 119285.72f,         7530687.11f, 
    293.49f,            779107.92f,     844352.71f,         -741200.00f, 
    0.00f,              176548.80f,     451267.42f,         8896701172.78f, 
    88735844892.30f,    430624.37f,     -1329107389518.10f, 134622027883.48f, 
    6200.26f,           300135560.06f,  -96689348.04f,      80762079253.31f, 
    5818581.82f,        9738903435.72f, -93098057.27f,      7813294699.22f};

float g_divRes1[32] = {
    1.10046E-05f,       6.436412593f,    0.000218321f,      -6.67993E-05f,
    1.915935427f,       0.114465349f,    -0.000434282f,     1.31694E-08f,
    0.027872074f,       -1.239328065f,   0.125246948f,      0.001309324f,
    0.032533165f,       0.010194877f,    0.008444473f,      -0.007412f,
    1948910000.0f,      4.669893194f,    0.018658333f,      0.935631266f,
    0.112498028f,       1.939618158f,    -0.013695584f,     4.921671434f,
    5.47013E-06f,       0.058548109f,    -0.996163689f,     821.7740411f,
    656.5779424f,       100.5590136f,    -96.67811395f,     1.279869809f};


#endif
