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
//  This piece of code was developed as part of ICE-DIP project at CERN.
//  "ICE-DIP is a European Industrial Doctorate project funded by the European Community's 
//  7th Framework programme Marie Curie Actions under grant PITN-GA-2012-316596".
//

#ifndef EXPLOG_SCALAR_H_
#define EXPLOG_SCALAR_H_

template<typename SCALAR_FLOAT_T>
UME_NEVER_INLINE void generate_some_exp_values(int N, SCALAR_FLOAT_T * in, SCALAR_FLOAT_T * out) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<SCALAR_FLOAT_T> dist(std::numeric_limits<SCALAR_FLOAT_T>::lowest(), std::numeric_limits<SCALAR_FLOAT_T>::max());

    for (int i = 0; i < N; i++) {
        in[i] = dist(gen);
        out[i] = std::exp(in[i]);
    }
}

template<>
UME_NEVER_INLINE void generate_some_exp_values<float> (int N, float * in, float * out) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // For 32b floating point numbers maximum value reachable is 3.4028235E38 from which natural logarithm is ~88.72. Using 
    // higher values would cause us to hit the infinities and then destroy our benchmarking environment.
    // Limit the argument to fall between 0 and 88.
    std::uniform_real_distribution<float> dist(0.0f, 88.0f);

    for (int i = 0; i < N; i++) {
        in[i] = dist(gen);
        out[i] = std::exp(in[i]);
    }
}

template<>
UME_NEVER_INLINE void generate_some_exp_values<double> (int N, double * in, double * out) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // For 64b floating point numbers maximum value reachable is 1.79769313486231570814527423732E308 from which natural logarithm is ~709.78. Using 
    // higher values would cause us to hit the infinities and then destroy our benchmarking environment.
    // Limit the argument to fall between 0 and 709.
    std::uniform_real_distribution<double> dist(std::numeric_limits<double>::lowest(), 709.0);

    for (int i = 0; i < N; i++) {
        in[i] = dist(gen);
        out[i] = std::exp(in[i]);
    }
}

template<typename SCALAR_FLOAT_T>
UME_NEVER_INLINE void generate_some_log_values(int N, SCALAR_FLOAT_T * in, SCALAR_FLOAT_T * out) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<SCALAR_FLOAT_T> dist(std::numeric_limits<SCALAR_FLOAT_T>::min(), std::numeric_limits<SCALAR_FLOAT_T>::max());

    for (int i = 0; i < N; i++) {
        in[i] = dist(gen);
        out[i] = std::log(in[i]);
    }
}

template<typename SCALAR_FLOAT_T>
UME_NEVER_INLINE void generate_some_log2_values(int N, SCALAR_FLOAT_T * in, SCALAR_FLOAT_T * out) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<SCALAR_FLOAT_T> dist(std::numeric_limits<SCALAR_FLOAT_T>::min(), std::numeric_limits<SCALAR_FLOAT_T>::max());

    for (int i = 0; i < N; i++) {
        in[i] = dist(gen);
        out[i] = std::log2(in[i]);
    }
}

template<typename SCALAR_FLOAT_T>
UME_NEVER_INLINE void generate_some_log10_values(int N, SCALAR_FLOAT_T * in, SCALAR_FLOAT_T * out) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<SCALAR_FLOAT_T> dist(std::numeric_limits<SCALAR_FLOAT_T>::min(), std::numeric_limits<SCALAR_FLOAT_T>::max());

    for (int i = 0; i < N; i++) {
        in[i] = dist(gen);
        out[i] = std::log10(in[i]);
    }
}

// Kernel for benchmarking using std::exp function calls.
template<typename SCALAR_FLOAT_T>
UME_NEVER_INLINE benchmark_results<SCALAR_FLOAT_T> test_exp_scalar(const int ARRAY_SIZE)
{
    unsigned long long start, end;    // Time measurements

    const int LEN = ARRAY_SIZE;
    SCALAR_FLOAT_T* input = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);
    SCALAR_FLOAT_T* output = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);
    SCALAR_FLOAT_T* values = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);

    generate_some_exp_values<SCALAR_FLOAT_T>(LEN, input, output);

    start = get_timestamp();

    for (int i = 0; i < LEN; i++) {
        values[i] = std::exp(input[i]);
    }

    end = get_timestamp();

    SCALAR_FLOAT_T max_err = 0;

    for (int i = 0; i < LEN; i++) {
        if(!(values[i] == std::numeric_limits<SCALAR_FLOAT_T>::infinity() && output[i] == std::numeric_limits<SCALAR_FLOAT_T>::infinity()) &&
           !(values[i] == -std::numeric_limits<SCALAR_FLOAT_T>::infinity() && output[i] == -std::numeric_limits<SCALAR_FLOAT_T>::infinity()) )
        {
    
            SCALAR_FLOAT_T next = NEXT_AFTER(values[i], HUGE_VALUE<SCALAR_FLOAT_T>());
            SCALAR_FLOAT_T reference_value_ulp = std::abs(next - values[i]);
            SCALAR_FLOAT_T error_ulp = (values[i] - output[i]) / reference_value_ulp;

            if (max_err < std::abs(error_ulp)) max_err = std::abs(error_ulp);
            /*if (output[i] != values[i])
                std::cout << " Difference in exp[" << i << "]: " << values[i]
                << " should be: " << output[i]
                << " error(ulp): " << error_ulp << std::endl;*/
        }
    }

    benchmark_results<SCALAR_FLOAT_T> result;
    result.elapsedTime = end - start;
    result.error_ulp = max_err;
    result.error_ulp = max_err;

    UME::DynamicMemory::AlignedFree(input);
    UME::DynamicMemory::AlignedFree(output);
    UME::DynamicMemory::AlignedFree(values);

    return result;
}

// Kernel for benchmarking using std::exp function calls.
template<typename SCALAR_FLOAT_T>
UME_NEVER_INLINE benchmark_results<SCALAR_FLOAT_T> test_log_scalar(int ARRAY_SIZE)
{
    unsigned long long start, end;    // Time measurements

    const int LEN = ARRAY_SIZE;
    SCALAR_FLOAT_T* input = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);
    SCALAR_FLOAT_T* output = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);
    SCALAR_FLOAT_T* values = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);

    generate_some_log_values<SCALAR_FLOAT_T>(LEN, input, output);

    start = get_timestamp();

    for (int i = 0; i < LEN; i++) {
        values[i] = std::log(input[i]);
    }

    end = get_timestamp();

    SCALAR_FLOAT_T max_err = 0;

    for (int i = 0; i < LEN; i++) {
        if(!(values[i] == std::numeric_limits<SCALAR_FLOAT_T>::infinity() && values[i] == std::numeric_limits<SCALAR_FLOAT_T>::infinity()) &&
           !(values[i] == -std::numeric_limits<SCALAR_FLOAT_T>::infinity() && values[i] == -std::numeric_limits<SCALAR_FLOAT_T>::infinity()) )
        {
    
            SCALAR_FLOAT_T next = NEXT_AFTER(values[i], HUGE_VALUE<SCALAR_FLOAT_T>());
            SCALAR_FLOAT_T reference_value_ulp = std::abs(next - values[i]);
            SCALAR_FLOAT_T error_ulp = (values[i] - output[i]) / reference_value_ulp;

            if (max_err < std::abs(error_ulp)) max_err = std::abs(error_ulp);
            /*if (output[i] != values[i])
                std::cout << " Difference in exp[" << i << "]: " << values[i]
                << " should be: " << output[i]
                << " error(ulp): " << error_ulp << std::endl;*/
        }
    }

    benchmark_results<SCALAR_FLOAT_T> result;
    result.elapsedTime = end - start;
    result.error_ulp = max_err;
    result.error_ulp = max_err;

    UME::DynamicMemory::AlignedFree(input);
    UME::DynamicMemory::AlignedFree(output);
    UME::DynamicMemory::AlignedFree(values);

    return result;
}

// Kernel for benchmarking using std::exp function calls.
template<typename SCALAR_FLOAT_T>
UME_NEVER_INLINE benchmark_results<SCALAR_FLOAT_T> test_log2_scalar(int ARRAY_SIZE)
{
    unsigned long long start, end;    // Time measurements

    const int LEN = ARRAY_SIZE;
    SCALAR_FLOAT_T* input = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);
    SCALAR_FLOAT_T* output = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);
    SCALAR_FLOAT_T* values = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);

    generate_some_log2_values<SCALAR_FLOAT_T>(LEN, input, output);

    start = get_timestamp();

    for (int i = 0; i < LEN; i++) {
        values[i] = std::log2(input[i]);
    }

    end = get_timestamp();

    SCALAR_FLOAT_T max_err = 0;

    for (int i = 0; i < LEN; i++) {
        if(!(values[i] == std::numeric_limits<SCALAR_FLOAT_T>::infinity() && values[i] == std::numeric_limits<SCALAR_FLOAT_T>::infinity()) &&
           !(values[i] == -std::numeric_limits<SCALAR_FLOAT_T>::infinity() && values[i] == -std::numeric_limits<SCALAR_FLOAT_T>::infinity()) )
        {
    
            SCALAR_FLOAT_T next = NEXT_AFTER(values[i], HUGE_VALUE<SCALAR_FLOAT_T>());
            SCALAR_FLOAT_T reference_value_ulp = std::abs(next - values[i]);
            SCALAR_FLOAT_T error_ulp = (values[i] - output[i]) / reference_value_ulp;

            if (max_err < std::abs(error_ulp)) max_err = std::abs(error_ulp);
            /*if (output[i] != values[i])
                std::cout << " Difference in exp[" << i << "]: " << values[i]
                << " should be: " << output[i]
                << " error(ulp): " << error_ulp << std::endl;*/
        }
    }

    benchmark_results<SCALAR_FLOAT_T> result;
    result.elapsedTime = end - start;
    result.error_ulp = max_err;
    result.error_ulp = max_err;

    UME::DynamicMemory::AlignedFree(input);
    UME::DynamicMemory::AlignedFree(output);
    UME::DynamicMemory::AlignedFree(values);

    return result;
}

// Kernel for benchmarking using std::exp function calls.
template<typename SCALAR_FLOAT_T>
UME_NEVER_INLINE benchmark_results<SCALAR_FLOAT_T> test_log10_scalar(int ARRAY_SIZE)
{
    unsigned long long start, end;    // Time measurements

    const int LEN = ARRAY_SIZE;
    SCALAR_FLOAT_T* input = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);
    SCALAR_FLOAT_T* output = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);
    SCALAR_FLOAT_T* values = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);

    generate_some_log10_values<SCALAR_FLOAT_T>(LEN, input, output);

    start = get_timestamp();

    for (int i = 0; i < LEN; i++) {
        values[i] = std::log10(input[i]);
    }

    end = get_timestamp();

    SCALAR_FLOAT_T max_err = 0;

    for (int i = 0; i < LEN; i++) {
        if(!(values[i] == std::numeric_limits<SCALAR_FLOAT_T>::infinity() && values[i] == std::numeric_limits<SCALAR_FLOAT_T>::infinity()) &&
           !(values[i] == -std::numeric_limits<SCALAR_FLOAT_T>::infinity() && values[i] == -std::numeric_limits<SCALAR_FLOAT_T>::infinity()) )
        {
    
            SCALAR_FLOAT_T next = NEXT_AFTER(values[i], HUGE_VALUE<SCALAR_FLOAT_T>());
            SCALAR_FLOAT_T reference_value_ulp = std::abs(next - values[i]);
            SCALAR_FLOAT_T error_ulp = (values[i] - output[i]) / reference_value_ulp;

            if (max_err < std::abs(error_ulp)) max_err = std::abs(error_ulp);
            /*if (output[i] != values[i])
                std::cout << " Difference in exp[" << i << "]: " << values[i]
                << " should be: " << output[i]
                << " error(ulp): " << error_ulp << std::endl;*/
        }
    }

    benchmark_results<SCALAR_FLOAT_T> result;
    result.elapsedTime = end - start;
    result.error_ulp = max_err;
    result.error_ulp = max_err;

    UME::DynamicMemory::AlignedFree(input);
    UME::DynamicMemory::AlignedFree(output);
    UME::DynamicMemory::AlignedFree(values);

    return result;
}
#endif
