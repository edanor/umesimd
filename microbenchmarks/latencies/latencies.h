#ifndef LATENCIES_H_
#define LATENCIES_H_

#include "../../UMESimd.h"

using namespace UME::SIMD;

// define RDTSC getter function
#if defined(__i386__)
static __inline__ unsigned long long __rdtsc(void)
{
    unsigned long long int x;
    __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
    return x;
}
#elif defined(__x86_64__)
static __inline__ unsigned long long __rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}
#endif

template<typename T>
T getRandomValue() {
    T value = T((std::numeric_limits<T>::max() - 1) * (float(rand()) / float(RAND_MAX)) + 1.0f);
    //std::cout << " " << uint32_t(value) << " ";
    return value;
}
template<>
bool getRandomValue() {
    return getRandomValue<uint8_t>() > 128 ? true : false;
}

const int ITERATIONS = 1000;

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0, vec1, vec2;
//   vec2 = vec0.<MFI_FUNCTION>(vec1);
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
    alignas(VEC_T::alignment()) SCALAR_T raw2[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
            raw2[k] = getRandomValue<SCALAR_T>(); \
        } \
 \
        VEC_T vec0(raw1); \
        VEC_T vec1(raw2); \
        VEC_T res; \
 \
        start = __rdtsc(); \
            res.assign(vec0.MFI_name(vec1)); \
        end = __rdtsc(); \
 \
        volatile SCALAR_T x = res.hadd(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0, vec1, vec2;
//   MASK_T mask;
//   vec2 = vec0.<MFI_FUNCTION>(mask, vec1);
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_MASK_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T MASK_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
    alignas(VEC_T::alignment()) SCALAR_T raw2[VEC_LEN]; \
    bool mask_raw[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
            raw2[k] = getRandomValue<SCALAR_T>(); \
            mask_raw[k] = float(rand()) / float(RAND_MAX) > 0.5f ? true : false; \
        } \
 \
        VEC_T vec0(raw1); \
        VEC_T vec1(raw2); \
        MASK_T mask(mask_raw); \
        VEC_T res; \
 \
        start = __rdtsc(); \
            res.assign(vec0.MFI_name(mask, vec1)); \
        end = __rdtsc(); \
 \
        volatile SCALAR_T x = res.hadd(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0, vec1;
//   SCALAR_T scalarOp;
//   vec1 = vec0.<MFI_FUNCTION>(scalarOp);
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_SCALAR_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
        } \
        SCALAR_T scalarOp = getRandomValue<SCALAR_T>(); \
 \
        VEC_T vec0(raw1); \
        VEC_T res; \
 \
        start = __rdtsc(); \
            res.assign(vec0.MFI_name(scalarOp)); \
        end = __rdtsc(); \
 \
        volatile SCALAR_T x = res.hadd(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0, vec1;
//   SCALAR_T scalarOp;
//   MASK_T mask;
//   vec1 = vec0.<MFI_FUNCTION>(mask, scalarOp);
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_MASK_SCALAR_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T MASK_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
    bool mask_raw[VEC_LEN]; \
 \
    unsigned long long sum = 0; \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
            mask_raw[k] = getRandomValue<bool>(); \
        } \
        SCALAR_T scalarOp = getRandomValue<SCALAR_T>(); \
 \
        VEC_T vec0(raw1); \
        MASK_T mask(mask_raw); \
        VEC_T res; \
 \
        start = __rdtsc(); \
            res.assign(vec0.MFI_name(mask, scalarOp)); \
        end = __rdtsc(); \
 \
        volatile SCALAR_T x = res.hadd(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
        sum += delta; \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}


// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0, vec1;
//   vec0.<MFI_FUNCTION>(vec1);
//
//  where MFI_FUNCTION is an in-place operation (e.g. ADDVA)
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_ASSIGN_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
    alignas(VEC_T::alignment()) SCALAR_T raw2[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
            raw2[k] = getRandomValue<SCALAR_T>(); \
        } \
 \
        VEC_T vec0(raw1); \
        VEC_T vec1(raw2); \
        VEC_T res; \
 \
        start = __rdtsc(); \
            vec0.MFI_name(vec1); \
        end = __rdtsc(); \
 \
        volatile SCALAR_T x = vec0.hadd(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0, vec1, vec2;
//   MASK_T mask;
//   vec0.<MFI_FUNCTION>(mask, vec1);
//
//  where MFI_FUNCTION is an in-place operation (e.g. MADDVA)
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_MASK_ASSIGN_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T MASK_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
    alignas(VEC_T::alignment()) SCALAR_T raw2[VEC_LEN]; \
    bool mask_raw[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
            raw2[k] = getRandomValue<SCALAR_T>(); \
            mask_raw[k] = float(rand()) / float(RAND_MAX) > 0.5f ? true : false; \
        } \
 \
        VEC_T vec0(raw1); \
        VEC_T vec1(raw2); \
        MASK_T mask(mask_raw); \
 \
        start = __rdtsc(); \
            vec0.MFI_name(mask, vec1); \
        end = __rdtsc(); \
 \
        volatile SCALAR_T x = vec0.hadd(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0;
//   SCALAR_T scalarOp;
//   vec0.<MFI_FUNCTION>(scalarOp);
//
//  where MFI_FUNCTION is an in-place operation (e.g. ADDSA)
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_SCALAR_ASSIGN_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
        } \
        SCALAR_T scalarOp = getRandomValue<SCALAR_T>(); \
 \
        VEC_T vec0(raw1); \
 \
        start = __rdtsc(); \
            vec0.MFI_name(scalarOp); \
        end = __rdtsc(); \
 \
        volatile SCALAR_T x = vec0.hadd(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0;
//   SCALAR_T scalarOp;
//   MASK_T mask;
//   vec0.<MFI_FUNCTION>(mask, scalarOp);
//
//  where MFI_FUNCTION is an in-place operation (e.g. MADDSA)
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_MASK_SCALAR_ASSIGN_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T MASK_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
    bool mask_raw[VEC_LEN]; \
 \
    unsigned long long sum = 0; \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
            mask_raw[k] = getRandomValue<bool>(); \
        } \
        SCALAR_T scalarOp = getRandomValue<SCALAR_T>(); \
 \
        VEC_T vec0(raw1); \
        MASK_T mask(mask_raw); \
 \
        start = __rdtsc(); \
            vec0.MFI_name(mask, scalarOp); \
        end = __rdtsc(); \
 \
        volatile SCALAR_T x = vec0.hadd(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
        sum += delta; \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0, vec1;
//   vec1 = vec0.<MFI_FUNCTION>();
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_UNARY_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
        } \
 \
        VEC_T vec0(raw1); \
        VEC_T res; \
 \
        start = __rdtsc(); \
            res.assign(vec0.MFI_name()); \
        end = __rdtsc(); \
 \
        volatile SCALAR_T x = res.hadd(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0, vec1;
//   MASK_T mask;
//   vec1 = vec0.<MFI_FUNCTION>(mask);
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_UNARY_MASK_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T MASK_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
    bool mask_raw[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
            mask_raw[k] = getRandomValue<bool>(); \
        } \
 \
        VEC_T vec0(raw1); \
        VEC_T res; \
        MASK_T mask(mask_raw); \
 \
        start = __rdtsc(); \
            res.assign(vec0.MFI_name(mask)); \
        end = __rdtsc(); \
 \
        volatile SCALAR_T x = res.hadd(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0;
//   vec0.<MFI_FUNCTION>();
//
//  where MFI_FUNCTION is an in-place operation (e.g. RCPA)
// 
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_UNARY_ASSIGN_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
        } \
 \
        VEC_T vec0(raw1); \
 \
        start = __rdtsc(); \
            vec0.MFI_name(); \
        end = __rdtsc(); \
 \
        volatile SCALAR_T x = vec0.hadd(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0;
//   vec0.<MFI_FUNCTION>();
//
//  where MFI_FUNCTION is an in-place operation (e.g. MRCPA)
// 
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_UNARY_MASK_ASSIGN_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T MASK_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
    bool mask_raw[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
            mask_raw[k] = getRandomValue<bool>(); \
        } \
 \
        VEC_T vec0(raw1); \
        MASK_T mask(mask_raw); \
 \
        start = __rdtsc(); \
            vec0.MFI_name(mask); \
        end = __rdtsc(); \
 \
        volatile SCALAR_T x = vec0.hadd(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}


// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0, vec1;
//   MASK_T mask;
//   mask = vec0.<MFI_FUNCTION>(vec1);
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_PREDICATE_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T MASK_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
    alignas(VEC_T::alignment()) SCALAR_T raw2[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
            raw2[k] = getRandomValue<SCALAR_T>(); \
        } \
 \
        VEC_T vec0(raw1); \
        VEC_T vec1(raw2); \
        MASK_T mask; \
 \
        start = __rdtsc(); \
            mask.assign(vec0.MFI_name(vec1)); \
        end = __rdtsc(); \
 \
        volatile SCALAR_T x = mask.hlxor(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0, vec1;
//   MASK_T mask;
//   mask = vec0.<MFI_FUNCTION>(vec1);
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_PREDICATE_SCALAR_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T MASK_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
        } \
 \
        VEC_T vec0(raw1); \
        SCALAR_T scalarOp = getRandomValue<SCALAR_T>(); \
        MASK_T mask; \
 \
        start = __rdtsc(); \
            mask.assign(vec0.MFI_name(scalarOp)); \
        end = __rdtsc(); \
 \
        volatile SCALAR_T x = mask.hlxor(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0;
//   SCALAR_T scalar1;
//   scalar1 = vec0.<MFI_FUNCTION>();
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_REDUCTION_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
        } \
 \
        VEC_T vec0(raw1); \
 \
        start = __rdtsc(); \
            volatile auto res = vec0.MFI_name(); \
        end = __rdtsc(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0;
//   SCALAR_T scalar1;
//   MASK_T mask;
//   scalar1 = vec0.<MFI_FUNCTION>(mask);
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_REDUCTION_MASK_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T MASK_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
    bool mask_raw[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
    \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
            mask_raw[k] = getRandomValue<bool>(); \
        } \
 \
        VEC_T vec0(raw1); \
        MASK_T mask(mask_raw); \
        volatile SCALAR_T res; \
 \
        start = __rdtsc(); \
            res = vec0.MFI_name(mask); \
        end = __rdtsc(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0;
//   SCALAR_T scalar1, scalar2;
//   scalar1 = vec0.<MFI_FUNCTION>(scalar2);
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_REDUCTION_SCALAR_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
        } \
 \
        VEC_T vec0(raw1); \
        SCALAR_T scalarOp = getRandomValue<SCALAR_T>(); \
 \
        start = __rdtsc(); \
            volatile auto res = vec0.MFI_name(scalarOp); \
        end = __rdtsc(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}


// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0;
//   SCALAR_T scalar1, scalar2;
//   scalar1 = vec0.<MFI_FUNCTION>(scalar2);
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_REDUCTION_SCALAR_MASK_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T MASK_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
    bool mask_raw[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
            mask_raw[k] = getRandomValue<bool>(); \
        } \
 \
        VEC_T vec0(raw1); \
        MASK_T mask(mask_raw); \
        SCALAR_T scalarOp = getRandomValue<SCALAR_T>(); \
        volatile SCALAR_T res; \
 \
        start = __rdtsc(); \
            res = vec0.MFI_name(scalarOp); \
        end = __rdtsc(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0, vec1;
//   bool predicate;
//   predicate = vec0.<MFI_FUNCTION>(vec1);
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_REDUCTION_PREDICATE_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
    alignas(VEC_T::alignment()) SCALAR_T raw2[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
            raw2[k] = getRandomValue<SCALAR_T>(); \
        } \
 \
        VEC_T vec0(raw1); \
        VEC_T vec1(raw2); \
        volatile bool predicate; \
 \
        start = __rdtsc(); \
            predicate = vec0.MFI_name(vec1); \
        end = __rdtsc(); \
 \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0, vec1, vec2, vec3;
//   vec3 = vec0.<MFI_FUNCTION>(vec1, vec2);
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_TERNARY_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
    alignas(VEC_T::alignment()) SCALAR_T raw2[VEC_LEN]; \
    alignas(VEC_T::alignment()) SCALAR_T raw3[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
            raw2[k] = getRandomValue<SCALAR_T>(); \
            raw3[k] = getRandomValue<SCALAR_T>(); \
        } \
 \
        VEC_T vec0(raw1); \
        VEC_T vec1(raw2); \
        VEC_T vec2(raw3); \
        VEC_T res; \
 \
        start = __rdtsc(); \
            res = vec0.MFI_name(vec1, vec2); \
        end = __rdtsc(); \
 \
        volatile SCALAR_T x = res.hadd(); \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}


// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0, vec1, vec2, vec3;
//   MASK_T mask;
//   vec3 = vec0.<MFI_FUNCTION>(mask, vec1, vec2);
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_BASE_TEST_TERNARY_MASK_TEMPLATE(instr_name, MFI_name) \
template<typename VEC_T> \
void instr_name##Latency() { \
    unsigned long long start = 0, end = 0; \
    unsigned long long delta = 0; \
    float latency_avg = 0.0f; \
 \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T; \
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T MASK_T; \
    const int VEC_LEN = VEC_T::length(); \
 \
    alignas(VEC_T::alignment()) SCALAR_T raw1[VEC_LEN]; \
    alignas(VEC_T::alignment()) SCALAR_T raw2[VEC_LEN]; \
    alignas(VEC_T::alignment()) SCALAR_T raw3[VEC_LEN]; \
    bool raw_mask[VEC_LEN]; \
 \
    for (int i = 0; i < ITERATIONS; i++) { \
        for (unsigned int k = 0; k < VEC_LEN; k++) \
        { \
            raw1[k] = getRandomValue<SCALAR_T>(); \
            raw2[k] = getRandomValue<SCALAR_T>(); \
            raw3[k] = getRandomValue<SCALAR_T>(); \
            raw_mask[k] = getRandomValue<bool>(); \
        } \
 \
        VEC_T vec0(raw1); \
        VEC_T vec1(raw2); \
        VEC_T vec2(raw3); \
        MASK_T mask(raw_mask); \
        VEC_T res; \
 \
        start = __rdtsc(); \
            res = vec0.MFI_name(mask, vec1, vec2); \
        end = __rdtsc(); \
 \
        volatile SCALAR_T x = res.hadd(); \
        delta = end - start; \
        float d = float(delta) - latency_avg; \
        latency_avg += d / (1.0f + float(i)); \
 \
    } \
 \
    std::cout << " Average latency is: " << latency_avg << \
        " cycles per element: " << latency_avg / float(VEC_LEN) << std::endl; \
}


// Define all template functions necessary to run tests.

// vec0 = vec1.<INSTR>(vec2)
//      Base interface operations
DEFINE_BASE_TEST_TEMPLATE(ADDV, add);
DEFINE_BASE_TEST_TEMPLATE(SUBV, sub);
DEFINE_BASE_TEST_TEMPLATE(SADDV, sadd);
DEFINE_BASE_TEST_TEMPLATE(SSUBV, ssub);
DEFINE_BASE_TEST_TEMPLATE(SUBFROMV, subfrom);
DEFINE_BASE_TEST_TEMPLATE(MULV, mul);
DEFINE_BASE_TEST_TEMPLATE(DIVV, div);
DEFINE_BASE_TEST_TEMPLATE(MAXV, max);
DEFINE_BASE_TEST_TEMPLATE(MINV, min);
//      Bitwise interface operations
DEFINE_BASE_TEST_TEMPLATE(BANDV, band);
DEFINE_BASE_TEST_TEMPLATE(BORV, bor);
DEFINE_BASE_TEST_TEMPLATE(BXORV, bxor);

// vec0 = vec1.<INSTR>(mask, vec2)
//      Base interface operations
DEFINE_BASE_TEST_MASK_TEMPLATE(MADDV, add);
DEFINE_BASE_TEST_MASK_TEMPLATE(MSUBV, sub);
DEFINE_BASE_TEST_MASK_TEMPLATE(MSADDV, sadd);
DEFINE_BASE_TEST_MASK_TEMPLATE(MSSUBV, ssub);
DEFINE_BASE_TEST_MASK_TEMPLATE(MSUBFROMV, subfrom);
DEFINE_BASE_TEST_MASK_TEMPLATE(MMULV, mul);
DEFINE_BASE_TEST_MASK_TEMPLATE(MDIVV, div);
DEFINE_BASE_TEST_MASK_TEMPLATE(MMAXV, max);
DEFINE_BASE_TEST_MASK_TEMPLATE(MMINV, min);
//      Bitwise interface operations
DEFINE_BASE_TEST_MASK_TEMPLATE(MBANDV, band);
DEFINE_BASE_TEST_MASK_TEMPLATE(MBORV, bor);
DEFINE_BASE_TEST_MASK_TEMPLATE(MBXORV, bxor);

// vec0 = vec1.<INSTR>(scalar2)
//      Base interface operations
DEFINE_BASE_TEST_SCALAR_TEMPLATE(ADDS, add);
DEFINE_BASE_TEST_SCALAR_TEMPLATE(SUBS, sub);
DEFINE_BASE_TEST_SCALAR_TEMPLATE(SADDS, sadd);
DEFINE_BASE_TEST_SCALAR_TEMPLATE(SSUBS, ssub);
DEFINE_BASE_TEST_SCALAR_TEMPLATE(SUBFROMS, subfrom);
DEFINE_BASE_TEST_SCALAR_TEMPLATE(MULS, mul);
DEFINE_BASE_TEST_SCALAR_TEMPLATE(DIVS, div);
DEFINE_BASE_TEST_SCALAR_TEMPLATE(RCPS, rcp);
DEFINE_BASE_TEST_SCALAR_TEMPLATE(MAXS, max);
DEFINE_BASE_TEST_SCALAR_TEMPLATE(MINS, min);
//      Bitwise interface operations
DEFINE_BASE_TEST_SCALAR_TEMPLATE(BANDS, band);
DEFINE_BASE_TEST_SCALAR_TEMPLATE(BORS, bor);
DEFINE_BASE_TEST_SCALAR_TEMPLATE(BXORS, bxor);

// vec0 = vec1.<INSTR>(mask, scalar2)
//      Base interface operations
DEFINE_BASE_TEST_MASK_SCALAR_TEMPLATE(MADDS, add);
DEFINE_BASE_TEST_MASK_SCALAR_TEMPLATE(MSUBS, add);
DEFINE_BASE_TEST_MASK_SCALAR_TEMPLATE(MSADDS, sadd);
DEFINE_BASE_TEST_MASK_SCALAR_TEMPLATE(MSSUBS, ssub);
DEFINE_BASE_TEST_MASK_SCALAR_TEMPLATE(MSUBFROMS, subfrom);
DEFINE_BASE_TEST_MASK_SCALAR_TEMPLATE(MMULS, mul);
DEFINE_BASE_TEST_MASK_SCALAR_TEMPLATE(MDIVS, div);
DEFINE_BASE_TEST_MASK_SCALAR_TEMPLATE(MRCPS, rcp);
DEFINE_BASE_TEST_MASK_SCALAR_TEMPLATE(MMAXS, max);
DEFINE_BASE_TEST_MASK_SCALAR_TEMPLATE(MMINS, min);
//      Bitwise interface operations
DEFINE_BASE_TEST_MASK_SCALAR_TEMPLATE(MBANDS, band);
DEFINE_BASE_TEST_MASK_SCALAR_TEMPLATE(MBORS, bor);
DEFINE_BASE_TEST_MASK_SCALAR_TEMPLATE(MBXORS, bxor);

// vec0 <- vec0.<INSTR>(vec1)
//      Base interface operations
DEFINE_BASE_TEST_ASSIGN_TEMPLATE(ADDVA, adda);
DEFINE_BASE_TEST_ASSIGN_TEMPLATE(SUBVA, suba);
DEFINE_BASE_TEST_ASSIGN_TEMPLATE(SADDVA, sadda);
DEFINE_BASE_TEST_ASSIGN_TEMPLATE(SSUBVA, ssuba);
DEFINE_BASE_TEST_ASSIGN_TEMPLATE(SUBFROMVA, subfroma);
DEFINE_BASE_TEST_ASSIGN_TEMPLATE(MULVA, mula);
DEFINE_BASE_TEST_ASSIGN_TEMPLATE(DIVVA, diva);
DEFINE_BASE_TEST_ASSIGN_TEMPLATE(MAXVA, maxa);
DEFINE_BASE_TEST_ASSIGN_TEMPLATE(MINVA, mina);
//      Bitwise interface operations
DEFINE_BASE_TEST_ASSIGN_TEMPLATE(BANDVA, banda);
DEFINE_BASE_TEST_ASSIGN_TEMPLATE(BORVA, bora)
DEFINE_BASE_TEST_ASSIGN_TEMPLATE(BXORVA, bxora)

// vec0 <- vec0.<INSTR>(mask, vec1)
//      Base interface operations
DEFINE_BASE_TEST_MASK_ASSIGN_TEMPLATE(MADDVA, adda);
DEFINE_BASE_TEST_MASK_ASSIGN_TEMPLATE(MSUBVA, suba);
DEFINE_BASE_TEST_MASK_ASSIGN_TEMPLATE(MSADDVA, sadda);
DEFINE_BASE_TEST_MASK_ASSIGN_TEMPLATE(MSSUBVA, ssuba);
DEFINE_BASE_TEST_MASK_ASSIGN_TEMPLATE(MSUBFROMVA, subfroma);
DEFINE_BASE_TEST_MASK_ASSIGN_TEMPLATE(MMULVA, mula);
DEFINE_BASE_TEST_MASK_ASSIGN_TEMPLATE(MDIVVA, diva);
DEFINE_BASE_TEST_MASK_ASSIGN_TEMPLATE(MMAXVA, maxa);
DEFINE_BASE_TEST_MASK_ASSIGN_TEMPLATE(MMINVA, mina);
//      Bitwise interface operations
DEFINE_BASE_TEST_MASK_ASSIGN_TEMPLATE(MBANDVA, banda);
DEFINE_BASE_TEST_MASK_ASSIGN_TEMPLATE(MBORVA, bora);
DEFINE_BASE_TEST_MASK_ASSIGN_TEMPLATE(MBXORVA, bxora);

// vec0 <- vec0.<INSTR>(scalar1)
//      Base interface operations
DEFINE_BASE_TEST_SCALAR_ASSIGN_TEMPLATE(ADDSA, adda);
DEFINE_BASE_TEST_SCALAR_ASSIGN_TEMPLATE(SUBSA, suba);
DEFINE_BASE_TEST_SCALAR_ASSIGN_TEMPLATE(SADDSA, sadda);
DEFINE_BASE_TEST_SCALAR_ASSIGN_TEMPLATE(SSUBSA, ssuba);
DEFINE_BASE_TEST_SCALAR_ASSIGN_TEMPLATE(SUBFROMSA, subfroma);
DEFINE_BASE_TEST_SCALAR_ASSIGN_TEMPLATE(MULSA, mula);
DEFINE_BASE_TEST_SCALAR_ASSIGN_TEMPLATE(DIVSA, diva);
DEFINE_BASE_TEST_SCALAR_ASSIGN_TEMPLATE(RCPSA, rcpa);
DEFINE_BASE_TEST_SCALAR_ASSIGN_TEMPLATE(MAXSA, maxa);
DEFINE_BASE_TEST_SCALAR_ASSIGN_TEMPLATE(MINSA, mina);
//      Base interface operations
DEFINE_BASE_TEST_SCALAR_ASSIGN_TEMPLATE(BANDSA, banda);
DEFINE_BASE_TEST_SCALAR_ASSIGN_TEMPLATE(BORSA, bora);
DEFINE_BASE_TEST_SCALAR_ASSIGN_TEMPLATE(BXORSA, bxora);

// vec0 <- vec0.<INSTR>(mask, scalar1)
//      Base interface operations
DEFINE_BASE_TEST_MASK_SCALAR_ASSIGN_TEMPLATE(MADDSA, adda);
DEFINE_BASE_TEST_MASK_SCALAR_ASSIGN_TEMPLATE(MSUBSA, suba);
DEFINE_BASE_TEST_MASK_SCALAR_ASSIGN_TEMPLATE(MSADDSA, sadda);
DEFINE_BASE_TEST_MASK_SCALAR_ASSIGN_TEMPLATE(MSSUBSA, ssuba);
DEFINE_BASE_TEST_MASK_SCALAR_ASSIGN_TEMPLATE(MSUBFROMSA, subfroma);
DEFINE_BASE_TEST_MASK_SCALAR_ASSIGN_TEMPLATE(MMULSA, mula);
DEFINE_BASE_TEST_MASK_SCALAR_ASSIGN_TEMPLATE(MDIVSA, diva);
DEFINE_BASE_TEST_MASK_SCALAR_ASSIGN_TEMPLATE(MRCPSA, rcpa);
DEFINE_BASE_TEST_MASK_SCALAR_ASSIGN_TEMPLATE(MMAXSA, maxa);
DEFINE_BASE_TEST_MASK_SCALAR_ASSIGN_TEMPLATE(MMINSA, mina);
//      Base interface operations
DEFINE_BASE_TEST_MASK_SCALAR_ASSIGN_TEMPLATE(MBANDSA, banda);
DEFINE_BASE_TEST_MASK_SCALAR_ASSIGN_TEMPLATE(MBORSA, bora);
DEFINE_BASE_TEST_MASK_SCALAR_ASSIGN_TEMPLATE(MBXORSA, bxora);

// vec1 = vec0.<INSTR>()
//      Base interface operations
DEFINE_BASE_TEST_UNARY_TEMPLATE(POSTINC, postinc);
DEFINE_BASE_TEST_UNARY_TEMPLATE(PREFINC, prefinc);
DEFINE_BASE_TEST_UNARY_TEMPLATE(POSTDEC, postdec);
DEFINE_BASE_TEST_UNARY_TEMPLATE(PREFDEC, prefdec);
DEFINE_BASE_TEST_UNARY_TEMPLATE(RCP, rcp);
//      Bitwise interface operations
DEFINE_BASE_TEST_UNARY_TEMPLATE(BNOT, bnot);

// vec1 = vec0.<INSTR>(mask)
//      Base interface operations
DEFINE_BASE_TEST_UNARY_MASK_TEMPLATE(MPOSTINC, postinc);
DEFINE_BASE_TEST_UNARY_MASK_TEMPLATE(MPREFINC, prefinc);
DEFINE_BASE_TEST_UNARY_MASK_TEMPLATE(MPOSTDEC, postdec);
DEFINE_BASE_TEST_UNARY_MASK_TEMPLATE(MPREFDEC, prefdec);
DEFINE_BASE_TEST_UNARY_MASK_TEMPLATE(MRCP, rcp);
//      Bitwise interface operations
DEFINE_BASE_TEST_UNARY_MASK_TEMPLATE(MBNOT, bnot);

// vec0 <- vec0.<INSTR>()
//      Base interface operations
DEFINE_BASE_TEST_UNARY_ASSIGN_TEMPLATE(RCPA, rcpa);
//      Bitwise interface operations
DEFINE_BASE_TEST_UNARY_ASSIGN_TEMPLATE(BNOTA, bnota);

// vec0 <- vec0.<INSTR>(mask)
//      Base interface operations
DEFINE_BASE_TEST_UNARY_MASK_ASSIGN_TEMPLATE(MRCPA, rcpa);
//      Bitwise interface operations
DEFINE_BASE_TEST_UNARY_MASK_ASSIGN_TEMPLATE(MBNOTA, bnota);

// mask = vec0.<INSTR>(vec1)
//      Base interface operations
DEFINE_BASE_TEST_PREDICATE_TEMPLATE(CMPEQV, cmpeq);
DEFINE_BASE_TEST_PREDICATE_TEMPLATE(CMPNEV, cmpne);
DEFINE_BASE_TEST_PREDICATE_TEMPLATE(CMPGTV, cmpgt);
DEFINE_BASE_TEST_PREDICATE_TEMPLATE(CMPLTV, cmplt);
DEFINE_BASE_TEST_PREDICATE_TEMPLATE(CMPGEV, cmpge);
DEFINE_BASE_TEST_PREDICATE_TEMPLATE(CMPLEV, cmple);

// mask = vec0.<INSTR>(scalar1)
//      Base interface operations
DEFINE_BASE_TEST_PREDICATE_SCALAR_TEMPLATE(CMPEQS, cmpeq);
DEFINE_BASE_TEST_PREDICATE_SCALAR_TEMPLATE(CMPNES, cmpne);
DEFINE_BASE_TEST_PREDICATE_SCALAR_TEMPLATE(CMPGTS, cmpgt);
DEFINE_BASE_TEST_PREDICATE_SCALAR_TEMPLATE(CMPLTS, cmplt);
DEFINE_BASE_TEST_PREDICATE_SCALAR_TEMPLATE(CMPGES, cmpge);
DEFINE_BASE_TEST_PREDICATE_SCALAR_TEMPLATE(CMPLES, cmple);

// scalar = vec0.<INSTR>()
//      Base interface operations
DEFINE_BASE_TEST_REDUCTION_TEMPLATE(HADD, hadd);
DEFINE_BASE_TEST_REDUCTION_TEMPLATE(HMUL, hmul);
DEFINE_BASE_TEST_REDUCTION_TEMPLATE(HMAX, hmax);
DEFINE_BASE_TEST_REDUCTION_TEMPLATE(IMAX, imax);
DEFINE_BASE_TEST_REDUCTION_TEMPLATE(HMIN, hmin);
DEFINE_BASE_TEST_REDUCTION_TEMPLATE(IMIN, imin);
//      Bitwise interface operations
DEFINE_BASE_TEST_REDUCTION_TEMPLATE(HBAND, hband);
DEFINE_BASE_TEST_REDUCTION_TEMPLATE(HBOR, hbor);
DEFINE_BASE_TEST_REDUCTION_TEMPLATE(HBXOR, hbxor);

// scalar = vec0.<INSTR>(mask)
//      Base interface operations
DEFINE_BASE_TEST_REDUCTION_MASK_TEMPLATE(MHADD, hadd);
DEFINE_BASE_TEST_REDUCTION_MASK_TEMPLATE(MHMUL, hmul);
DEFINE_BASE_TEST_REDUCTION_MASK_TEMPLATE(MHMAX, hmax);
DEFINE_BASE_TEST_REDUCTION_MASK_TEMPLATE(MIMAX, imax);
DEFINE_BASE_TEST_REDUCTION_MASK_TEMPLATE(MHMIN, hmin);
DEFINE_BASE_TEST_REDUCTION_MASK_TEMPLATE(MIMIN, imin);
//      Bitwise interface operations
DEFINE_BASE_TEST_REDUCTION_MASK_TEMPLATE(MHBAND, hband);
DEFINE_BASE_TEST_REDUCTION_MASK_TEMPLATE(MHBOR, hbor);
DEFINE_BASE_TEST_REDUCTION_MASK_TEMPLATE(MHBXOR, hbxor);

// scalar2 = vec0.<INSTR>(scalar1)
//      Base interface operations
DEFINE_BASE_TEST_REDUCTION_SCALAR_TEMPLATE(HADDS, hadd);
DEFINE_BASE_TEST_REDUCTION_SCALAR_TEMPLATE(HMULS, hmul);
//      Bitwise interface operations
DEFINE_BASE_TEST_REDUCTION_SCALAR_TEMPLATE(HBANDS, hband);
DEFINE_BASE_TEST_REDUCTION_SCALAR_TEMPLATE(HBORS, hbor);
DEFINE_BASE_TEST_REDUCTION_SCALAR_TEMPLATE(HBXORS, hbxor);

// scalar2 = vec0.<INSTR>(mask, scalar1)
//      Base interface operations
DEFINE_BASE_TEST_REDUCTION_SCALAR_MASK_TEMPLATE(MHADDS, hadd);
DEFINE_BASE_TEST_REDUCTION_SCALAR_MASK_TEMPLATE(MHMULS, hmul);
//      Bitwise interface operations
DEFINE_BASE_TEST_REDUCTION_SCALAR_MASK_TEMPLATE(MHBANDS, hband);
DEFINE_BASE_TEST_REDUCTION_SCALAR_MASK_TEMPLATE(MHBORS, hbor);
DEFINE_BASE_TEST_REDUCTION_SCALAR_MASK_TEMPLATE(MHBXORS, hbxor);

// bool_scalar = vec0.<INSTR>(vec1)
//      Base interface operations
DEFINE_BASE_TEST_REDUCTION_PREDICATE_TEMPLATE(CMPEV, cmpe);

// bool_scalar = vec0.<INSTR>()
//      Base interface operations
DEFINE_BASE_TEST_REDUCTION_TEMPLATE(UNIQUE, unique);

// bool_scalar = vec0.<INSTR>(scalar1)
//      Base interface operations
DEFINE_BASE_TEST_REDUCTION_SCALAR_TEMPLATE(CMPES, cmpe);

// vec0 = vec1.<INSTR>(vec2, vec3)
//      Base interface operations
DEFINE_BASE_TEST_TERNARY_TEMPLATE(FMULADDV, fmuladd);
DEFINE_BASE_TEST_TERNARY_TEMPLATE(FMULSUBV, fmulsub);
DEFINE_BASE_TEST_TERNARY_TEMPLATE(FADDMULV, faddmul);
DEFINE_BASE_TEST_TERNARY_TEMPLATE(FSUBMULV, fsubmul);

// vec0 = vec1.<INSTR>(mask, vec2, vec3)
//      Base interface operations
DEFINE_BASE_TEST_TERNARY_MASK_TEMPLATE(MFMULADDV, fmuladd);
DEFINE_BASE_TEST_TERNARY_MASK_TEMPLATE(MFMULSUBV, fmulsub);
DEFINE_BASE_TEST_TERNARY_MASK_TEMPLATE(MFADDMULV, faddmul);
DEFINE_BASE_TEST_TERNARY_MASK_TEMPLATE(MFSUBMULV, fsubmul);

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define CALL_TEST(instr_name, vecname) \
        std::cout << TOSTRING(vecname) << ": " << TOSTRING(instr_name) << ": "; \
        instr_name##Latency<vecname>();

#define CALL_BASE_TESTS(vecname) \
    CALL_TEST(ADDV, vecname); \
    CALL_TEST(MADDV, vecname) \
    CALL_TEST(ADDS, vecname); \
    CALL_TEST(MADDS, vecname); \
    CALL_TEST(ADDVA, vecname); \
    CALL_TEST(MADDVA, vecname); \
    CALL_TEST(ADDSA, vecname); \
    CALL_TEST(MADDSA, vecname); \
    CALL_TEST(SADDV, vecname); \
    CALL_TEST(MSADDV, vecname); \
    CALL_TEST(SADDS, vecname); \
    CALL_TEST(MSADDS, vecname); \
    CALL_TEST(SADDVA, vecname); \
    CALL_TEST(MSADDVA, vecname); \
    CALL_TEST(SADDSA, vecname); \
    CALL_TEST(MSADDSA, vecname); \
    CALL_TEST(POSTINC, vecname); \
    CALL_TEST(MPOSTINC, vecname); \
    CALL_TEST(PREFINC, vecname); \
    CALL_TEST(MPREFINC, vecname); \
    CALL_TEST(SUBV, vecname); \
    CALL_TEST(MSUBV, vecname) \
    CALL_TEST(SUBS, vecname); \
    CALL_TEST(MSUBS, vecname); \
    CALL_TEST(SUBVA, vecname); \
    CALL_TEST(MSUBVA, vecname); \
    CALL_TEST(SUBSA, vecname); \
    CALL_TEST(MSUBSA, vecname); \
    CALL_TEST(SSUBV, vecname); \
    CALL_TEST(MSSUBV, vecname); \
    CALL_TEST(SSUBS, vecname); \
    CALL_TEST(MSSUBS, vecname); \
    CALL_TEST(SSUBVA, vecname); \
    CALL_TEST(MSSUBVA, vecname); \
    CALL_TEST(SSUBSA, vecname); \
    CALL_TEST(MSSUBSA, vecname); \
    CALL_TEST(SUBFROMV, vecname); \
    CALL_TEST(MSUBFROMV, vecname); \
    CALL_TEST(SUBFROMS, vecname); \
    CALL_TEST(MSUBFROMS, vecname); \
    CALL_TEST(SUBFROMVA, vecname); \
    CALL_TEST(MSUBFROMVA, vecname); \
    CALL_TEST(SUBFROMSA, vecname); \
    CALL_TEST(MSUBFROMSA, vecname); \
    CALL_TEST(POSTDEC, vecname); \
    CALL_TEST(MPOSTDEC, vecname); \
    CALL_TEST(PREFDEC, vecname); \
    CALL_TEST(MPREFDEC, vecname); \
    CALL_TEST(MULV, vecname); \
    CALL_TEST(MMULV, vecname); \
    CALL_TEST(MULS, vecname); \
    CALL_TEST(MMULS, vecname); \
    CALL_TEST(MULVA, vecname); \
    CALL_TEST(MMULVA, vecname); \
    CALL_TEST(MULSA, vecname); \
    CALL_TEST(MMULSA, vecname); \
    CALL_TEST(DIVV, vecname); \
    CALL_TEST(MDIVV, vecname); \
    CALL_TEST(DIVS, vecname); \
    CALL_TEST(MDIVS, vecname); \
    CALL_TEST(DIVVA, vecname); \
    CALL_TEST(MDIVVA, vecname); \
    CALL_TEST(DIVSA, vecname); \
    CALL_TEST(MDIVSA, vecname); \
    CALL_TEST(RCP, vecname); \
    CALL_TEST(MRCP, vecname); \
    CALL_TEST(RCPS, vecname); \
    CALL_TEST(MRCPS, vecname); \
    CALL_TEST(RCPA, vecname); \
    CALL_TEST(MRCPA, vecname); \
    CALL_TEST(RCPSA, vecname); \
    CALL_TEST(MRCPSA, vecname); \
    CALL_TEST(CMPEQV, vecname); \
    CALL_TEST(CMPEQS, vecname); \
    CALL_TEST(CMPNEV, vecname); \
    CALL_TEST(CMPNES, vecname); \
    CALL_TEST(CMPGTV, vecname); \
    CALL_TEST(CMPGTS, vecname); \
    CALL_TEST(CMPLTV, vecname); \
    CALL_TEST(CMPLTS, vecname); \
    CALL_TEST(CMPGEV, vecname); \
    CALL_TEST(CMPGES, vecname); \
    CALL_TEST(CMPLEV, vecname); \
    CALL_TEST(CMPLES, vecname); \
    CALL_TEST(CMPEV, vecname); \
    CALL_TEST(CMPES, vecname); \
    CALL_TEST(UNIQUE, vecname); \
    CALL_TEST(HADD, vecname); \
    CALL_TEST(MHADD, vecname); \
    CALL_TEST(HADDS, vecname); \
    CALL_TEST(MHADDS, vecname); \
    CALL_TEST(HMUL, vecname); \
    CALL_TEST(MHMUL, vecname); \
    CALL_TEST(HMULS, vecname); \
    CALL_TEST(MHMULS, vecname); \
    CALL_TEST(FMULADDV, vecname); \
    CALL_TEST(MFMULADDV, vecname); \
    CALL_TEST(FMULSUBV, vecname); \
    CALL_TEST(MFMULSUBV, vecname); \
    CALL_TEST(FADDMULV, vecname); \
    CALL_TEST(MFADDMULV, vecname); \
    CALL_TEST(FSUBMULV, vecname); \
    CALL_TEST(MFSUBMULV, vecname); \
    CALL_TEST(MAXV, vecname); \
    CALL_TEST(MMAXV, vecname); \
    CALL_TEST(MAXS, vecname); \
    CALL_TEST(MMAXS, vecname); \
    CALL_TEST(MAXVA, vecname); \
    CALL_TEST(MMAXVA, vecname); \
    CALL_TEST(MAXSA, vecname); \
    CALL_TEST(MMAXSA, vecname); \
    CALL_TEST(MINV, vecname); \
    CALL_TEST(MMINV, vecname); \
    CALL_TEST(MINS, vecname); \
    CALL_TEST(MMINS, vecname); \
    CALL_TEST(MINVA, vecname); \
    CALL_TEST(MMINVA, vecname); \
    CALL_TEST(MINSA, vecname); \
    CALL_TEST(MMINSA, vecname); \
    CALL_TEST(HMAX, vecname); \
    CALL_TEST(MHMAX, vecname); \
    CALL_TEST(IMAX, vecname); \
    CALL_TEST(MIMAX, vecname); \
    CALL_TEST(HMIN, vecname); \
    CALL_TEST(MHMIN, vecname); \
    CALL_TEST(IMIN, vecname); \
    CALL_TEST(MIMIN, vecname); 

#define CALL_TESTS_BITWISE(vecname); \
    CALL_TEST(BANDV, vecname); \
    CALL_TEST(MBANDV, vecname); \
    CALL_TEST(BANDS, vecname); \
    CALL_TEST(MBANDS, vecname); \
    CALL_TEST(BANDVA, vecname); \
    CALL_TEST(MBANDVA, vecname); \
    CALL_TEST(BANDSA, vecname); \
    CALL_TEST(MBANDSA, vecname); \
    CALL_TEST(BORV, vecname); \
    CALL_TEST(MBORV, vecname); \
    CALL_TEST(BORS, vecname); \
    CALL_TEST(MBORS, vecname); \
    CALL_TEST(BORVA, vecname); \
    CALL_TEST(MBORVA, vecname); \
    CALL_TEST(BORSA, vecname); \
    CALL_TEST(MBORSA, vecname); \
    CALL_TEST(BXORV, vecname); \
    CALL_TEST(MBXORV, vecname); \
    CALL_TEST(BXORS, vecname); \
    CALL_TEST(MBXORS, vecname); \
    CALL_TEST(BXORVA, vecname); \
    CALL_TEST(MBXORVA, vecname); \
    CALL_TEST(BXORSA, vecname); \
    CALL_TEST(MBXORSA, vecname); \
    CALL_TEST(BNOT, vecname); \
    CALL_TEST(MBNOT, vecname); \
    CALL_TEST(BNOTA, vecname); \
    CALL_TEST(MBNOTA, vecname); \
    CALL_TEST(HBAND, vecname); \
    CALL_TEST(MHBAND, vecname); \
    CALL_TEST(HBANDS, vecname); \
    CALL_TEST(MHBANDS, vecname); \
    CALL_TEST(HBOR, vecname); \
    CALL_TEST(MHBOR, vecname); \
    CALL_TEST(HBORS, vecname); \
    CALL_TEST(MHBORS, vecname); \
    CALL_TEST(HBXOR, vecname); \
    CALL_TEST(MHBXOR, vecname); \
    CALL_TEST(HBXORS, vecname); \
    CALL_TEST(MHBXORS, vecname);

#define CALL_TESTS_UINT(vecname) \
    std::cout << "Testing: " << TOSTRING(vecname) << "\n"; \
    CALL_BASE_TESTS(vecname); \
    CALL_TESTS_BITWISE(vecname);

#define CALL_TESTS_INT(vecname) \
    std::cout << "Testing: " << TOSTRING(vecname) << "\n"; \
    CALL_BASE_TESTS(vecname); \
    CALL_TESTS_BITWISE(vecname);

#define CALL_TESTS_FLOAT(vecname) \
    std::cout << "Testing: " << TOSTRING(vecname) << "\n"; \
    CALL_BASE_TESTS(vecname);

#endif
