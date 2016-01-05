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

template<typename VEC_T, typename SCALAR_T>
SCALAR_T forceReduction(VEC_T & x) {
    return x.hadd();
}

// specialization for masks
template <> bool forceReduction<SIMDMask1, bool>(SIMDMask1 & x) { return x.hlxor(); }
template <> bool forceReduction<SIMDMask2, bool>(SIMDMask2 & x) { return x.hlxor(); }
template <> bool forceReduction<SIMDMask4, bool>(SIMDMask4 & x) { return x.hlxor(); }
template <> bool forceReduction<SIMDMask8, bool>(SIMDMask8 & x) { return x.hlxor(); }
template <> bool forceReduction<SIMDMask16, bool>(SIMDMask16 & x) { return x.hlxor(); }
template <> bool forceReduction<SIMDMask32, bool>(SIMDMask32 & x) { return x.hlxor(); }
template <> bool forceReduction<SIMDMask64, bool>(SIMDMask64 & x) { return x.hlxor(); }
template <> bool forceReduction<SIMDMask128, bool> (SIMDMask128 & x) { return x.hlxor(); }

const int ITERATIONS = 1000;

// Generate test function for Base vector operations of following form:
//
//   VEC_T vec0, vec1, vec2;
//   vec2 = vec0.<MFI_FUNCTION>(vec1);
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_VEC_VEC_TEST_TEMPLATE(instr_name, MFI_name) \
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
        volatile SCALAR_T x = forceReduction<VEC_T, SCALAR_T>(res); \
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
#define DEFINE_VEC_MASK_VEC_TEST_TEMPLATE(instr_name, MFI_name) \
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
#define DEFINE_VEC_SCALAR_TEST_TEMPLATE(instr_name, MFI_name) \
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
        volatile SCALAR_T x = forceReduction<VEC_T, SCALAR_T>(res); \
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
#define DEFINE_VEC_MASK_SCALAR_TEST_TEMPLATE(instr_name, MFI_name) \
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
#define DEFINE_ASSIGN_VEC_TEST_TEMPLATE(instr_name, MFI_name) \
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
        volatile SCALAR_T x = forceReduction<VEC_T, SCALAR_T>(vec0); \
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
#define DEFINE_ASSIGN_MASK_VEC_TEST_TEMPLATE(instr_name, MFI_name) \
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
#define DEFINE_ASSIGN_SCALAR_TEST_TEMPLATE(instr_name, MFI_name) \
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
        volatile SCALAR_T x = forceReduction<VEC_T, SCALAR_T>(vec0); \
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
#define DEFINE_ASSIGN_MASK_SCALAR_TEST_TEMPLATE(instr_name, MFI_name) \
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
#define DEFINE_VEC_TEST_TEMPLATE(instr_name, MFI_name) \
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
#define DEFINE_VEC_MASK_TEST_TEMPLATE(instr_name, MFI_name) \
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
#define DEFINE_ASSIGN_TEST_TEMPLATE(instr_name, MFI_name) \
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
//   MASK_T mask;
//   vec0.<MFI_FUNCTION>(mask);
//
//  where MFI_FUNCTION is an in-place operation (e.g. MRCPA)
// 
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_ASSIGN_MASK_TEST_TEMPLATE(instr_name, MFI_name) \
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
#define DEFINE_MASK_VEC_TEST_TEMPLATE(instr_name, MFI_name) \
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
//   VEC_T vec0;
//   SCALAR_T scalar;
//   MASK_T mask;
//   mask = vec0.<MFI_FUNCTION>(scalar);
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_MASK_SCALAR_TEST_TEMPLATE(instr_name, MFI_name) \
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
#define DEFINE_SCALAR_TEST_TEMPLATE(instr_name, MFI_name) \
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
#define DEFINE_SCALAR_MASK_TEST_TEMPLATE(instr_name, MFI_name) \
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
#define DEFINE_SCALAR_SCALAR_TEST_TEMPLATE(instr_name, MFI_name) \
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
//   MASK_T mask;
//   SCALAR_T scalar1, scalar2;
//   scalar1 = vec0.<MFI_FUNCTION>(scalar2);
//
// instr_name - name of instruction as defined in UME::SIMD interface spec
// MFI_name   - name of function in Member Function Interface used to implement instr_name
#define DEFINE_SCALAR_MASK_SCALAR_TEST_TEMPLATE(instr_name, MFI_name) \
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
            res = vec0.MFI_name(mask, scalarOp); \
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
#define DEFINE_BOOL_VEC_TEST_TEMPLATE(instr_name, MFI_name) \
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
#define DEFINE_VEC_VEC_VEC_TEST_TEMPLATE(instr_name, MFI_name) \
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
#define DEFINE_VEC_MASK_VEC_VEC_TEST_TEMPLATE(instr_name, MFI_name) \
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
//      Mask interface operations
DEFINE_VEC_VEC_TEST_TEMPLATE(LANDV, land);
DEFINE_VEC_VEC_TEST_TEMPLATE(LORV, lor);
DEFINE_VEC_VEC_TEST_TEMPLATE(LXORV, lxor);
//      Base interface operations
DEFINE_VEC_VEC_TEST_TEMPLATE(ADDV, add);
DEFINE_VEC_VEC_TEST_TEMPLATE(SUBV, sub);
DEFINE_VEC_VEC_TEST_TEMPLATE(SADDV, sadd);
DEFINE_VEC_VEC_TEST_TEMPLATE(SSUBV, ssub);
DEFINE_VEC_VEC_TEST_TEMPLATE(SUBFROMV, subfrom);
DEFINE_VEC_VEC_TEST_TEMPLATE(MULV, mul);
DEFINE_VEC_VEC_TEST_TEMPLATE(DIVV, div);
DEFINE_VEC_VEC_TEST_TEMPLATE(MAXV, max);
DEFINE_VEC_VEC_TEST_TEMPLATE(MINV, min);
//      Bitwise interface operations
DEFINE_VEC_VEC_TEST_TEMPLATE(BANDV, band);
DEFINE_VEC_VEC_TEST_TEMPLATE(BORV, bor);
DEFINE_VEC_VEC_TEST_TEMPLATE(BXORV, bxor);

// vec0 = vec1.<INSTR>(mask, vec2)
//      Base interface operations
DEFINE_VEC_MASK_VEC_TEST_TEMPLATE(MADDV, add);
DEFINE_VEC_MASK_VEC_TEST_TEMPLATE(MSUBV, sub);
DEFINE_VEC_MASK_VEC_TEST_TEMPLATE(MSADDV, sadd);
DEFINE_VEC_MASK_VEC_TEST_TEMPLATE(MSSUBV, ssub);
DEFINE_VEC_MASK_VEC_TEST_TEMPLATE(MSUBFROMV, subfrom);
DEFINE_VEC_MASK_VEC_TEST_TEMPLATE(MMULV, mul);
DEFINE_VEC_MASK_VEC_TEST_TEMPLATE(MDIVV, div);
DEFINE_VEC_MASK_VEC_TEST_TEMPLATE(MMAXV, max);
DEFINE_VEC_MASK_VEC_TEST_TEMPLATE(MMINV, min);
//      Bitwise interface operations
DEFINE_VEC_MASK_VEC_TEST_TEMPLATE(MBANDV, band);
DEFINE_VEC_MASK_VEC_TEST_TEMPLATE(MBORV, bor);
DEFINE_VEC_MASK_VEC_TEST_TEMPLATE(MBXORV, bxor);

// vec0 = vec1.<INSTR>(scalar2)
//      Mask interface operations
DEFINE_VEC_SCALAR_TEST_TEMPLATE(LANDS, land);
DEFINE_VEC_SCALAR_TEST_TEMPLATE(LORS, lor);
DEFINE_VEC_SCALAR_TEST_TEMPLATE(LXORS, lxor);
//      Base interface operations
DEFINE_VEC_SCALAR_TEST_TEMPLATE(ADDS, add);
DEFINE_VEC_SCALAR_TEST_TEMPLATE(SUBS, sub);
DEFINE_VEC_SCALAR_TEST_TEMPLATE(SADDS, sadd);
DEFINE_VEC_SCALAR_TEST_TEMPLATE(SSUBS, ssub);
DEFINE_VEC_SCALAR_TEST_TEMPLATE(SUBFROMS, subfrom);
DEFINE_VEC_SCALAR_TEST_TEMPLATE(MULS, mul);
DEFINE_VEC_SCALAR_TEST_TEMPLATE(DIVS, div);
DEFINE_VEC_SCALAR_TEST_TEMPLATE(RCPS, rcp);
DEFINE_VEC_SCALAR_TEST_TEMPLATE(MAXS, max);
DEFINE_VEC_SCALAR_TEST_TEMPLATE(MINS, min);
//      Bitwise interface operations
DEFINE_VEC_SCALAR_TEST_TEMPLATE(BANDS, band);
DEFINE_VEC_SCALAR_TEST_TEMPLATE(BORS, bor);
DEFINE_VEC_SCALAR_TEST_TEMPLATE(BXORS, bxor);

// vec0 = vec1.<INSTR>(mask, scalar2)
//      Base interface operations
DEFINE_VEC_MASK_SCALAR_TEST_TEMPLATE(MADDS, add);
DEFINE_VEC_MASK_SCALAR_TEST_TEMPLATE(MSUBS, add);
DEFINE_VEC_MASK_SCALAR_TEST_TEMPLATE(MSADDS, sadd);
DEFINE_VEC_MASK_SCALAR_TEST_TEMPLATE(MSSUBS, ssub);
DEFINE_VEC_MASK_SCALAR_TEST_TEMPLATE(MSUBFROMS, subfrom);
DEFINE_VEC_MASK_SCALAR_TEST_TEMPLATE(MMULS, mul);
DEFINE_VEC_MASK_SCALAR_TEST_TEMPLATE(MDIVS, div);
DEFINE_VEC_MASK_SCALAR_TEST_TEMPLATE(MRCPS, rcp);
DEFINE_VEC_MASK_SCALAR_TEST_TEMPLATE(MMAXS, max);
DEFINE_VEC_MASK_SCALAR_TEST_TEMPLATE(MMINS, min);
//      Bitwise interface operations
DEFINE_VEC_MASK_SCALAR_TEST_TEMPLATE(MBANDS, band);
DEFINE_VEC_MASK_SCALAR_TEST_TEMPLATE(MBORS, bor);
DEFINE_VEC_MASK_SCALAR_TEST_TEMPLATE(MBXORS, bxor);

// vec0 <- vec0.<INSTR>(vec1)
//      Mask interface operations
DEFINE_ASSIGN_VEC_TEST_TEMPLATE(LANDVA, landa);
DEFINE_ASSIGN_VEC_TEST_TEMPLATE(LORVA, lora);
DEFINE_ASSIGN_VEC_TEST_TEMPLATE(LXORVA, lxora);
//      Base interface operations
DEFINE_ASSIGN_VEC_TEST_TEMPLATE(ADDVA, adda);
DEFINE_ASSIGN_VEC_TEST_TEMPLATE(SUBVA, suba);
DEFINE_ASSIGN_VEC_TEST_TEMPLATE(SADDVA, sadda);
DEFINE_ASSIGN_VEC_TEST_TEMPLATE(SSUBVA, ssuba);
DEFINE_ASSIGN_VEC_TEST_TEMPLATE(SUBFROMVA, subfroma);
DEFINE_ASSIGN_VEC_TEST_TEMPLATE(MULVA, mula);
DEFINE_ASSIGN_VEC_TEST_TEMPLATE(DIVVA, diva);
DEFINE_ASSIGN_VEC_TEST_TEMPLATE(MAXVA, maxa);
DEFINE_ASSIGN_VEC_TEST_TEMPLATE(MINVA, mina);
//      Bitwise interface operations
DEFINE_ASSIGN_VEC_TEST_TEMPLATE(BANDVA, banda);
DEFINE_ASSIGN_VEC_TEST_TEMPLATE(BORVA, bora)
DEFINE_ASSIGN_VEC_TEST_TEMPLATE(BXORVA, bxora)

// vec0 <- vec0.<INSTR>(mask, vec1)
//      Base interface operations
DEFINE_ASSIGN_MASK_VEC_TEST_TEMPLATE(MADDVA, adda);
DEFINE_ASSIGN_MASK_VEC_TEST_TEMPLATE(MSUBVA, suba);
DEFINE_ASSIGN_MASK_VEC_TEST_TEMPLATE(MSADDVA, sadda);
DEFINE_ASSIGN_MASK_VEC_TEST_TEMPLATE(MSSUBVA, ssuba);
DEFINE_ASSIGN_MASK_VEC_TEST_TEMPLATE(MSUBFROMVA, subfroma);
DEFINE_ASSIGN_MASK_VEC_TEST_TEMPLATE(MMULVA, mula);
DEFINE_ASSIGN_MASK_VEC_TEST_TEMPLATE(MDIVVA, diva);
DEFINE_ASSIGN_MASK_VEC_TEST_TEMPLATE(MMAXVA, maxa);
DEFINE_ASSIGN_MASK_VEC_TEST_TEMPLATE(MMINVA, mina);
//      Bitwise interface operations
DEFINE_ASSIGN_MASK_VEC_TEST_TEMPLATE(MBANDVA, banda);
DEFINE_ASSIGN_MASK_VEC_TEST_TEMPLATE(MBORVA, bora);
DEFINE_ASSIGN_MASK_VEC_TEST_TEMPLATE(MBXORVA, bxora);

// vec0 <- vec0.<INSTR>(scalar1)
//      Mask interface operations
DEFINE_ASSIGN_SCALAR_TEST_TEMPLATE(LANDSA, landa);
DEFINE_ASSIGN_SCALAR_TEST_TEMPLATE(LORSA, lora);
DEFINE_ASSIGN_SCALAR_TEST_TEMPLATE(LXORSA, lxora);
//      Base interface operations
DEFINE_ASSIGN_SCALAR_TEST_TEMPLATE(ADDSA, adda);
DEFINE_ASSIGN_SCALAR_TEST_TEMPLATE(SUBSA, suba);
DEFINE_ASSIGN_SCALAR_TEST_TEMPLATE(SADDSA, sadda);
DEFINE_ASSIGN_SCALAR_TEST_TEMPLATE(SSUBSA, ssuba);
DEFINE_ASSIGN_SCALAR_TEST_TEMPLATE(SUBFROMSA, subfroma);
DEFINE_ASSIGN_SCALAR_TEST_TEMPLATE(MULSA, mula);
DEFINE_ASSIGN_SCALAR_TEST_TEMPLATE(DIVSA, diva);
DEFINE_ASSIGN_SCALAR_TEST_TEMPLATE(RCPSA, rcpa);
DEFINE_ASSIGN_SCALAR_TEST_TEMPLATE(MAXSA, maxa);
DEFINE_ASSIGN_SCALAR_TEST_TEMPLATE(MINSA, mina);
//      Base interface operations
DEFINE_ASSIGN_SCALAR_TEST_TEMPLATE(BANDSA, banda);
DEFINE_ASSIGN_SCALAR_TEST_TEMPLATE(BORSA, bora);
DEFINE_ASSIGN_SCALAR_TEST_TEMPLATE(BXORSA, bxora);

// vec0 <- vec0.<INSTR>(mask, scalar1)
//      Base interface operations
DEFINE_ASSIGN_MASK_SCALAR_TEST_TEMPLATE(MADDSA, adda);
DEFINE_ASSIGN_MASK_SCALAR_TEST_TEMPLATE(MSUBSA, suba);
DEFINE_ASSIGN_MASK_SCALAR_TEST_TEMPLATE(MSADDSA, sadda);
DEFINE_ASSIGN_MASK_SCALAR_TEST_TEMPLATE(MSSUBSA, ssuba);
DEFINE_ASSIGN_MASK_SCALAR_TEST_TEMPLATE(MSUBFROMSA, subfroma);
DEFINE_ASSIGN_MASK_SCALAR_TEST_TEMPLATE(MMULSA, mula);
DEFINE_ASSIGN_MASK_SCALAR_TEST_TEMPLATE(MDIVSA, diva);
DEFINE_ASSIGN_MASK_SCALAR_TEST_TEMPLATE(MRCPSA, rcpa);
DEFINE_ASSIGN_MASK_SCALAR_TEST_TEMPLATE(MMAXSA, maxa);
DEFINE_ASSIGN_MASK_SCALAR_TEST_TEMPLATE(MMINSA, mina);
//      Base interface operations
DEFINE_ASSIGN_MASK_SCALAR_TEST_TEMPLATE(MBANDSA, banda);
DEFINE_ASSIGN_MASK_SCALAR_TEST_TEMPLATE(MBORSA, bora);
DEFINE_ASSIGN_MASK_SCALAR_TEST_TEMPLATE(MBXORSA, bxora);

// vec1 = vec0.<INSTR>()
//      Base interface operations
DEFINE_VEC_TEST_TEMPLATE(POSTINC, postinc);
DEFINE_VEC_TEST_TEMPLATE(PREFINC, prefinc);
DEFINE_VEC_TEST_TEMPLATE(POSTDEC, postdec);
DEFINE_VEC_TEST_TEMPLATE(PREFDEC, prefdec);
DEFINE_VEC_TEST_TEMPLATE(RCP, rcp);
//      Bitwise interface operations
DEFINE_VEC_TEST_TEMPLATE(BNOT, bnot);

// vec1 = vec0.<INSTR>(mask)
//      Base interface operations
DEFINE_VEC_MASK_TEST_TEMPLATE(MPOSTINC, postinc);
DEFINE_VEC_MASK_TEST_TEMPLATE(MPREFINC, prefinc);
DEFINE_VEC_MASK_TEST_TEMPLATE(MPOSTDEC, postdec);
DEFINE_VEC_MASK_TEST_TEMPLATE(MPREFDEC, prefdec);
DEFINE_VEC_MASK_TEST_TEMPLATE(MRCP, rcp);
//      Bitwise interface operations
DEFINE_VEC_MASK_TEST_TEMPLATE(MBNOT, bnot);

// vec0 <- vec0.<INSTR>()
//      Base interface operations
DEFINE_ASSIGN_TEST_TEMPLATE(RCPA, rcpa);
//      Bitwise interface operations
DEFINE_ASSIGN_TEST_TEMPLATE(BNOTA, bnota);

// vec0 <- vec0.<INSTR>(mask)
//      Base interface operations
DEFINE_ASSIGN_MASK_TEST_TEMPLATE(MRCPA, rcpa);
//      Bitwise interface operations
DEFINE_ASSIGN_MASK_TEST_TEMPLATE(MBNOTA, bnota);

// mask = vec0.<INSTR>(vec1)
//      Base interface operations
DEFINE_MASK_VEC_TEST_TEMPLATE(CMPEQV, cmpeq);
DEFINE_MASK_VEC_TEST_TEMPLATE(CMPNEV, cmpne);
DEFINE_MASK_VEC_TEST_TEMPLATE(CMPGTV, cmpgt);
DEFINE_MASK_VEC_TEST_TEMPLATE(CMPLTV, cmplt);
DEFINE_MASK_VEC_TEST_TEMPLATE(CMPGEV, cmpge);
DEFINE_MASK_VEC_TEST_TEMPLATE(CMPLEV, cmple);

// mask = vec0.<INSTR>(scalar1)
//      Base interface operations
DEFINE_MASK_SCALAR_TEST_TEMPLATE(CMPEQS, cmpeq);
DEFINE_MASK_SCALAR_TEST_TEMPLATE(CMPNES, cmpne);
DEFINE_MASK_SCALAR_TEST_TEMPLATE(CMPGTS, cmpgt);
DEFINE_MASK_SCALAR_TEST_TEMPLATE(CMPLTS, cmplt);
DEFINE_MASK_SCALAR_TEST_TEMPLATE(CMPGES, cmpge);
DEFINE_MASK_SCALAR_TEST_TEMPLATE(CMPLES, cmple);

// scalar = vec0.<INSTR>()
//      Base interface operations
DEFINE_SCALAR_TEST_TEMPLATE(HADD, hadd);
DEFINE_SCALAR_TEST_TEMPLATE(HMUL, hmul);
DEFINE_SCALAR_TEST_TEMPLATE(HMAX, hmax);
DEFINE_SCALAR_TEST_TEMPLATE(IMAX, imax);
DEFINE_SCALAR_TEST_TEMPLATE(HMIN, hmin);
DEFINE_SCALAR_TEST_TEMPLATE(IMIN, imin);
//      Bitwise interface operations
DEFINE_SCALAR_TEST_TEMPLATE(HBAND, hband);
DEFINE_SCALAR_TEST_TEMPLATE(HBOR, hbor);
DEFINE_SCALAR_TEST_TEMPLATE(HBXOR, hbxor);

// scalar = vec0.<INSTR>(mask)
//      Base interface operations
DEFINE_SCALAR_MASK_TEST_TEMPLATE(MHADD, hadd);
DEFINE_SCALAR_MASK_TEST_TEMPLATE(MHMUL, hmul);
DEFINE_SCALAR_MASK_TEST_TEMPLATE(MHMAX, hmax);
DEFINE_SCALAR_MASK_TEST_TEMPLATE(MIMAX, imax);
DEFINE_SCALAR_MASK_TEST_TEMPLATE(MHMIN, hmin);
DEFINE_SCALAR_MASK_TEST_TEMPLATE(MIMIN, imin);
//      Bitwise interface operations
DEFINE_SCALAR_MASK_TEST_TEMPLATE(MHBAND, hband);
DEFINE_SCALAR_MASK_TEST_TEMPLATE(MHBOR, hbor);
DEFINE_SCALAR_MASK_TEST_TEMPLATE(MHBXOR, hbxor);

// scalar2 = vec0.<INSTR>(scalar1)
//      Base interface operations
DEFINE_SCALAR_SCALAR_TEST_TEMPLATE(HADDS, hadd);
DEFINE_SCALAR_SCALAR_TEST_TEMPLATE(HMULS, hmul);
//      Bitwise interface operations
DEFINE_SCALAR_SCALAR_TEST_TEMPLATE(HBANDS, hband);
DEFINE_SCALAR_SCALAR_TEST_TEMPLATE(HBORS, hbor);
DEFINE_SCALAR_SCALAR_TEST_TEMPLATE(HBXORS, hbxor);

// scalar2 = vec0.<INSTR>(mask, scalar1)
//      Base interface operations
DEFINE_SCALAR_MASK_SCALAR_TEST_TEMPLATE(MHADDS, hadd);
DEFINE_SCALAR_MASK_SCALAR_TEST_TEMPLATE(MHMULS, hmul);
//      Bitwise interface operations
DEFINE_SCALAR_MASK_SCALAR_TEST_TEMPLATE(MHBANDS, hband);
DEFINE_SCALAR_MASK_SCALAR_TEST_TEMPLATE(MHBORS, hbor);
DEFINE_SCALAR_MASK_SCALAR_TEST_TEMPLATE(MHBXORS, hbxor);

// bool_scalar = vec0.<INSTR>(vec1)
//      Base interface operations
DEFINE_BOOL_VEC_TEST_TEMPLATE(CMPEV, cmpe);

// bool_scalar = vec0.<INSTR>()
//      Base interface operations
DEFINE_SCALAR_TEST_TEMPLATE(UNIQUE, unique);

// bool_scalar = vec0.<INSTR>(scalar1)
//      Base interface operations
DEFINE_SCALAR_SCALAR_TEST_TEMPLATE(CMPES, cmpe);

// vec0 = vec1.<INSTR>(vec2, vec3)
//      Base interface operations
DEFINE_VEC_VEC_VEC_TEST_TEMPLATE(FMULADDV, fmuladd);
DEFINE_VEC_VEC_VEC_TEST_TEMPLATE(FMULSUBV, fmulsub);
DEFINE_VEC_VEC_VEC_TEST_TEMPLATE(FADDMULV, faddmul);
DEFINE_VEC_VEC_VEC_TEST_TEMPLATE(FSUBMULV, fsubmul);

// vec0 = vec1.<INSTR>(mask, vec2, vec3)
//      Base interface operations
DEFINE_VEC_MASK_VEC_VEC_TEST_TEMPLATE(MFMULADDV, fmuladd);
DEFINE_VEC_MASK_VEC_VEC_TEST_TEMPLATE(MFMULSUBV, fmulsub);
DEFINE_VEC_MASK_VEC_VEC_TEST_TEMPLATE(MFADDMULV, faddmul);
DEFINE_VEC_MASK_VEC_VEC_TEST_TEMPLATE(MFSUBMULV, fsubmul);

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

#define CALL_TESTS_MASK(vecname) \
    std::cout << "Testing: " << TOSTRING(vecname) << "\n"; \
    CALL_TEST(LANDV, vecname); \
    CALL_TEST(LANDS, vecname); \
    CALL_TEST(LANDVA, vecname); \
    CALL_TEST(LANDSA, vecname); \
    CALL_TEST(LORV, vecname); \
    CALL_TEST(LORS, vecname); \
    CALL_TEST(LORVA, vecname); \
    CALL_TEST(LORSA, vecname); \
    CALL_TEST(LXORV, vecname); \
    CALL_TEST(LXORS, vecname); \
    CALL_TEST(LXORVA, vecname); \
    CALL_TEST(LXORSA, vecname); 

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
