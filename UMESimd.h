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

#ifndef UME_SIMD_H_
#define UME_SIMD_H_

#include "UMEBasicTypes.h"
#include "UMEMemory.h"

// Always include emulated vector types, so that plugins can use them to emulate unsupported types.
// If an emulated mode is selected, this plugin defines all necessary vector types.

// Define static warning for emulation instantiation. This will be used to inform user which parts of vectorization code are emulated.

// ************************************************************************************************
// Uncomment this define for debug/inefficiences identification. 
//  VS requires at least W3 warning level!!!

//#define UME_SIMD_SHOW_EMULATION_WARNINGS 1

//  Enabling emulation warnings allows tracing of following information:
//    1. UME_EMULATION_WARNING informs which emulated functions are used. Emulated functions are 
//       created for compatibility reasons and thus are considered unoptimal. They should
//       be used for functional testing and debugging, and should be avoided in production codes.
//
//          This type of warning is a STATIC warning emited at compile time.
//
//    2. UME_PERFORMANCE_UNOPTIMAL_WARNING used in plugins to inform that certain operation
//       does not have a counter part in instruction set or specific library. When this warning
//       appears, user should re-consider use of designated function/operator and replace it with
//       specialized code. 
//         
//         This type of warning is a STATIC warning emited at compile time
//
//    3. UME_ALIGNMENT_CHECK used to verify if data pointer is aligned correctly.
//
//         This type of warning is a DYNAMIC warning emited at run time.
//
// ************************************************************************************************
//#define UME_SIMD_SHOW_EMULATION_WARNINGS
#ifdef UME_SIMD_SHOW_EMULATION_WARNINGS
namespace UME
{
namespace SIMD
{
#if defined(__GNUC__) || defined(__ICC__)
#define DEPRECATE(ret_type, msg) ret_type __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
#define DEPRECATE(ret_type, msg) __declspec(deprecated(msg)) ret_type
#else
#error Compiler not supported
#endif
    
    #define PP_CAT(x, y) x##y
    namespace emulation_warning
    {
        struct true_type {};
        struct false_type {};
        template <int test> struct converter : public true_type {};
        template <> struct converter<0> : public false_type {};
    }
    /*
    #define UME_SIMD_STATIC_WARN(cond, msg) \
    struct PP_CAT(static_warning, __LINE__ ) { \
        DEPRECATE(void _(UME::SIMD::emulation_warning::false_type const & ), msg) {}; \
        void _(UME::SIMD::emulation_warning::true_type const & ) {};\
        PP_CAT(static_warning, __LINE__)() {_(UME::SIMD::emulation_warning::converter<(cond)>());} \
    }*/
    #define UME_SIMD_STATIC_WARN(cond, msg) \
    struct PP_CAT(static_warning, __LINE__ ) { \
        DEPRECATE(void _(UME::SIMD::emulation_warning::false_type const & ), msg) {}; \
        void _(UME::SIMD::emulation_warning::true_type const & ) {};\
        PP_CAT(static_warning, __LINE__)() {_(UME::SIMD::emulation_warning::converter<(cond)>());} \
    }

    #define UME_SIMD_DYNAMIC_WARN(cond, msg) if(cond) std::cerr << msg;
    

    // Specific warnings
    #define UME_EMULATION_WARNING() UME_SIMD_STATIC_WARN(false, "Using emulation")
    #define UME_PERFORMANCE_UNOPTIMAL_WARNING() UME_SIMD_STATIC_WARN(false, "This function is not optimized");
    #define UME_UNIMPLEMENTED_WARNING() UME_SIMD_STATIC_WARN(false, "This function is not implemented!");

    #define UME_ALIGNMENT_CHECK(ptr, alignment) UME_SIMD_DYNAMIC_WARN(  (( ((uint64_t)ptr) % alignment) != 0), "Warning: unaligned memory used!\n")
}
}
#else
    #define UME_EMULATION_WARNING()
    #define UME_PERFORMANCE_UNOPTIMAL_WARNING()
    #define UME_UNIMPLEMENTED_WARNING()

    #define UME_ALIGNMENT_CHECK(ptr, alignment)
#endif

#define UME_SIMD UME_SIMD_NATIVE
// Just fall through. We are including this emulation regardless of library selected.
#if UME_SIMD == UME_SIMD_NATIVE 
    #if defined __AVX512F__
        #include "plugins/UMESimdPluginNativeAVX512.h"
    #elif defined (__MIC__)
        #include "plugins/UMESimdPluginNativeKNC.h"
    #elif defined __AVX2__
        #include "plugins/UMESimdPluginNativeAVX2.h"
    #elif defined __AVX__
        #include "plugins/UMESimdPluginNativeAVX.h"
    #else
        // Use scalar emulation if not specializing
        #define USE_EMULATED_TYPES 1
    #include "plugins/UMESimdPluginScalarEmulation.h"
    #endif // INSTRUCTION SET
#define USE_EMULATED_TYPES 1
#include "plugins/UMESimdPluginScalarEmulation.h"
#endif

// Traits need to be defined after all SIMD vectors are defined. 
#include "UMESimdTraits.h"

/**************************************************************************************************/
/* Define basic data structures useful for vector operations                                      */
/**************************************************************************************************/
#define INT_CEIL(N, D) (1 + ((N - 1) / D))

namespace UME
{
namespace SIMD
{

template<typename ELEMENT_TYPE>
class DynamicPaddedArray1D
{
public:
    const uint32_t LENGTH;
    const uint32_t PADDED_LENGTH;
    const uint32_t VECTOR_LENGTH;

    DynamicPaddedArray1D(uint32_t length, uint32_t vectorLength) :
        LENGTH(length),
        PADDED_LENGTH(INT_CEIL(length, vectorLength)*vectorLength),
        VECTOR_LENGTH(INT_CEIL(length, vectorLength))
    {
        uint32_t alignment = vectorLength*sizeof(ELEMENT_TYPE);
        mAllocationSize = PADDED_LENGTH*sizeof(ELEMENT_TYPE);
        mArray = (ELEMENT_TYPE*)UME::DynamicMemory::AlignedMalloc(mAllocationSize, alignment);
        memset(mArray, 0, mAllocationSize);
    }
    
    ~DynamicPaddedArray1D()
    {
        UME::DynamicMemory::AlignedFree(mArray);
    }

    // takes regular array and creates a padded copy
    void initializeWithArray(ELEMENT_TYPE* inArray)
    {
        for(uint32_t x = 0; x < LENGTH; x++)
        {                
            mArray[x] = inArray[x];
        }
    }

    void initializeWithValue(uint32_t value)
    {
        for(uint32_t x = 0; x < LENGTH; x++)
        {
            mArray[x] = value;
        }
        /* TODO: remove this from non-debug version */
        for(uint32_t x = LENGTH; x< PADDED_LENGTH; x++)
        {
            mArray[x] = 0;
        }
    }

    void storeInArray(ELEMENT_TYPE* outArray)
    {
        for(uint32_t x = 0; x < LENGTH; x++)
        {
            outArray[x] = mArray[x];
        }
    }

    ELEMENT_TYPE* getArrayPtr() { return mArray; }

private:
    ELEMENT_TYPE* mArray;
    uint32_t mAllocationSize;
};

template<typename ELEMENT_TYPE>
class DynamicPaddedArray2D
{
public:
    const uint32_t WIDTH;
    const uint32_t HEIGHT;
    const uint32_t PADDED_WIDTH;
    const uint32_t PADDED_HEIGHT;
    const uint32_t VECTOR_WIDTH;
    const uint32_t VECTOR_HEIGHT;

    DynamicPaddedArray2D(uint32_t width, uint32_t height, uint32_t vectorLength) :
        WIDTH(width),
        HEIGHT(height),
        PADDED_WIDTH(INT_CEIL(width, vectorLength)*vectorLength),
        PADDED_HEIGHT(INT_CEIL(height, vectorLength)*vectorLength),
        VECTOR_WIDTH(INT_CEIL(width, vectorLength)),
        VECTOR_HEIGHT(INT_CEIL(height, vectorLength))
    {
        uint32_t alignment = vectorLength*sizeof(ELEMENT_TYPE);
        mAllocationSize = PADDED_WIDTH*PADDED_HEIGHT*sizeof(ELEMENT_TYPE);
        mArray = (ELEMENT_TYPE*)UME::DynamicMemory::AlignedMalloc(mAllocationSize, alignment);
        memset(mArray, 0, mAllocationSize);
    }

    ~DynamicPaddedArray2D()
    {
        UME::DynamicMemory::AlignedFree(mArray);
    }

    // takes regular array and creates a padded copy
    void initializeWithArray(ELEMENT_TYPE* inArray)
    {
        for(uint32_t y = 0; y < HEIGHT; y++)
        {
            for(uint32_t x = 0; x < WIDTH; x++)
            {
                uint32_t originalOffset = WIDTH*y + x;
                uint32_t paddedOffset = PADDED_WIDTH*y + x;
                
                mArray[paddedOffset] = inArray[originalOffset];
            }
        }
    }

    void initializeWithValue(ELEMENT_TYPE value)
    {
        for(uint32_t y = 0; y < HEIGHT; y++)
        {
            for(uint32_t x = 0; x < WIDTH; x++)
            {
                mArray[PADDED_WIDTH*y + x] = value;
            }
            /* TODO: remove this from non-debug version */
            for(uint32_t x = WIDTH; x<PADDED_WIDTH; x++)
            {
                mArray[PADDED_WIDTH*y + x] = 0;
            }
        }

        for(uint32_t y = HEIGHT; y < PADDED_HEIGHT; y++)
        {
            for(uint32_t x = 0; x < PADDED_WIDTH; x++)
            {
                mArray[PADDED_WIDTH*y + x] = 0;
            }

        }
    }

    void storeInArray(ELEMENT_TYPE* outArray)
    {
        for(uint32_t y = 0; y < HEIGHT; y++)
        {
            for(uint32_t x = 0; x < WIDTH; x++)
            {
                uint32_t originalOffset = WIDTH*y + x;
                uint32_t paddedOffset = PADDED_WIDTH*y + x;

                outArray[originalOffset] = mArray[paddedOffset];
            }
        }
    }

    ELEMENT_TYPE* getArrayPtr() { return mArray; }  // mArray has to be accessed using

private:
    ELEMENT_TYPE* mArray;
    uint32_t mAllocationSize;
};


} // namespace SIMD
} // namespace UME

#endif

