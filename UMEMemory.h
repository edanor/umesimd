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

#ifndef MEMORY_H_
#define MEMORY_H_

#include <cstring>
#include <cstdlib>

#if defined (_MSC_VER)
//#define UME_ALIGN(alignment) __declspec(align(alignment))
    #if (_MSC_VER <= 1900)
        // Visual studio until 2015 is not supporting standard 'alignas' keyword
        #ifdef alignas
            // This check can be removed when verified that for all other versions alignas works as requested
            #error "UME error: alignas already defined" 
        #else
            #define alignas(alignment) __declspec(align(alignment))
        #endif
    #endif
#elif defined (__GNUC__)
#define UME_ALIGN(alignment) __attribute__ ((aligned(alignment)))
#elif defined (__ICC) || defined(__INTEL_COMPILER)
#endif

#define ALIGNED_TYPE(type, alignment) typedef type UME_ALIGN(alignment)

namespace UME
{

    class DynamicMemory
    {
    public:
        static inline void* Malloc(std::size_t size)
        {
            // TODO: specialize it depending on the architecture and OS
            return std::malloc(size);
        }

        static inline void* AlignedMalloc(std::size_t size, std::size_t alignment)
        {
#if defined(_MSC_VER)
            return _aligned_malloc(size, alignment);
#elif defined(__GNUC__) || (__ICC) || defined(__INTEL_COMPILER)
            void* memptr;
            posix_memalign( &memptr, alignment, size); 
            return memptr;
#endif
        }

        static inline void Free(void *ptr)
        {
            // TODO: specialize it depending on the architecture and OS
            std::free(ptr);
        }

        static inline void AlignedFree(void *ptr)
        {
#if defined(_MSC_VER)
            _aligned_free(ptr);
#elif defined(__GNUC__) || (__ICC) || defined(__INTEL_COMPILER)
            free(ptr);
#endif
        }

        static inline void* MemCopy(void *dst, void *src, size_t num)
        {
            // TODO: specialize it depending on the architecture and OS
            return std::memcpy(dst, src, num);
        }
        static inline void MemSet(void *dst, int ch, std::size_t count)
        {
            // TODO: specialize it depending on the architecture and OS
            std::memset(dst, ch, count);
        }
    };


}


#endif
