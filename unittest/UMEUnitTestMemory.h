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

#ifndef UME_UNIT_TEST_MEMORY_H_

#include "UMEUnitTestCommon.h"
#include "../UMEMemory.h"

template<typename SCALAR_T, int SIMD_STRIDE>
void generic_AlignedAllocatorTest(std::string const & scalar_type) {
    
    UME::AlignedAllocator<SCALAR_T, SIMD_STRIDE> allocator;
    
    SCALAR_T* mem = allocator.allocate(sizeof(SCALAR_T)*100);
    //SCALAR_T* mem = nullptr;
    int alignment = UME::SIMD::SIMDVec<SCALAR_T, SIMD_STRIDE>::alignment();
    bool isAligned = ((uint64_t(mem) % alignment) == 0);
    bool isNullptr = (mem == nullptr);
    std::string msg = "ALLOCATOR <";
    msg.append(scalar_type);
    msg.append(", ");
    msg.append(std::to_string(SIMD_STRIDE));
    msg.append(">");
    check_condition(isAligned && !isNullptr, msg.c_str());
    
    allocator.deallocate(mem, sizeof(SCALAR_T)*100);
}

int test_allocators(bool supressMessages)
{
    char header[] = "UME::AlignedAllocator test";
    INIT_TEST(header, supressMessages);
    
    generic_AlignedAllocatorTest<uint8_t, 1> (std::string("uint8_t"));
    generic_AlignedAllocatorTest<uint8_t, 2> (std::string("uint8_t"));
    generic_AlignedAllocatorTest<uint8_t, 4> (std::string("uint8_t"));
    generic_AlignedAllocatorTest<uint8_t, 8> (std::string("uint8_t"));
    generic_AlignedAllocatorTest<uint8_t, 16> (std::string("uint8_t"));
    generic_AlignedAllocatorTest<uint8_t, 32> (std::string("uint8_t"));
    generic_AlignedAllocatorTest<uint8_t, 64> (std::string("uint8_t"));
    generic_AlignedAllocatorTest<uint8_t, 128> (std::string("uint8_t"));
    generic_AlignedAllocatorTest<int8_t, 1> (std::string("int8_t"));
    generic_AlignedAllocatorTest<int8_t, 2> (std::string("int8_t"));
    generic_AlignedAllocatorTest<int8_t, 4> (std::string("int8_t"));
    generic_AlignedAllocatorTest<int8_t, 8> (std::string("int8_t"));
    generic_AlignedAllocatorTest<int8_t, 16> (std::string("int8_t"));
    generic_AlignedAllocatorTest<int8_t, 32> (std::string("int8_t"));
    generic_AlignedAllocatorTest<int8_t, 64> (std::string("int8_t"));
    generic_AlignedAllocatorTest<int8_t, 128> (std::string("int8_t"));
    generic_AlignedAllocatorTest<uint16_t, 1> (std::string("uint16_t"));
    generic_AlignedAllocatorTest<uint16_t, 2> (std::string("uint16_t"));
    generic_AlignedAllocatorTest<uint16_t, 4> (std::string("uint16_t"));
    generic_AlignedAllocatorTest<uint16_t, 8> (std::string("uint16_t"));
    generic_AlignedAllocatorTest<uint16_t, 16> (std::string("uint16_t"));
    generic_AlignedAllocatorTest<uint16_t, 32> (std::string("uint16_t"));
    generic_AlignedAllocatorTest<uint16_t, 64> (std::string("uint16_t"));
    generic_AlignedAllocatorTest<int16_t, 2> (std::string("int16_t"));
    generic_AlignedAllocatorTest<int16_t, 4> (std::string("int16_t"));
    generic_AlignedAllocatorTest<int16_t, 8> (std::string("int16_t"));
    generic_AlignedAllocatorTest<int16_t, 16> (std::string("int16_t"));
    generic_AlignedAllocatorTest<int16_t, 32> (std::string("int16_t"));
    generic_AlignedAllocatorTest<int16_t, 64> (std::string("int16_t"));
    generic_AlignedAllocatorTest<uint32_t, 1> (std::string("uint32_t"));
    generic_AlignedAllocatorTest<uint32_t, 2> (std::string("uint32_t"));
    generic_AlignedAllocatorTest<uint32_t, 4> (std::string("uint32_t"));
    generic_AlignedAllocatorTest<uint32_t, 8> (std::string("uint32_t"));
    generic_AlignedAllocatorTest<uint32_t, 16> (std::string("uint32_t"));
    generic_AlignedAllocatorTest<uint32_t, 32> (std::string("uint32_t"));
    generic_AlignedAllocatorTest<int32_t, 1> (std::string("int32_t"));
    generic_AlignedAllocatorTest<int32_t, 2> (std::string("int32_t"));
    generic_AlignedAllocatorTest<int32_t, 4> (std::string("int32_t"));
    generic_AlignedAllocatorTest<int32_t, 8> (std::string("int32_t"));
    generic_AlignedAllocatorTest<int32_t, 16> (std::string("int32_t"));
    generic_AlignedAllocatorTest<int32_t, 32> (std::string("int32_t"));
    generic_AlignedAllocatorTest<uint64_t, 1> (std::string("uint64_t"));
    generic_AlignedAllocatorTest<uint64_t, 2> (std::string("uint64_t"));
    generic_AlignedAllocatorTest<uint64_t, 4> (std::string("uint64_t"));
    generic_AlignedAllocatorTest<uint64_t, 8> (std::string("uint64_t"));
    generic_AlignedAllocatorTest<uint64_t, 16> (std::string("uint64_t"));
    generic_AlignedAllocatorTest<int64_t, 1> (std::string("int64_t"));
    generic_AlignedAllocatorTest<int64_t, 2> (std::string("int64_t"));
    generic_AlignedAllocatorTest<int64_t, 4> (std::string("int64_t"));
    generic_AlignedAllocatorTest<int64_t, 8> (std::string("int64_t"));
    generic_AlignedAllocatorTest<int64_t, 16> (std::string("int64_t"));
    
    generic_AlignedAllocatorTest<float, 1> (std::string("float(32b)"));
    generic_AlignedAllocatorTest<float, 2> (std::string("float(32b)"));
    generic_AlignedAllocatorTest<float, 4> (std::string("float(32b)"));
    generic_AlignedAllocatorTest<float, 8> (std::string("float(32b)"));
    generic_AlignedAllocatorTest<float, 16> (std::string("float(32b)"));
    generic_AlignedAllocatorTest<float, 32> (std::string("float(32b)"));
    generic_AlignedAllocatorTest<double, 1> (std::string("double(64b)"));
    generic_AlignedAllocatorTest<double, 2> (std::string("double(64b)"));
    generic_AlignedAllocatorTest<double, 4> (std::string("double(64b)"));
    generic_AlignedAllocatorTest<double, 8> (std::string("double(64b)"));
    generic_AlignedAllocatorTest<double, 16> (std::string("double(64b)"));
    
    return g_failCount;
}
#endif
