#include "../../UMESimd.h"

int main()
{
    std::cout << "Native (default) vector lengths:\n";
    
    std::cout << "uint8_t    : " << UME::SIMD::ISATraits::NativeLength<uint8_t>() << "\n";
    std::cout << "uint16_t   : " << UME::SIMD::ISATraits::NativeLength<uint16_t>() << "\n";
    std::cout << "uint32_t   : " << UME::SIMD::ISATraits::NativeLength<uint32_t>() << "\n";
    std::cout << "uint64_t   : " << UME::SIMD::ISATraits::NativeLength<uint64_t>() << "\n";
    std::cout << "int8_t     : " << UME::SIMD::ISATraits::NativeLength<int8_t>() << "\n";
    std::cout << "int16_t    : " << UME::SIMD::ISATraits::NativeLength<int16_t>() << "\n";
    std::cout << "int32_t    : " << UME::SIMD::ISATraits::NativeLength<int32_t>() << "\n";
    std::cout << "int64_t    : " << UME::SIMD::ISATraits::NativeLength<int64_t>() << "\n";
    std::cout << "float(32b) : " << UME::SIMD::ISATraits::NativeLength<float>() << "\n";
    std::cout << "double(64b): " << UME::SIMD::ISATraits::NativeLength<double>() << "\n";

    return 0;
}