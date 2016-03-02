#include "../UMESimd.h"

// When developing a templated code, it is often easier to use 
// scalar constants. The problem with scalar constants is, that there
// are multiple representations of constants.

template<typename SCALAR_T>
SCALAR_T scalarOffsetInit(SCALAR_T d)
{
    // Removing explicit type-cast from this constant
    // will result in implicit conversion.
    // Compilers might issue a warning about this behaviour.
    return SCALAR_T(1.0) + d;
}

template<typename VEC_T, typename SCALAR_T>
VEC_T offsetInit(SCALAR_T d)
{
    // Because vector types are generalization of scalar types,
    // the same C++ rules apply. This function uses cast similar to
    // above to solve problem of implicit constant conversion for
    // case in which vectors are used.
    return VEC_T(SCALAR_T(1.0) + d);
}

template<typename VEC_T>
void printVector(VEC_T const & x) {
    // If a certain type is not used in prototype of the template function,
    // it is not necessary to pass it using template type-list.
    // Instead, it is possible to get specific type from a helper traits
    // class 'SIMDTraits' to be used in function scope. The SIMDTraits class defines 
    // typedefs for types that might be helpful when operating on a particular vector type.
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T SCALAR_T;

    for (unsigned int i = 0; i < VEC_T::length(); i++) {
        SCALAR_T x_i = x[i];
        std::cout << "vec[" << i << "] = " << x_i << "\n";
    }
    std::cout << std::endl;
}

int main()
{
    float s0 = scalarOffsetInit(3.14f);
    std::cout << "s0 : " << s0 << "\n";
    double s1 = scalarOffsetInit(3.14);
    std::cout << "s1 : " << s1 << "\n\n";

    UME::SIMD::SIMD4_32f v0(1.0f, 2.0f, 3.0f, 4.0f);
    std::cout << "SIMD4_32f before:\n";
    printVector(v0);
    v0 = offsetInit<UME::SIMD::SIMD4_32f, float>(3.2f);
    std::cout << "SIMD4_32f after:\n";
    printVector(v0);

    UME::SIMD::SIMD4_64f v1(1.0, 2.0, 3.0, 4.0);
    std::cout << "SIMD4_64f before:\n";
    printVector(v1);
    v1 = offsetInit<UME::SIMD::SIMD4_64f, double>(3.2);
    std::cout << "SIMD4_64f after:\n";
    printVector(v1);

    return 0;
}