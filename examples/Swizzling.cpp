#include <ume/simd>

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

using namespace UME::SIMD;

int main() {
    SIMDVec<float, 4> v0(1.0f, 2.0f, 3.0f, 4.0f); // Initialize input vector
    SIMDSwizzle<4> s0(0, 2, 1, 3);       // Initialize indices
    SIMDVec<float, 4> v1, v2, v3;         // Initialize output vectors

    v1 = v0.swizzle(s0); // Permute using run-time defined swizzle vector

    v2 = v0.template swizzle<3, 2, 0, 1>(); // Permute using compile-time defined permutation

    v3 = v0.template swizzle<0, 0, 0, 0>(); //

    printVector(v0);
    printVector(v1);
    printVector(v2);
    printVector(v3);

    return 0;
}
