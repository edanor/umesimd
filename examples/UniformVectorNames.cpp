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

/* One problem of basic naming convention is the fact that user needs to use
   SIMDVec_u, SIMDVec_i, and SIMDVec_f names for vectors of unsinged integer,
   signed integer and floating point elements. These types are then typedefed into
   corresponding SIMD<Length>_<scalar_type_suffix>. For example 

        SIMDVec_f<float, 4> <=> SIMD4_32f
        SIMDVec_u<uint8_t, 128> <=> SIMD128_8u

   For some applications it might be better to have a uniform name, and specify
   scalar type and vector length as template parameters, e.g.

        SIMDVec<float, 4> <=> SIMDVec_f<float, 4> <==> SIMD4_32f
        SIMDVec<int32_t, 8> <=> SIMDVec_i<int32_t, 8> <==> SIMD8_32i

    UME::SIMD allows all three naming conventions to be used and compatible with each
    other. 

    While mixing of naming conventions is possible, it is highly recommended to use only
    one of them in scope of a project. */

using namespace UME::SIMD;

int main() {
    SIMDVec<uint32_t, 4> v0(1, 2, 3, 4);    // Uniform name + template parameters
    SIMDVec_u<uint32_t, 4> v1(5, 6, 7, 8);  // Non-uniform names + template parmeters
    SIMD4_32u v2(254);                      // Unambiguous names (No template parameteres)

    std::cout << "v0:\n";
    v0.adda(v1);
    v0.sub(v2);
    printVector(v0);

    SIMDVec<int16_t, 1> v4(8);
    SIMDVec_i<int16_t, 1> v5(2);
    SIMD1_16i v6(9), v7;

    std::cout << "\nv7:\n";
    v7 = v6.fmuladd(v4, v5);
    printVector(v7);

    return 0;
}
