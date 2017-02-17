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

#include <ume/simd>

using namespace UME::SIMD;

template<typename VEC_T>
void printVector(VEC_T & v, const char * prefix) {
    std::cout << prefix << " ";
    for(int i = 0; i < VEC_T::length(); i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}

//   First test function uses a 'Member Function Interface' (MFI). All operations
// permissible on vector types are exposed using this interface. The interface
// is inherited from a scalar emulation engine which provides functional
// support for all operations not implemented using any particular instruction
// set technology (e.g. vector intrinsic functions).
SIMD4_64f test(SIMD4_64f & a, SIMD4_64f & b) {
    return a.add(b.mul(2.0f));
}

//   Second test function uses operator arithmetics. This is the most wanted mode
// of operations however it doesn't expose whole functionality of the MFI. A very
// important group of operations that are not permitted using operator syntax,
// are masking operations (e.g. MMUL - masked multiplication). Another are reduction
// operations (e.g. HADD - horizontal addition). For some applications operator
// syntax might be enough, while for others it will also
//   Operator syntax does not allow mixing scalar/SIMD types. A vector with broadcasted
// scalar value needs to be created before that (e.g. using broadcast constructor).
SIMD4_64f test2(SIMD4_64f & a, SIMD4_64f & b) {
    return a + SIMD4_64f(2.)*b;
}

//   Third test function shows usage of 'Fused Multiply and Add' operation. FMA are
// a group of specialized instructions performing multiple arithmetic operations.
// This set of instructions have been introduced so that common computational patterns
// can be accelerated.
SIMD4_64f test3(SIMD4_64f & a, SIMD4_64f & b) {
    return b.fmuladd(SIMD4_64f(2.), a);
}

int main()
{
    SIMD4_64f a(5., 3., 8., 4.);
    SIMD4_64f b(13.23, 984.91, -13.42, -0.000001);
    SIMD4_64f c, d, e;

// Labels are useful for searching through assembly listings.
test_start:
    c = test(a, b);
test_end:
    printVector(c, "c:" );

test2_start:
    d = test2(a, b);
test2_end:
    printVector(d, "d:" );

test3_start:
    e = test3(a, b);
test3_end:
    printVector(e, "e:" );

    return 0;
}
