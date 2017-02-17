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

#ifndef MATMUL_COMMON_H_
#define MATMUL_COMMON_H_

//#define UME_SIMD_SHOW_EMULATION_WARNINGS 1
#include <ume/simd>
#include "../utilities/TimingStatistics.h"

template<typename FLOAT_T>
struct RESULTS {
    TIMING_RES elapsed;
    FLOAT_T    RMS_error;
};

// For non-padded array representations
template<typename FLOAT_T, int MAT_RANK>
UME_NEVER_INLINE FLOAT_T calculate_RMS_error_scalar(FLOAT_T* A, FLOAT_T* B, FLOAT_T* C) {
    FLOAT_T error = FLOAT_T(0);
    FLOAT_T C_ij;
    for (int i = 0; i < MAT_RANK; i++) {
        // For each element in a row of C
        for (int j = 0; j < MAT_RANK; j++) {
            C_ij = 0;
            for (int k = 0; k < MAT_RANK; k++) {
                C_ij += A[i*MAT_RANK + k] * B[k*MAT_RANK + j];
            }
            error += (C[i*MAT_RANK + j] - C_ij)*(C[i*MAT_RANK + j] - C_ij);
        }
    }
    return std::sqrt(error / (FLOAT_T(MAT_RANK)*FLOAT_T(MAT_RANK)));
}

// Take padding into consideration
template<typename FLOAT_T, int MAT_RANK, int SIMD_STRIDE>
UME_NEVER_INLINE FLOAT_T calculate_RMS_error_SIMD(FLOAT_T* A, FLOAT_T* B, FLOAT_T* C) {
    FLOAT_T error = FLOAT_T(0);
    FLOAT_T C_ij;

    int PADDING = SIMD_STRIDE - (MAT_RANK % SIMD_STRIDE);

    for (int i = 0; i < MAT_RANK; i++) {
        // For each element in a row of C
        for (int j = 0; j < MAT_RANK; j++) {
            C_ij = 0;
            for (int k = 0; k < MAT_RANK; k++) {
                C_ij += A[i*(MAT_RANK + PADDING) + k] * B[k*MAT_RANK + j];
            }
            error += (C[i*(MAT_RANK)+j] - C_ij)*(C[i*(MAT_RANK)+j] - C_ij);
        }
    }
    return std::sqrt(error / (FLOAT_T(MAT_RANK)*FLOAT_T(MAT_RANK)));
}


#endif
