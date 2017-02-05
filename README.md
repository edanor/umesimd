
**NOTE**: UME::Vector library has been moved to github! Please see: [https://github.com/edanor/umevector](https://github.com/edanor/umevector)

[![Build Status](https://travis-ci.org/edanor/umesimd.svg?style=flat-square)](https://travis-ci.org/edanor/umesimd)

**Current stable release is: v0.7.1-stable**  
**To checkout stable release use:**  
 > git clone https://edanor@bitbucket.org/edanor/umesimd.git  
 > git checkout tags/v0.7.1-stable


UME::SIMD is an explicit vectorization library. The library defines homogeneous interface for accessing functionality of SIMD registers of AVX, AVX2, AVX512 and IMCI (KNCNI, k1om) instruction set. 

Draft of the UME::SIMD specification: [UME::SIMD spec](https://gainperformance.files.wordpress.com/2016/11/ume_simd-interface_v0_5.pdf)

This piece of code was developed as part of ICE-DIP project at CERN.

 "ICE-DIP is a European Industrial Doctorate project funded by the 
 European Community's 7th Framework programme Marie Curie Actions under grant
 PITN-GA-2012-316596".

 All questions should be submitted using the bug tracking system:


   >   [bug tracker](https://bitbucket.org/edanor/umesimd/issues)


or by sending e-mail to:


   >   przemyslaw.karpinski@cern.ch


Please refer to the wiki for introduction and additional information:


   >   [wiki pages](https://bitbucket.org/edanor/umesimd/wiki/Home)


**RELEASE NOTES for v0.7.1-stable**
Interface:
    - Add swizzle for compile-time permutations.


Performance tuning:
    - AVX2, AVX512: Swizzle specializations
    - Scalar: Add SIMD1/2x64 specializations.
    - ARM: SIMD64x2: simplified emulation.
 
Benchmarks:
    - Fix makefiles to compile with different comiler versions. 
    - Add 'vertical FIR' microbenchmark. 
    - Mandelbrot: Fix AVX512 intrinsic compatibility problem.
    - QuadSolver: Update benchmarks with AVX512 intrinsic codes.

Fixes:
    - Do not redefine alignas() in VS2015,  Intel compiler and clang compiler set _MSC_VER=1900
    - Various swizzle errors
    - [Issue #46] Update copyright headers to 2015-2017. 
    - [Issue #57]: Rename TRUE/FALSE macros in AVX and AVX2. 
    - [Issue #54]: Fix compilation errors by WA on GCC version. 
    - [Issue #55]: [AVX512] Unsigned 64bit-integer comparison fails 
    - [Issues #48, #50] Fixes for IntermediateIndex handling.
    - [Scalar]: incorret return type from IMAX/IMIN
    
Tests:
    - add Travis automation
    - Update 'gold' image for unittest results.


Other:
    - Add simple installation cmake file. 
    - [Issue #58] Add version macros to the main header.
    - Update licence header dates.

