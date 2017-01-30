
**NOTE**: UME::Vector library has been moved to github! Please see: [https://github.com/edanor/umevector](https://github.com/edanor/umevector)

[![Build Status](https://travis-ci.org/edanor/umesimd.svg?style=flat-square)](https://travis-ci.org/edanor/umesimd)

**Current stable release is: v0.6.1-stable**  
**To checkout stable release use:**  
 > git clone https://edanor@bitbucket.org/edanor/umesimd.git  
 > git checkout tags/v0.6.1-stable


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

   
**RELEASE NOTES for v0.6.1-stable**   
Interface:   
- Add GATHERU/SCATTERU (uniform stride gather/scatter   
- Add operators for LSH/RSH   
- Add LAND/LOR (logical AND/OR) for integer types   
- Add REM (division reminder) operation for integer types   
- Add SORTA/D (sort using Ascending/Descending order)   
- Add BANDNOT/LANDNOT (Bitwise/Logical AND-NOT)   
- Add COPYSIGN to interface     
- Allow SET-CONSTR to use scalar types other than SIMDVec base type   
- Add SCALAR_FLOAT_T to traits classes   
   
Performance tuning:   
- Add generalized, vectorized EXP   
- Add generalized, vectorized implementation for LOG/MLOG   
AVX:   
- add specializations for EXP & MEXP   
- SIMD8_32/64f: Use SVML for sin/cos functions   
AVX2:   
- Add specializations for EXP & MEXP   
AVX512:   
 - 32/64 u/i/f : M/GATHERU, M/SCATTERU   
 - 32/64 u/i/f : M/GATHERS/V, M/SCATTERS/V   
 - 32 u: GATHER/SCATTER   
 - Add specialization for EXP   
 - Specialized SIN/COS/SINCOS   
 - force inline on MFI functions.   
 - Bulk update (SIN/COS/EXP/LOG/LSH/RSH/FLOOR/CEIL)   
 - Bulk update mask types.   
 - Bulk performance upgrade.   
OPENMP:   
 - Added openmp plugin. The plugin can be forced by -DFORCE_OPENMP compilation flag   
   
Benchmarks:   
 - Increase benchmark automation by adding Makefiles and testing scripts   
 - Add compilation with forced scalar plugin   
 - Refactor microbenchmark codes to remove g++/clang++ warnings   
 - Create separate directories for all benchmarks   
 - Add matrix multiplication (matmul) benchmark (WIP)   
 - Add placeholder for openmp based implementation for mandelbrot2   
 - Add 'explog' microbenchmark to test EXP and LOGx operations   
   
Fixes   
 - AVX: Missing 'const' qualifier in operator-
 - AVX2: Fixes for failing tests: FTOU, SADD, SUBFROM   
 - AVX: IMIN/IMAX fix   
 - Fix MIMAX/MIMIN errors   
 - AVX512: SIMD16_32u: fixes for DIV
 - Scalar: force inline on plugin functions   
 - AVX512: Fixes for failsing MIN/MAX tests   
 - AVX512: GATHER/SCATTER fails   
 - COPYSIGN - scalar kernel updated   
 - Fix some build problems when using OpenMP plugin   
 - Assignment operators (+=, -=, /=) returning value and not reference   
 - AVX512: SIMD2_64f incorrect implementation for EXP   
   
Tests   
 - Use SVML when building with ICC   
 - Use random generated data sets for LSH/RSH   
 - Fix alignment problem in GATHER/SCATTER tests   
 - Add tests for IMIN/IMAX   
 - Replace Cmake build with makefile. Enable parallel compilation   
 - Add VS2015 solution for unit tests   
 - Add test for SWIZZLE   
 - Add tests for PACK, PACKLO, PACKHI, UNPACK, UNPACKLO, UNPACKHI   
 - Allow building with OpenMP plugins   
   
Internal code:   
- Move emulation warnings from scalar emulation functions to interface methods   
   
   
   
**RELEASE NOTES for v0.5.1-stable**   
Interface:  
- Fix function name for mask interface LOAD.  
- Inverse logic for BLEND operations.  
- Add Non-temporal load/store operations (SSTORE/SLOAD).  
  
Performance tuning:   
scalar:  
- Add specialized implementation for SIMD4/8x32.  
AVX:  
- SIMD4_64f + SIMDMask4  
- simplify ABS  
- AVX: enable performance for SIMDx_64  
AVX2:  
- SIMD4_64f + SIMDMask4  
AVX512:  
- AVX512: MASK4 + MASK8 add missing operators.  
- SIMD1_64f TRUNC/MTRUNC  
- SIMD16_64x   
  
  
Benchmarks:  
- Add QuadraticSolver microbenchmark.  
- Add 'SINCOS' benchmark.  
- Update displayed information.  
- Modifications to prohibit streaming-stores optimization.  
  
Fixes:   
- KNC: add missing 'const' function qualifiers.  
- Incorrect mask used for write mask operator.  
- AVX: Incorrect logic for CMPLT  
- SIMD8_64f replace fast reciprocal with precise one.  
- AVX2: incorrect intermediate mask used when '()' enabled  
- FIX: Saturated addition scalar emulation kernel.  
- AVX: SIMD16_64f - use unaligned load instructions.  
- AVX: FTOI - use C++ compatible conversions.  
- AVX: ROUND - use double precision version of std::round  
- AVX: SIMD4_64f: CMPEQ/CMPNE incorrect masks returned.  
- AVX2: fix unitialized memory bug in avx2 hland function  
- Fix compilation errors using GCC/Clang  
- AVX512: Incorrect kernels for pack/unpack  
  
Tests:  
- Use random generated tests for badly defined scenarios.  
- Add unit tests for MLOAD/MSTORE  
  
Internal code:  
- Force inlining on interface and emulation. New defines: UME_FORCE_INLINE, UME_NEVER_INLINE  
- Remove declspec from interface emulation.  
- Add template specialization forward declarations.  
- Split emulation into pure scalar and vector based.  
- Propagate scalar emulation changes to plugins.  
  
**RELEASE NOTES for v0.4.1-stable**   
  
Interface:   
- Faster ROL/ROR emulation using LOAD/STORE   
- Aliases for vector types. Now possible to use SIMDVec<BASE_T, VEC_LEN> instead of SIMDVec_u/i/f<BASE_T, VEC_LEN>   
- Added non-member function interface. It is now possible to do:   
        
        add(vec_a, vec_b);
        
instead of:
    
        vec_a.add(vec_b);   
   
   
Performance tuning:   
- Major updates for AVX, AVX2 and AVX512.
   
Benchmarks:   
- extended mandelbrot benchmark with 64b floating point implementation   
- added mandelbrot2 benchmark. This benchmark is based on code available at: https://software.intel.com/en-us/articles/introduction-to-intel-advanced-vector-extensions   
   
   
Fixes:   
- KNC: add missing 'const' function qualifiers.   
- KNL: MULV - incorrect temporaries.   
- Fix compilation warnings with -Wall (GCC/ICC).   
- Fix multiple errors in unit test data sets.   
- Fix narrowing conversion errors in unit test data sets.   
   
Examples:   
- Add example using scalar constant literals in templates.   
   


**RELEASE NOTES for v0.3.2-stable**

Interface:
- reintroduced mask-assignment operations on masks
- gather scatter using scalar types of correlated precision

Performance tuning:
AVX:
- performance improvements: SIMDMask4, SIMD2_32x, SIMD4_32x, SIMD8_32x
AVX2:
- performance improvements: SIMDMask4, SIMD2_32x, SIMD4_32x, SIMD8_32x
AVX512:
- missing operators SIMD4_32u
- performance improvements: SIMD4_32f

Benchmarks:
- extend benchmarks with uniform statistics
- statistics calculate also 90% and 95% confidence intervals
   
**RELEASE NOTES for v0.3.1-stable**
   
Interface:   
- added PROMOTE/DEGRADE operations to convert between vectors using scalars of different precision (e.g. PROMOTE SIMD4_32f to SIMD4_64f)   
- added LOG, LOG2, LOG10 to floating point interface   
- added CMPEQS/CMPNES for masks   
- added compilation flag to switch between '[]' and '()' syntax for writemasks   
- added overloaded operators for mixed scalar<->vector operations (Issue #25)   
- added missing operator= (ISSUE #26)   
- added writemask operators for scalars (e.g. vec[mask] = scalar)   
   
Performance tuning:   
- AVX512 (SKX + KNL): SIMDMask4/8/32, SIMD4_32u/i/f, SIMD8_32u/i/f, SIMD32_32u/i/f, (extensive update)   
   
Bug fixes:   
- AVX2: SIMD4_32f missing 'const' qualifier in STORE   
- AVX: add missing definitions for float vectors   
- AVX512: separation between different AVX512 ISA variations   
- AVX512: AVX512: missing explicit in constructor of SIMDMask<8>   
   
   
Examples:   
- added basic example for SIMD vector showing MFI(Member Function Interface) and operator syntax.   
   
Tests:   
- added generic unit tests for SIMD using 64f, 8u/i and 16u/i scalar types   
   
Benchmarks:   
- added Latencies benchmark allowing monitoring of library performance with instruction-level granularity   
- added Mandelbrot Set benchmark   
   
Internal code:   
- fixed missing template<> for templated cast operators   
- extended NullTypes and eliminate SIMD1 template specializations; this change simplifies the plugin system and fixes loose ends of the typeset system;   
- remove 'final' class specifiers to allow custom extensions of SIMD types; this change allows using SIMD types as base classes for custom vectorization interfaces;
