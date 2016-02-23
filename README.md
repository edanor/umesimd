**Current stable release is: v0.3.2-stable**  
**To checkout stable release use:**  
 > git clone https://edanor@bitbucket.org/edanor/umesimd.git  
 > git checkout tags/v0.3.2-stable


UME::SIMD is an explicit vectorization library. The library defines homogeneous interface for accessing functionality of SIMD registers of AVX, AVX2, AVX512 and IMCI (KNCNI, k1om) instruction set. 

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