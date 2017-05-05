
**NOTE**: UME::Vector library has been moved to github! Please see: [https://github.com/edanor/umevector](https://github.com/edanor/umevector)

[![Build Status](https://travis-ci.org/edanor/umesimd.svg?style=flat-square)](https://travis-ci.org/edanor/umesimd)

**Current stable release is: v0.8.1**  
**To checkout stable release use:**  
 > git clone https://edanor@bitbucket.org/edanor/umesimd.git  
 > git checkout tags/v0.8.1

UME::SIMD is an explicit vectorization library. The library defines homogeneous interface for accessing functionality of SIMD registers of AVX, AVX2, AVX512 and IMCI (KNCNI, k1om) instruction set. 

You can find the most recent documentation and tutorials here: [UME::SIMD tutorials](https://gain-performance.com/ume/).   
Also a link to older wiki(deprecated): [wiki pages](https://bitbucket.org/edanor/umesimd/wiki/Home).

For quotations please refer to: [A high-performance portable abstract interface for explicit SIMD vectorization](http://dl.acm.org/citation.cfm?id=3026939)
  
This piece of code was developed as part of ICE-DIP project at CERN:  
 >   "ICE-DIP is a European Industrial Doctorate project funded by the 
 >   European Community's 7th Framework programme Marie Curie Actions under grant
 >   PITN-GA-2012-316596".

 All questions should be submitted using the bug tracking system:

   >   [bug tracker](https://bitbucket.org/edanor/umesimd/issues)


or by sending e-mail to:

   >   przemyslaw.karpinski@cern.ch

   
**RELEASE NOTES for v0.8.1**  

Interface:  
    -  
Performance tuning:  
    -  
Benchmarks:  
    - Add VS2015 solution for benchmarks.  
  
Fixes:  
    - remove unnecessary include in explog.  
    - fix explog to use more portable reinterpret-cast  
    
Tests:  
    -  

Other:  
    - Update Readme  
