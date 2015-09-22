This piece of code was developed as part of ICE-DIP project at CERN.
 "ICE-DIP is a European Industrial Doctorate project funded by the European Community's 
 7th Framework programme Marie Curie Actions under grant PITN-GA-2012-316596".

All questions should be submitted using the bug tracking system or by sending e-mail to:

   przemyslaw.karpinski@cern.ch
   
// --------------------------------
// TABLE OF CONTENTS
// --------------------------------

1. Introduction
2. Why to use UME::SIMD?
3. When not to use UME::SIMD?
4. Performance
5. Compatibility
6. Workflow
7. Quick start
8. Interface overview
 
// --------------------------------
// 1. INTRODUCTION
//--------------------------------


UME::SIMD is an explicit SIMD vectorization library for modern CPUs. 

The library is implemented using C++ 11 and so requires a compliant compiler.

Modern CPU architectures introduce concept of 'SIMD vector registers'. These registers are capable
of packing multiple data elements and performing a single instruction on all vector elements at 
the same time. Execution of SIMD code can bring a significant speedup over 'scalar code', that is
code executing on one data element ( a 'scalar' element) at a time.

'Explicit' vectorization refers to the software development process in which the programmer is
aware of vectorization capabilities and writes the code so that it utilises underlying hardware. 
This approach contradicts so called 'auto-vectorization' in which the compiler is responsible for
recognizing pieces of code subject for vectorization, and then performing certain optimizations 
resulting in generating vector instructions. In auto-vectorization model, the user doesn't have to
be aware of vectorization on CPU instruction set level. Unfortunately the auto-vectorization is 
not (yet!) very good and so there is a need to have other, more direct means of interacting with 
hardware architecture.

There are multiple problems with actual support for vectorization on different CPUs. Few of 
these problems are:

- Explicit vector programming requires from the user usage of assembly or 'vector intrinsic 
  functions'. Both methods are not portable (over different CPU or even compilers). Effectively 
  this makes it only possible to write short vectorized kernels instead of using vectorization 
  on the same scale as regular scalar code is used. 
  
- Not all simd vector types are supported in form of CPU SIMD registers. Since certain algorithms
  are only possible to execute using certain SIMD lengths, it is necessary for the user to create
  complicated workarounds.

- Not all operations a user would like to perform are supported for given vector types. This is 
  clearly a design flaw or engineering tradeof made during CPU design process. Regardless of reason
  the users face a problem of developing workarounds repetitively.

- It is not easy to write code that would work for both scalar and vector data types. Because the
  set of operations available on scalar types is different than set of operations available on
  SIMD types, the same code cannot be written for both scalar and SIMD types.

- It is not easy to write code for which we could easily modify vector type used. Compiler intrinsic
  functions are forcing the user to write code in a non-portable way. Whenever a user wants to change
  vector length or base element type, he is forced to re-write whole piece of code. While it is not
  always possible to write code that executes the same way with different SIMD lengths, there are 
  multiple occasions in which this can be necessary.

- It is not easy to run vectorized code on a non-vectorizing CPU. Both inline assembly and compiler
  intrinsics require compilation with specific architecture-dependant compiler flags. This makes it
  necessary for the users to use compile-time directives to either include or exclude specific fragments
  of code. This decreases the code maintainability.
  
- It is not easy to prepare vectorized code to be ran on future vectorizing CPUs. That means writing and
  debugging code that does exact the same thing, each time a new vectorizing CPU arises. 

// -------------------------------
// 2. WHY TO USE UME::SIMD?
// -------------------------------


UME::SIMD defines a set of hermetic data types that hide underlying vectorizing hardware from the 
user. While the library is using compiler intrinsics extensively, it is no longer necessary for the
programmer to understand how these intrinsics map to the code. User sees only UME::SIMD types and
has to operate only on these types. All types have well defined and wide interface so there is no
need (except for some really extreme, low level optimizations) to understand intrinsics code and to
understand the detailed capabilities of underlying hardware. While SIMD arithmetic itself is a
little bit different from what most of the programmers are used to write, it is no longer
complicated by the hardware complexity.

The library is introducing SIMD1 data types to be used in regular code. SIMD1 data type is
essentially a code running on one scalar element at a time with one exception: the SIMD1 data
container is able to use the same interface as other SIMD types!!! This makes it possible to write
code only once and run it either as scalar or SIMD code.
As some of the included microbenchmarks indicate, the performance of SIMD1 is very similar to
performance of equivalent scalar code. While this relation doesn't hold for all algorithms, it still
gives the users an additional tool for analysing slowdown resulting from using SIMD code.

By creating abstraction layer, it is possible to create workarounds for multiple problems such as:
missing vector types, missing ISA instructions or some hardware issues workarounds. Since operations
like this could impact the performance, the library can give compile time guidelines to the user about
potential problems with library performance.

Because the interface is exactly the same for all data types, there is no longer problem in place that
would forbid the user from writing reusable (e.g. templated) code. As presented in code examples, this
library is pretty handy in providing means for code reusability.

The library is very simple in use. All the users need to do is to include "UMESimd.h" file to their
project and enable C++ 11 functionality (-std=c++11) in their compiler. The vectorization extension used
is relying on some additional compiler flags, but the library will recognize them and select proper
implementation without any additional modifications to the project. In case that code will be compiled
without any vectorization enabled, the library will execute all operations in emulated mode and using
array of scalar types to represent vectors. While this can be really bad for performance, the compilers
are also very good in optimizing scalar code so the performance should be similar to one of regular
code.

Different CPU's use different instruction sets. Because explicit programming requires the user to 
write the code for all types of CPU 'explicitly', the UME::SIMD was designed so that it was possible
(and relatively easy) to add new CPU's to the supported list. This can be done by writing a plugin
and implementing whole interface for that specific instruction set. While number of instructions
to be overriden is overwhelming at first sight, the existing interface classes limit the amount of
code necessary to be written before the plugin can be used to only few hundred lines. Thanks to that
the further development can be done incrementaly, and some minimum necessary capabilities enabled in
matter of hours.

One of platforms this library is targeting in the first place is Intel Xeon Phi. Because of that
the support will be provided on similar level as for Xeon processors.

The library is released under MIT license. It is free for any type of application with the limitation 
of preserving the original authorship information. You can copy, redistribute, modify, delete and do
whatever you want with this code, for free. The license was chosen that way for few reasons:
1) I believe that introducing vector types is necessary for future evolution of compute and it shouldn't
   be blocked by the licensing problems. Because the library is "include like" it is necessary to 
   prevent any license spoiling for any project that is potentially using it.
   
2) This library can grow really large. Initial estimate is about 500000 lines of C++ code with heavy use
   of intrinsics and template metaprogramming. Because of that it cannot be well developed without some
   community support. 

3) This code is low-level enough to be useful in many domains, in both commercial and academic applications.
   Opening the source code for such library is a great opportunity to share some effort that would be 
   beneficial for everyone.

// -------------------------------
// 3. WHEN NOT TO USE UME::SIMD?
// -------------------------------


There are few trivial situations when you don't need to use SIMD vectorization:
1) "I don't need more performance from my application." 

   It is usually easier to stick to regular scalar code and only optimize whatever is 
   performance critical. If your project only needs speedup in one critical algorithm on one 
   specific platform it might be faster to just write intrinsic code. Although integrating UME::SIMD
   into a project is trivial, the compilation time will suffer due to extensive templatization usage.
   
2) "I want to program CUDA and other GP GPU devices."

   These devices have completely different approaches towards SIMD programming and overall hardware
   architecture. You can use UME::SIMD under some other abstraction layer to hide different
   hardware, but there are no plans on implementing a separate plugin for CUDA in this library.
   If you are interested in both CPU and CUDA, there is another library developed at CERN that has
   support for CUDA devices:
     
     https://github.com/VcDevel/Vc
     
   VC is pretty good in terms of performance on CPUs, but has some limited capabilities in terms of 
   supported vector types.
   
3) "I want the performance RIGHT NOW!"

   UME::SIMD is not yet mature with performance, although it should reach the top performance of
   other vectorization approaches in not-so-far future. If you still need some means to test
   your ideas, you can use existing explicit vectorization libraries such as Vector Code 
   library (VC) or Vector Class Library(VCL):
   
     https://github.com/VcDevel/Vc
     http://www.agner.org/optimize/#vectorclass
     
4) "I don't need portability, I just want to program KNC (Intel Knight's Corner) with fancy vector 
    classes". 
    
    Well there is a plugin for VCL that allows Vector code to be used on KNC that I developed last year.
    The code is merged with the original VCL and available at:
    
     https://bitbucket.org/edanor/vclknc_integrated
     
    The code acts the same way and uses the same approach as VCL. The VCL documentation from:
     
      http://www.agner.org/optimize/#vectorclass
     
    applies in general also for VCLKNC.
    
5) "I want to vectorize some standard containers without the changes of my code"

    Ther is YET ANOTHER explicit vectorization library: boost::simd available at:

       https://github.com/jfalcou/boost.simd

    This library targets improvements in boost and cooperation with existing boost components.
       
// -------------------------------
// 4. PERFORMANCE
// -------------------------------


Vectorization concepts were introduced into CPUs for one reason, and one reason only: performance.
This library is designed so that it is possible to extract as much as possible performance from CPUs.

The number of supported data types is large, and it is larger than number of data types supported as
SIMD register types in all existing vectorisation ISA. The reason for that is to give the user
the biggest flexibility in terms of software development as possible and to prepare code for execution
on new architectures that might be available in 3-5 years.

Developing intrinsic code for all (over 60!!!) vector types and for all supported instruction sets
is a time consuming task. Instead of ENABLING FUNCTIONALITY, this library gives full programming interface
using scalar types and will ENABLE PERFORMANCE over time.

Using extensive scalar emulation makes it possible to compile
UME::SIMD based code for all types of SIMD to test result correctness and to be able to perform
development even on platforms that don't support vectorization! Since the interface is close to
being complete (for first version of the library), the functionality will not change over time (hopefully...).
This will increase the portability of the code as well as code reusability.

// -------------------------------
// 5. COMPATIBILITY
// -------------------------------


The code is compiled on a regular basis using following compilers:
- MS Visual C++ compiler (CL v 17.00 and newer, available since MS Visual Studio 2012)
- GNU g++ compiler (4.8.2 and higher)
- INTEL C++ compiler (15.0Compiler Studio XE 201

In current version following instruction sets are supported and targeted for full implementation:
- AVX + FMA
- AVX2 + FMA
- AVX512 (With planned support for Intel Xeon Phi: Knight's Landing coprocessor)
- IMCI (Initial Many-Core Instructions, an Intel Xeon Phi: Knight's Corner coprocessor instruction set)

// NOTE: Code compiled for SSExxx will compile to scalar emulation. This is because we don't
//       plan to support SSE instruction set as a standalone. 

// NOTE: While current support is targeted for instruction sets developed for Intel processors
//       the interface can be developed also for other ISA (such as ARM NEON). 
//       If you think there is a visible need to develop support for other instruction sets
//       and you are willing to spend your resources (time or/and money) on that, feel invited.

// ------------------------------
// 6. WORKFLOW
// ------------------------------


The process of ENABLING PERFORMANCE will be performed over time. Unfortunately this can only
be done by first specializing every possible combination of:
    SIMD vector type (over 60) + Instruction set (4 currently) + operation (~250 operations per)

this gives in total 60000 overloadable member functions!

The overloading of a single member function can take as little as 1 line, and as much as 100 lines
of intrinsic code. Adding to that some testing code (average unit test length is ~5 lines of code)
this gives in total at least 300000 lines of code (most of it using intrinsics).

Because of limited resources of this project it is necessary to rely on community support.
If you are a user and you would like to improve code basis, but you don't have time to do
development, you can still help! All you have to do is to submit an issue in our tracking system with
[PERFORMANCE REQUEST] tag in the title. The issue should state:
 
 1. instruction set to be enabled
 2. data type to be optimized
 3. list of member functions that are required to be optimized
 4. number of cores (order of magnitude) you are targeting with your application
 
If a similar issue already exists, please add a comment saying that you are also interested in
such improvement.

The code is pretty buggy so far. This comes from the amount of combinations inside the library.
Most of the bugs should be pretty easy to resolve and were not caught yet because of small number
of unit tests developed. 
If you find a bug in the software, please submit a ticket with [BUG] tag in the title. If you
can provide list of 

 1. instruction set
 2. data type 
 3. member function 
 
that cause a problem or implement a unit test that exposes the bug - that would be great!

There are some problems that we are not planning to resolve as:
 - supporting previous lower versions of compilers
 - implementing the interface in other languages
 - adding support for CUDA
 - overloading C++ operators to perform some operations

If you think it is necessary to have certain functionality and it is still not there, please
feel free to ask submit a ticket or ask a direct question.

// ------------------------------
// 7. QUICK START
// ------------------------------

After downloading this repository you can try compiling and executing unit test code and 
some microbenchmarks provided with the library. The code doesn't contain any 'make' like file system.

1) Windows:
For windows users it is possible to create unit testing solution. Simply execute

  > cd unittest
  > create_solution_vs2012.bat

solution should be created in 'unittest/vs2012project'. Opening the solution and building 
can be performed as with regular solution.

To debug unit test solution 
    a) RMB on UMSSIMDUnitTest project
    b) select "Set as startup project"
    c) press F5 or select "Debug->Start Debugging"

To build for different instruction set:
    a) RMB on UMESIMDUnitTest project
    b) select "properties"
    c) "Configuration Properties->C/C++->Code Generation"
    d) Change "Enable Enhanced Instruction Set" field

2) Linux:
  $ cd unittest
  $ g++ UMESIMDUnitTest.cpp -std=c++11 -O2

// NOTE the same build can be performed using ICC, but an additional flag:
//
//    -fp-model precise
//
//  needs to be added to make sure the same floating point model is used as for g++.

To build for different instruction set use following flags:

GCC:
- FMA:  -mfma
- AVX:  -mavx
- AVX2: -mavx2
- AVX512: -mavx512f -mavx512pf -mavx512er -mavx512cd

ICC:
- FMA:    -fma
- AVX:    -xavx
- AVX2:   -xCORE-AVX2
- AVX512: -xCORE-AVX512

If you are able to compile unit tests and spot no fails, you can try compiling some of our
microbenchmarks:

   $ cd microbenchmars
   $ g++ average.cpp -std=c++11 -O2 
   $ g++ polynomial.cpp -std=c++11 -O2
   

If you want to use library in your own code, simply include

 UMESimd.h
 
file into your project. Data types can be then used by adding:

 using namespace UME::SIMD
 
or by prefixing name of data types with UME::SIMD::
   
// -------------------------------
// 8. INTERFACE OVERVIEW
//--------------------------------


This library uses a very simple concept of a SIMD vector being represented by a pair of:
    1) vector length (VEC_LEN) - that is a number of elements packed in a SIMD vector
    2) scalar type (SCALAR_TYPE) - the type of a scalar used to represent a single element in a vector.

This leads to a following naming convention for all SIMD vectors:
     SIMD<VEC_LEN>_<SCALAR_TYPE>

SCALAR_TYPE can be one of:
1) unsigned integer scalar type: uint8_t, uint16_t, uint32_t, uint64_t
2) signed integer scalar types: int8_t, int16_t, int32_t, int64_t
3) floating point scalar types: float (32b), double (64b)

In its current form the library defines all vector types of lengths up to 1024 bits.

According to that taxonomy, UMESIMD defines following data types:

    8 bit vectors:
        Unsigned integer      : SIMD1_8u
        Signed integer        : SIMD1_8i

    16 bit vectors:
        Unsigned integer      : SIMD2_8u, SIMD1_16u
        Signed integer        : SIMD2_8i, SIMD1_16i

    32 bit vectors:
        Unsigned integer      : SIMD4_8u, SIMD2_16u, SIMD1_32u
        Signed integer        : SIMD4_8i, SIMD2_16i, SIMD1_32i
        Floating point vectors: SIMD1_32f

    64 bit vectors:
        Unsigned integer      : SIMD8_8u, SIMD4_16u, SIMD2_32u, SIMD1_64u
        Signed integer        : SIMD8_8i, SIMD4_16i, SIMD2_32i, SIMD1_64i
        Floating point vectors: SIMD2_32f, SIMD1_64f

    128 bit vectors:
        Unsigned integer      : SIMD16_8u, SIMD8_16u, SIMD4_32u, SIMD2_64u
        Signed integer        : SIMD16_8i, SIMD8_16i, SIMD4_32i, SIMD2_64i
        Floating point vectors: SIMD4_32f, SIMD2_64f

    256 bit vectors:
        Unsigned integer      : SIMD32_8u, SIMD16_16u, SIMD8_32u, SIMD4_64u
        Signed integer        : SIMD32_8i, SIMD16_16i, SIMD8_32i, SIMD4_64i
        Floating point vectors: SIMD8_32f, SIMD4_64f

    512 bit vectors:
        Unsigned integer      : SIMD64_8u, SIMD32_16u, SIMD16_32u, SIMD8_64u
        Signed integer        : SIMD64_8i, SIMD32_16i, SIMD16_32i, SIMD8_64i
        Floating point vectors: SIMD16_32f, SIMD8_64f

    1024 bit vectors:
        Unsigned integer      : SIMD128_8u, SIMD64_16u, SIMD32_32u, SIMD16_64u
        Signed integer        : SIMD128_8i, SIMD64_16i, SIMD32_32i, SIMD16_64i
        Floating point vectors: SIMD32_32f, SIMD16_64f

As well as having full spectrum of SIMD vectors, the library also supports 'Mask' types and
'Swizzle Mask' types. Mask types are used widely in the library for conditional execution of
operations on SIMD vectors, and for comparison operations. Swizzle Mask types are used in 
swizzle (that is shuffling/reordering/permuting) operations.

These types are:

Mask types:
 SIMDMask1, SIMDMask2, SIMDMask4, SIMDMask8, SIMDMask16, SIMDMask32, SIMDMask64, SIMDMask128

Swizzle Mask types:
 SIMDSwizzle1, SIMDSwizzle2, SIMDSwizzle4, SIMDSwizzle8, SIMDSwizzle16, SIMDSwizzle32,
 SIMDSwizzle64, SIMDSwizzle128

All data types are implemented using C++ classes. All data types implement uniform interface
accessible as member functions of a particular data type class. Calling a member function uses
the object as on of operation operands. In general it is possible to execute the same operation
on all vector types, with certain restrictions:
- SIMD1 types dont implement PACK/UNPACK operations.
- The swizzle operations on SIMD1 are identity operations.
- Certain operations are restricted to Signed/floating point data types (e.g. square root of a
  number, sign negation)
- The masking operations require mask of the same length as SIMD vector. An example: SIMDMask4
  needs to be used when performing MADD (masked addition) operation on SIMD4_32u, SIMD4_32i, SIMD4_32f.

The library in its current form implements following operations (using interface codenames -
function prototypes can be found in SIMDInterface.h):

//NOTE: C++ interface definition will be provided later on. So far majority (about 99%) of the
//      inteface is available with scalar emulation, although not all combinations are tested.

1) Operations available on all vector types, mask types and swizzle mask types:
- LENGTH - returns length of vector (VEC_LEN)
- ALIGNMENT - returns required alignment for memory operations

2) Operations available on all SIMD vector types:

(Initialization)
- ZERO-CONSTR - Zero element constructor 
- SET-CONSTR  - One element constructor
- FULL-CONSTR - constructor with VEC_LEN scalar element 
- ASSIGNV     - Assignment with another vector
- MASSIGNV    - Masked assignment with another vector
- ASSIGNS     - Assignment with scalar
- MASSIGNS    - Masked assign with scalar

(Memory access)
- LOAD    - Load from memory (either aligned or unaligned) to vector 
- MLOAD   - Masked load from memory (either aligned or unaligned) to vector
- LOADA   - Load from aligned memory to vector
- MLOADA  - Masked load from aligned memory to vector
- STORE   - Store vector content into memory (either aligned or unaligned)
- MSTORE  - Masked store vector content into memory (either aligned or unaligned)
- STOREA  - Store vector content into aligned memory
- MSTOREA - Masked store vector content into aligned memory
- EXTRACT - Extract single element from a vector
- INSERT  - Insert single element into a vector

(Addition operations)
- ADDV     - Add with vector 
- MADDV    - Masked add with vector
- ADDS     - Add with scalar
- MADDS    - Masked add with scalar
- ADDVA    - Add with vector and assign
- MADDVA   - Masked add with vector and assign
- ADDSA    - Add with scalar and assign
- MADDSA   - Masked add with scalar and assign
- SADDV    - Saturated add with vector
- MSADDV   - Masked saturated add with vector
- SADDS    - Saturated add with scalar
- MSADDS   - Masked saturated add with scalar
- SADDVA   - Saturated add with vector and assign
- MSADDVA  - Masked saturated add with vector and assign
- SADDSA   - Satureated add with scalar and assign
- MSADDSA  - Masked staturated add with vector and assign
- POSTINC  - Postfix increment
- MPOSTINC - Masked postfix increment
- PREFINC  - Prefix increment
- MPREFINC - Masked prefix increment

(Subtraction operations)
- SUBV       - Sub with vector
- MSUBV      - Masked sub with vector
- SUBS       - Sub with scalar
- MSUBS      - Masked subtraction with scalar
- SUBVA      - Sub with vector and assign
- MSUBVA     - Masked sub with vector and assign
- SUBSA      - Sub with scalar and assign
- MSUBSA     - Masked sub with scalar and assign
- SSUBV      - Saturated sub with vector
- MSSUBV     - Masked saturated sub with vector
- SSUBS      - Saturated sub with scalar
- MSSUBS     - Masked saturated sub with scalar
- SSUBVA     - Saturated sub with vector and assign
- MSSUBVA    - Masked saturated sub with vector and assign
- SSUBSA     - Saturated sub with scalar and assign
- MSSUBSA    - Masked saturated sub with scalar and assign
- SUBFROMV   - Sub from vector
- MSUBFROMV  - Masked sub from vector
- SUBFROMS   - Sub from scalar (promoted to vector)
- MSUBFROMS  - Masked sub from scalar (promoted to vector)
- SUBFROMVA  - Sub from vector and assign
- MSUBFROMVA - Masked sub from vector and assign
- SUBFROMSA  - Sub from scalar (promoted to vector) and assign
- MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
- POSTDEC    - Postfix decrement
- MPOSTDEC   - Masked postfix decrement
- PREFDEC    - Prefix decrement
- MPREFDEC   - Masked prefix decrement

(Multiplication operations)
- MULV   - Multiplication with vector
- MMULV  - Masked multiplication with vector
- MULS   - Multiplication with scalar
- MMULS  - Masked multiplication with scalar
- MULVA  - Multiplication with vector and assign
- MMULVA - Masked multiplication with vector and assign
- MULSA  - Multiplication with scalar and assign
- MMULSA - Masked multiplication with scalar and assign

(Division operations)
- DIVV   - Division with vector
- MDIVV  - Masked division with vector
- DIVS   - Division with scalar
- MDIVS  - Masked division with scalar
- DIVVA  - Division with vector and assign
- MDIVVA - Masked division with vector and assign
- DIVSA  - Division with scalar and assign
- MDIVSA - Masked division with scalar and assign
- RCP    - Reciprocal
- MRCP   - Masked reciprocal
- RCPS   - Reciprocal with scalar numerator
- MRCPS  - Masked reciprocal with scalar
- RCPA   - Reciprocal and assign
- MRCPA  - Masked reciprocal and assign
- RCPSA  - Reciprocal with scalar and assign
- MRCPSA - Masked reciprocal with scalar and assign

(Comparison operations)
- CMPEQV - Element-wise 'equal' with vector
- CMPEQS - Element-wise 'equal' with scalar
- CMPNEV - Element-wise 'not equal' with vector
- CMPNES - Element-wise 'not equal' with scalar
- CMPGTV - Element-wise 'greater than' with vector
- CMPGTS - Element-wise 'greater than' with scalar
- CMPLTV - Element-wise 'less than' with vector
- CMPLTS - Element-wise 'less than' with scalar
- CMPGEV - Element-wise 'greater than or equal' with vector
- CMPGES - Element-wise 'greater than or equal' with scalar
- CMPLEV - Element-wise 'less than or equal' with vector
- CMPLES - Element-wise 'less than or equal' with scalar
- CMPEX  - Check if vectors are exact (returns scalar 'bool')

(Bitwise operations)
- ANDV   - AND with vector
- MANDV  - Masked AND with vector
- ANDS   - AND with scalar
- MANDS  - Masked AND with scalar
- ANDVA  - AND with vector and assign
- MANDVA - Masked AND with vector and assign
- ANDSA  - AND with scalar and assign
- MANDSA - Masked AND with scalar and assign
- ORV    - OR with vector
- MORV   - Masked OR with vector
- ORS    - OR with scalar
- MORS   - Masked OR with scalar
- ORVA   - OR with vector and assign
- MORVA  - Masked OR with vector and assign
- ORSA   - OR with scalar and assign
- MORSA  - Masked OR with scalar and assign
- XORV   - XOR with vector
- MXORV  - Masked XOR with vector
- XORS   - XOR with scalar
- MXORS  - Masked XOR with scalar
- XORVA  - XOR with vector and assign
- MXORVA - Masked XOR with vector and assign
- XORSA  - XOR with scalar and assign
- MXORSA - Masked XOR with scalar and assign
- NOT    - Negation of bits
- MNOT   - Masked negation of bits
- NOTA   - Negation of bits and assign
- MNOTA  - Masked negation of bits and assign

(Pack/Unpack operations - not available for SIMD1)
- PACK     - assign vector with two half-length vectors
- PACKLO   - assign lower half of a vector with a half-length vector
- PACKHI   - assign upper half of a vector with a half-length vector
- UNPACK   - Unpack lower and upper halfs to half-length vectors.
- UNPACKLO - Unpack lower half and return as a half-length vector.
- UNPACKHI - Unpack upper half and return as a half-length vector.

(Blend/Swizzle operations)
- BLENDV   - Blend (mix) two vectors
- BLENDS   - Blend (mix) vector with scalar (promoted to vector)
- BLENDVA  - Blend (mix) two vectors and assign
- BLENDSA  - Blend (mix) vector with scalar (promoted to vector) and assign
- SWIZZLE  - Swizzle (reorder/permute) vector elements
- SWIZZLEA - Swizzle (reorder/permute) vector elements and assign

(Reduction to scalar operations)
- HADD  - Add elements of a vector (horizontal add)
- MHADD - Masked add elements of a vector (horizontal add)
- HMUL  - Multiply elements of a vector (horizontal mul)
- MHMUL - Masked multiply elements of a vector (horizontal mul)
- HAND  - AND of elements of a vector (horizontal AND)
- MHAND - Masked AND of elements of a vector (horizontal AND)
- HOR   - OR of elements of a vector (horizontal OR)
- MHOR  - Masked OR of elements of a vector (horizontal OR)
- HXOR  - XOR of elements of a vector (horizontal XOR)
- MHXOR - Masked XOR of elements of a vector (horizontal XOR)

(Fused arithmetics)
- FMULADDV  - Fused multiply and add (A*B + C) with vectors
- MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
- FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
- MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
- FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
- MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
- FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
- MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors

(Mathematical operations)
- MAXV   - Max with vector
- MMAXV  - Masked max with vector
- MAXS   - Max with scalar
- MMAXS  - Masked max with scalar
- MAXVA  - Max with vector and assign
- MMAXVA - Masked max with vector and assign
- MAXSA  - Max with scalar (promoted to vector) and assign
- MMAXSA - Masked max with scalar (promoted to vector) and assign
- MINV   - Min with vector
- MMINV  - Masked min with vector
- MINS   - Min with scalar (promoted to vector)
- MMINS  - Masked min with scalar (promoted to vector)
- MINVA  - Min with vector and assign
- MMINVA - Masked min with vector and assign
- MINSA  - Min with scalar (promoted to vector) and assign
- MMINSA - Masked min with scalar (promoted to vector) and assign
- HMAX   - Max of elements of a vector (horizontal max)
- MHMAX  - Masked max of elements of a vector (horizontal max)
- IMAX   - Index of max element of a vector
- HMIN   - Min of elements of a vector (horizontal min)
- MHMIN  - Masked min of elements of a vector (horizontal min)
- IMIN   - Index of min element of a vector
- MIMIN  - Masked index of min element of a vector

(Gather/Scatter operations)
- GATHERS   - Gather from memory using indices from array
- MGATHERS  - Masked gather from memory using indices from array
- GATHERV   - Gather from memory using indices from vector
- MGATHERV  - Masked gather from memory using indices from vector
- SCATTERS  - Scatter to memory using indices from array
- MSCATTERS - Masked scatter to memory using indices from array
- SCATTERV  - Scatter to memory using indices from vector
- MSCATTERV - Masked scatter to memory using indices from vector

(Binary shift operations)
- LSHV   - Element-wise logical shift bits left (shift values in vector)
- MLSHV  - Masked element-wise logical shift bits left (shift values in vector) 
- LSHS   - Element-wise logical shift bits left (shift value in scalar)
- MLSHS  - Masked element-wise logical shift bits left (shift value in scalar)
- LSHVA  - Element-wise logical shift bits left (shift values in vector) and assign
- MLSHVA - Masked element-wise logical shift bits left (shift values in vector) and assign
- LSHSA  - Element-wise logical shift bits left (shift value in scalar) and assign
- MLSHSA - Masked element-wise logical shift bits left (shift value in scalar) and assign
- RSHV   - Logical shift bits right (shift values in vector)
- MRSHV  - Masked logical shift bits right (shift values in vector)
- RSHS   - Logical shift bits right (shift value in scalar)
- MRSHV  - Masked logical shift bits right (shift value in scalar)
- RSHVA  - Logical shift bits right (shift values in vector) and assign
- MRSHVA - Masked logical shift bits right (shift values in vector) and assign
- RSHSA  - Logical shift bits right (shift value in scalar) and assign
- MRSHSA - Masked logical shift bits right (shift value in scalar) and assign

(Binary rotation operations)
- ROLV   - Rotate bits left (shift values in vector)
- MROLV  - Masked rotate bits left (shift values in vector)
- ROLS   - Rotate bits right (shift value in scalar)
- MROLS  - Masked rotate bits left (shift value in scalar)
- ROLVA  - Rotate bits left (shift values in vector) and assign
- MROLVA - Masked rotate bits left (shift values in vector) and assign
- ROLSA  - Rotate bits left (shift value in scalar) and assign
- MROLSA - Masked rotate bits left (shift value in scalar) and assign
- RORV   - Rotate bits right (shift values in vector)
- MRORV  - Masked rotate bits right (shift values in vector) 
- RORS   - Rotate bits right (shift values in scalar)
- MRORS  - Masked rotate bits right (shift values in scalar) 
- RORVA  - Rotate bits right (shift values in vector) and assign 
- MRORVA - Masked rotate bits right (shift values in vector) and assign
- RORSA  - Rotate bits right (shift values in scalar) and assign
- MRORSA - Masked rotate bits right (shift values in scalar) and assign

3) Operations available for Signed integer and Unsigned integer data types:\

(Signed/Unsigned cast)
- UTOI - Cast unsigned vector to signed vector
- ITOU - Cast signed vector to unsigned vector

4) Operations available for Signed integer and floating point SIMD types:

(Sign modification)
- NEG   - Negate signed values
- MNEG  - Masked negate signed values
- NEGA  - Negate signed values and assign
- MNEGA - Masked negate signed values and assign

(Mathematical functions)
- ABS   - Absolute value
- MABS  - Masked absolute value
- ABSA  - Absolute value and assign
- MABSA - Masked absolute value and assign

5) Operations available for floating point SIMD types:

(Comparison operations)
- CMPEQRV - Compare 'Equal within range' with margins from vector
- CMPEQRS - Compare 'Equal within range' with scalar margin

(Mathematical functions)
- SQR       - Square of vector values
- MSQR      - Masked square of vector values
- SQRA      - Square of vector values and assign
- MSQRA     - Masked square of vector values and assign
- SQRT      - Square root of vector values
- MSQRT     - Masked square root of vector values 
- SQRTA     - Square root of vector values and assign
- MSQRTA    - Masked square root of vector values and assign
- POWV      - Power (exponents in vector)
- MPOWV     - Masked power (exponents in vector)
- POWS      - Power (exponent in scalar)
- MPOWS     - Masked power (exponent in scalar) 
- ROUND     - Round to nearest integer
- MROUND    - Masked round to nearest integer
- TRUNC     - Truncate to integer (returns Signed integer vector)
- MTRUNC    - Masked truncate to integer (returns Signed integer vector)
- FLOOR     - Floor
- MFLOOR    - Masked floor
- CEIL      - Ceil
- MCEIL     - Masked ceil
- ISFIN     - Is finite
- ISINF     - Is infinite (INF)
- ISAN      - Is a number
- ISNAN     - Is 'Not a Number (NaN)'
- ISSUB     - Is subnormal
- ISZERO    - Is zero
- ISZEROSUB - Is zero or subnormal
- SIN       - Sine
- MSIN      - Masked sine
- COS       - Cosine
- MCOS      - Masked cosine
- TAN       - Tangent
- MTAN      - Masked tangent
- CTAN      - Cotangent
- MCTAN     - Masked cotangent

6) Operations available on Mask types
(construction)
- SET-CONSTR  - one element constructor (promote scalar to vector)
- FULL-CONSTR - MASK-LEN element constructor

(Memory access)
- LOAD    - Load values from memory (either aligned or unaligned) to mask
- LOADA   - Load values from aligned memory to mask 
- STORE   - Store values from mask to memory (either aligned or unaligned)
- STOREA  - Store values from mask to aligned memory
- EXTRACT - Extract single element from mask
- INSERT  - Insert single element into mask

(Initialization)
- ASSIGN  - Assign with mask

(Binary operations)
- AND  - Binary AND
- ANDA - Binary AND and assign
- OR   - Binary OR
- ANDA - Binary OR and assign
- XOR  - Binary XOR
- XORA - Binary XOR and assign
- NOT  - Binary NOT
- NOTA - Binary NOT and assign

7) Operations available on Swizzle Mask types

(Still working on this...)

