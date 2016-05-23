// The MIT License (MIT)
//
// Copyright (c) 2016 CERN
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

#ifndef UME_INLINE_H_
#define UME_INLINE_H_

#if defined(_MSC_VER)
#define UME_FORCE_INLINE __forceinline
#define UME_NEVER_INLINE __declspec(noinline)
#elif defined(__INTEL_COMPILER)
// Intel compiler also implies __GNUC__ flag. For that reason we have to check it first.
#define UME_FORCE_INLINE inline __attribute__ ((always_inline))
#define UME_NEVER_INLINE __attribute__ ((noinline))
#elif defined(__GNUC__)
#define UME_FORCE_INLINE inline __attribute__ ((always_inline))
#define UME_NEVER_INLINE __attribute__ ((noinline))
#else
// Default fallback: if the compiler is unrecognized, simply try to ask it to inline.
#define UME_FORCE_INLINE inline
// Default fallback: do nothing...
#define UME_NEVER_INLINE
#endif


#endif
