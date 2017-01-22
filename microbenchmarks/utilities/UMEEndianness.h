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

#ifndef ENDIANNESS_H_
#define ENDIANNESS_H_

// TODO: for now assuming big-endian for all cases
// Extract LSB out of a DWORD value
#define EXTRACT_BYTE_0_FROM_DWORD(x)  (x & 0x000000FF)
// Extract second LSB out of a DWORD value
#define EXTRACT_BYTE_1_FROM_DWORD(x) ((x & 0x0000FF00) >> 8)
// Extract third LSB out of a DWORD value
#define EXTRACT_BYTE_2_FROM_DWORD(x) ((x & 0x00FF0000) >> 16)
// Extract MSB out of a DWORD value
#define EXTRACT_BYTE_3_FROM_DWORD(x) ((x & 0xFF000000) >> 24)

#define READ_WORD(x)  ((x)[0] | ((x)[1] << 8))
#define READ_DWORD(x) ((x)[0] | ((x)[1] << 8) | ((x)[2] << 16) | ((x)[3] << 24))
#define READ_QWORD(x) ((x)[0] | ((x)[1] << 8) | ((x)[2] << 16) | ((x)[3] << 24) | ((x)[4] << 32) | ((x)[5] << 40) | ((x)[6] << 48) | ((x)[7] << 56))

#define WRITE_WORD(dst, x) (((uint8_t*)(dst))[0] = ((x) & 0xFF)); (((uint8_t*)(dst))[1] = (((x) & 0xFF00) >> 8));
#define WRITE_DWORD(dst, x) WRITE_WORD((dst), ((x) & 0xFFFF)); WRITE_WORD((dst + 2), ((x) & 0xFFFF0000) >> 16); 
#define WRITE_QWORD(dst, x) WRITE_DWORD((dst), ((x) & 0xFFFFFFFF)); WRITE_DWORD((dst + 4), ((x) & 0xFFFFFFFF00000000) >> 32);

#endif
