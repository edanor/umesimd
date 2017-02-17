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
// This piece of code was developed as part of ICE-DIP project at CERN.
//  "ICE-DIP is a European Industrial Doctorate project funded by the European Community's
//  7th Framework programme Marie Curie Actions under grant PITN-GA-2012-316596".
//
#ifndef UME_UNIT_TEST_DATA_SETS_16_H_
#define UME_UNIT_TEST_DATA_SETS_16_H_

#include <cstdint>

struct DataSet_1_16u {
    struct inputs {
        static const uint16_t inputA[64];
        static const uint16_t inputB[64];
        static const uint16_t inputC[64];
        static const uint16_t inputShiftA[64];
        static const uint16_t scalarA;
        static const uint16_t inputShiftScalarA;
        static const bool  maskA[64];
    };

    struct outputs {
        static const uint16_t ADDV[64];
        static const uint16_t MADDV[64];
        static const uint16_t ADDS[64];
        static const uint16_t MADDS[64];
        static const uint16_t POSTPREFINC[64];
        static const uint16_t MPOSTPREFINC[64];
        static const uint16_t SUBV[64];
        static const uint16_t MSUBV[64];
        static const uint16_t SUBS[64];
        static const uint16_t MSUBS[64];
        static const uint16_t SUBFROMV[64];
        static const uint16_t MSUBFROMV[64];
        static const uint16_t SUBFROMS[64];
        static const uint16_t MSUBFROMS[64];
        static const uint16_t POSTPREFDEC[64];
        static const uint16_t MPOSTPREFDEC[64];
        static const uint16_t MULV[64];
        static const uint16_t MMULV[64];
        static const uint16_t MULS[64];
        static const uint16_t MMULS[64];
        static const uint16_t DIVV[64];
        static const uint16_t MDIVV[64];
        static const uint16_t DIVS[64];
        static const uint16_t MDIVS[64];
        static const uint16_t RCP[64];
        static const uint16_t MRCP[64];
        static const uint16_t RCPS[64];
        static const uint16_t MRCPS[64];
        static const bool  CMPEQV[64];
        static const bool  CMPEQS[64];
        static const bool  CMPNEV[64];
        static const bool  CMPNES[64];
        static const bool  CMPGTV[64];
        static const bool  CMPGTS[64];
        static const bool  CMPLTV[64];
        static const bool  CMPLTS[64];
        static const bool  CMPGEV[64];
        static const bool  CMPGES[64];
        static const bool  CMPLEV[64];
        static const bool  CMPLES[64];
        static const bool  CMPEV;
        static const bool  CMPES;

        static const uint16_t HADD[64];
        static const uint16_t MHADD[64];
        static const uint16_t HMUL[64];
        static const uint16_t MHMUL[64];

        static const uint16_t BANDV[64];
        static const uint16_t MBANDV[64];
        static const uint16_t BANDS[64];
        static const uint16_t MBANDS[64];
        static const uint16_t BORV[64];
        static const uint16_t MBORV[64];
        static const uint16_t BORS[64];
        static const uint16_t MBORS[64];
        static const uint16_t BXORV[64];
        static const uint16_t MBXORV[64];
        static const uint16_t BXORS[64];
        static const uint16_t MBXORS[64];
        static const uint16_t BNOT[64];
        static const uint16_t MBNOT[64];

        static const uint16_t HBAND[64];
        static const uint16_t MHBAND[64];
        static const uint16_t HBANDS[64];
        static const uint16_t MHBANDS[64];
        static const uint16_t HBOR[64];
        static const uint16_t MHBOR[64];
        static const uint16_t HBORS[64];
        static const uint16_t MHBORS[64];
        static const uint16_t HBXOR[64];
        static const uint16_t MHBXOR[64];
        static const uint16_t HBXORS[64];
        static const uint16_t MHBXORS[64];

        static const uint16_t FMULADDV[64];
        static const uint16_t MFMULADDV[64];
        static const uint16_t FMULSUBV[64];
        static const uint16_t MFMULSUBV[64];
        static const uint16_t FADDMULV[64];
        static const uint16_t MFADDMULV[64];
        static const uint16_t FSUBMULV[64];
        static const uint16_t MFSUBMULV[64];
        static const uint16_t MAXV[64];
        static const uint16_t MMAXV[64];
        static const uint16_t MAXS[64];
        static const uint16_t MMAXS[64];
        static const uint16_t MINV[64];
        static const uint16_t MMINV[64];
        static const uint16_t MINS[64];
        static const uint16_t MMINS[64];
        static const uint16_t HMAX[64];
        static const uint16_t MHMAX[64];
        static const uint16_t HMIN[64];
        static const uint16_t MHMIN[64];
        static const uint16_t SQR[64];
        static const uint16_t MSQR[64];
        static const uint16_t SQRT[64];
        static const uint16_t MSQRT[64];

        static const uint16_t LSHV[64];
        static const uint16_t MLSHV[64];
        static const uint16_t LSHS[64];
        static const uint16_t MLSHS[64];
        static const uint16_t RSHV[64];
        static const uint16_t MRSHV[64];
        static const uint16_t RSHS[64];
        static const uint16_t MRSHS[64];
        static const uint16_t ROLV[64];
        static const uint16_t MROLV[64];
        static const uint16_t ROLS[64];
        static const uint16_t MROLS[64];
        static const uint16_t RORV[64];
        static const uint16_t MRORV[64];
        static const uint16_t RORS[64];
        static const uint16_t MRORS[64];

        static const int16_t   UTOI[64];
        //static const float8_t UTOF[64];
    };
};

struct DataSet_1_16i {
    struct inputs {
        static const int16_t inputA[64];
        static const int16_t inputB[64];
        static const int16_t inputC[64];
        static const uint16_t inputShiftA[64];
        static const int16_t scalarA;
        static const uint16_t inputShiftScalarA;
        static const bool  maskA[64];
    };

    struct outputs {
        static const int16_t ADDV[64];
        static const int16_t MADDV[64];
        static const int16_t ADDS[64];
        static const int16_t MADDS[64];
        static const int16_t POSTPREFINC[64];
        static const int16_t MPOSTPREFINC[64];
        static const int16_t SUBV[64];
        static const int16_t MSUBV[64];
        static const int16_t SUBS[64];
        static const int16_t MSUBS[64];
        static const int16_t SUBFROMV[64];
        static const int16_t MSUBFROMV[64];
        static const int16_t SUBFROMS[64];
        static const int16_t MSUBFROMS[64];
        static const int16_t POSTPREFDEC[64];
        static const int16_t MPOSTPREFDEC[64];
        static const int16_t MULV[64];
        static const int16_t MMULV[64];
        static const int16_t MULS[64];
        static const int16_t MMULS[64];
        static const int16_t DIVV[64];
        static const int16_t MDIVV[64];
        static const int16_t DIVS[64];
        static const int16_t MDIVS[64];
        static const int16_t RCP[64];
        static const int16_t MRCP[64];
        static const int16_t RCPS[64];
        static const int16_t MRCPS[64];
        static const bool  CMPEQV[64];
        static const bool  CMPEQS[64];
        static const bool  CMPNEV[64];
        static const bool  CMPNES[64];
        static const bool  CMPGTV[64];
        static const bool  CMPGTS[64];
        static const bool  CMPLTV[64];
        static const bool  CMPLTS[64];
        static const bool  CMPGEV[64];
        static const bool  CMPGES[64];
        static const bool  CMPLEV[64];
        static const bool  CMPLES[64];
        static const bool  CMPEV;
        static const bool  CMPES;

        static const int16_t HADD[64];
        static const int16_t MHADD[64];
        static const int16_t HMUL[64];
        static const int16_t MHMUL[64];

        static const int16_t BANDV[64];
        static const int16_t MBANDV[64];
        static const int16_t BANDS[64];
        static const int16_t MBANDS[64];
        static const int16_t BORV[64];
        static const int16_t MBORV[64];
        static const int16_t BORS[64];
        static const int16_t MBORS[64];
        static const int16_t BXORV[64];
        static const int16_t MBXORV[64];
        static const int16_t BXORS[64];
        static const int16_t MBXORS[64];
        static const int16_t BNOT[64];
        static const int16_t MBNOT[64];

        static const int16_t HBAND[64];
        static const int16_t MHBAND[64];
        static const int16_t HBANDS[64];
        static const int16_t MHBANDS[64];
        static const int16_t HBOR[64];
        static const int16_t MHBOR[64];
        static const int16_t HBORS[64];
        static const int16_t MHBORS[64];
        static const int16_t HBXOR[64];
        static const int16_t MHBXOR[64];
        static const int16_t HBXORS[64];
        static const int16_t MHBXORS[64];

        static const int16_t FMULADDV[64];
        static const int16_t MFMULADDV[64];
        static const int16_t FMULSUBV[64];
        static const int16_t MFMULSUBV[64];
        static const int16_t FADDMULV[64];
        static const int16_t MFADDMULV[64];
        static const int16_t FSUBMULV[64];
        static const int16_t MFSUBMULV[64];
        static const int16_t MAXV[64];
        static const int16_t MMAXV[64];
        static const int16_t MAXS[64];
        static const int16_t MMAXS[64];
        static const int16_t MINV[64];
        static const int16_t MMINV[64];
        static const int16_t MINS[64];
        static const int16_t MMINS[64];
        static const int16_t HMAX[64];
        static const int16_t MHMAX[64];
        static const int16_t HMIN[64];
        static const int16_t MHMIN[64];
        static const int16_t SQR[64];
        static const int16_t MSQR[64];
        static const int16_t SQRT[64];
        static const int16_t MSQRT[64];

        // define as uint16_t, but load as int16_t
        static const int16_t LSHV[64];
        static const int16_t MLSHV[64];
        static const int16_t LSHS[64];
        static const int16_t MLSHS[64];
        static const int16_t RSHV[64];
        static const int16_t MRSHV[64];
        static const int16_t RSHS[64];
        static const int16_t MRSHS[64];
        static const int16_t ROLV[64];
        static const int16_t MROLV[64];
        static const int16_t ROLS[64];
        static const int16_t MROLS[64];
        static const int16_t RORV[64];
        static const int16_t MRORV[64];
        static const int16_t RORS[64];
        static const int16_t MRORS[64];

        static const int16_t NEG[64];
        static const int16_t MNEG[64];
        static const int16_t ABS[64];
        static const int16_t MABS[64];

        static const uint16_t ITOU[64];
        //static const float8_t    ITOF[64];
    };

};

#endif
