// The MIT License (MIT)
//
// Copyright (c) 2015 CERN
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
#ifndef UME_UNIT_TEST_DATA_SETS_8_H_
#define UME_UNIT_TEST_DATA_SETS_8_H_

#include "../UMEBasicTypes.h"


struct DataSet_1_8u {
    struct inputs {
        static const uint8_t inputA[128];
        static const uint8_t inputB[128];
        static const uint8_t inputC[128];
        static const uint8_t inputShiftA[128];
        static const uint8_t scalarA;
        static const uint8_t inputShiftScalarA;
        static const bool  maskA[128];
    };

    struct outputs {
        static const uint8_t ADDV[128];
        static const uint8_t MADDV[128];
        static const uint8_t ADDS[128];
        static const uint8_t MADDS[128];
        static const uint8_t POSTPREFINC[128];
        static const uint8_t MPOSTPREFINC[128];
        static const uint8_t SUBV[128];
        static const uint8_t MSUBV[128];
        static const uint8_t SUBS[128];
        static const uint8_t MSUBS[128];
        static const uint8_t SUBFROMV[128];
        static const uint8_t MSUBFROMV[128];
        static const uint8_t SUBFROMS[128];
        static const uint8_t MSUBFROMS[128];
        static const uint8_t POSTPREFDEC[128];
        static const uint8_t MPOSTPREFDEC[128];
        static const uint8_t MULV[128];
        static const uint8_t MMULV[128];
        static const uint8_t MULS[128];
        static const uint8_t MMULS[128];
        static const uint8_t DIVV[128];
        static const uint8_t MDIVV[128];
        static const uint8_t DIVS[128];
        static const uint8_t MDIVS[128];
        static const uint8_t RCP[128];
        static const uint8_t MRCP[128];
        static const uint8_t RCPS[128];
        static const uint8_t MRCPS[128];
        static const bool  CMPEQV[128];
        static const bool  CMPEQS[128];
        static const bool  CMPNEV[128];
        static const bool  CMPNES[128];
        static const bool  CMPGTV[128];
        static const bool  CMPGTS[128];
        static const bool  CMPLTV[128];
        static const bool  CMPLTS[128];
        static const bool  CMPGEV[128];
        static const bool  CMPGES[128];
        static const bool  CMPLEV[128];
        static const bool  CMPLES[128];
        static const bool  CMPEV;
        static const bool  CMPES;

        static const uint8_t HADD[128];
        static const uint8_t MHADD[128];
        static const uint8_t HMUL[128];
        static const uint8_t MHMUL[128];

        static const uint8_t BANDV[128];
        static const uint8_t MBANDV[128];
        static const uint8_t BANDS[128];
        static const uint8_t MBANDS[128];
        static const uint8_t BORV[128];
        static const uint8_t MBORV[128];
        static const uint8_t BORS[128];
        static const uint8_t MBORS[128];
        static const uint8_t BXORV[128];
        static const uint8_t MBXORV[128];
        static const uint8_t BXORS[128];
        static const uint8_t MBXORS[128];
        static const uint8_t BNOT[128];
        static const uint8_t MBNOT[128];

        static const uint8_t HBAND[128];
        static const uint8_t MHBAND[128];
        static const uint8_t HBANDS[128];
        static const uint8_t MHBANDS[128];
        static const uint8_t HBOR[128];
        static const uint8_t MHBOR[128];
        static const uint8_t HBORS[128];
        static const uint8_t MHBORS[128];
        static const uint8_t HBXOR[128];
        static const uint8_t MHBXOR[128];
        static const uint8_t HBXORS[128];
        static const uint8_t MHBXORS[128];

        static const uint8_t FMULADDV[128];
        static const uint8_t MFMULADDV[128];
        static const uint8_t FMULSUBV[128];
        static const uint8_t MFMULSUBV[128];
        static const uint8_t FADDMULV[128];
        static const uint8_t MFADDMULV[128];
        static const uint8_t FSUBMULV[128];
        static const uint8_t MFSUBMULV[128];
        static const uint8_t MAXV[128];
        static const uint8_t MMAXV[128];
        static const uint8_t MAXS[128];
        static const uint8_t MMAXS[128];
        static const uint8_t MINV[128];
        static const uint8_t MMINV[128];
        static const uint8_t MINS[128];
        static const uint8_t MMINS[128];
        static const uint8_t HMAX[128];
        static const uint8_t MHMAX[128];
        static const uint8_t HMIN[128];
        static const uint8_t MHMIN[128];
        static const uint8_t SQR[128];
        static const uint8_t MSQR[128];
        static const uint8_t SQRT[128];
        static const uint8_t MSQRT[128];

        static const uint8_t LSHV[128];
        static const uint8_t MLSHV[128];
        static const uint8_t LSHS[128];
        static const uint8_t MLSHS[128];
        static const uint8_t RSHV[128];
        static const uint8_t MRSHV[128];
        static const uint8_t RSHS[128];
        static const uint8_t MRSHS[128];
        static const uint8_t ROLV[128];
        static const uint8_t MROLV[128];
        static const uint8_t ROLS[128];
        static const uint8_t MROLS[128];
        static const uint8_t RORV[128];
        static const uint8_t MRORV[128];
        static const uint8_t RORS[128];
        static const uint8_t MRORS[128];

        static const int8_t   UTOI[128];
        //static const float8_t UTOF[128];
    };
};

struct DataSet_1_8i {
    struct inputs {
        static const int8_t inputA[128];
        static const int8_t inputB[128];
        static const int8_t inputC[128];
        static const uint8_t inputShiftA[128];
        static const int8_t scalarA;
        static const uint8_t inputShiftScalarA;
        static const bool  maskA[128];
    };

    struct outputs {
        static const int8_t ADDV[128];
        static const int8_t MADDV[128];
        static const int8_t ADDS[128];
        static const int8_t MADDS[128];
        static const int8_t POSTPREFINC[128];
        static const int8_t MPOSTPREFINC[128];
        static const int8_t SUBV[128];
        static const int8_t MSUBV[128];
        static const int8_t SUBS[128];
        static const int8_t MSUBS[128];
        static const int8_t SUBFROMV[128];
        static const int8_t MSUBFROMV[128];
        static const int8_t SUBFROMS[128];
        static const int8_t MSUBFROMS[128];
        static const int8_t POSTPREFDEC[128];
        static const int8_t MPOSTPREFDEC[128];
        static const int8_t MULV[128];
        static const int8_t MMULV[128];
        static const int8_t MULS[128];
        static const int8_t MMULS[128];
        static const int8_t DIVV[128];
        static const int8_t MDIVV[128];
        static const int8_t DIVS[128];
        static const int8_t MDIVS[128];
        static const int8_t RCP[128];
        static const int8_t MRCP[128];
        static const int8_t RCPS[128];
        static const int8_t MRCPS[128];
        static const bool  CMPEQV[128];
        static const bool  CMPEQS[128];
        static const bool  CMPNEV[128];
        static const bool  CMPNES[128];
        static const bool  CMPGTV[128];
        static const bool  CMPGTS[128];
        static const bool  CMPLTV[128];
        static const bool  CMPLTS[128];
        static const bool  CMPGEV[128];
        static const bool  CMPGES[128];
        static const bool  CMPLEV[128];
        static const bool  CMPLES[128];
        static const bool  CMPEV;
        static const bool  CMPES;

        static const int8_t HADD[128];
        static const int8_t MHADD[128];
        static const int8_t HMUL[128];
        static const int8_t MHMUL[128];

        static const int8_t BANDV[128];
        static const int8_t MBANDV[128];
        static const int8_t BANDS[128];
        static const int8_t MBANDS[128];
        static const int8_t BORV[128];
        static const int8_t MBORV[128];
        static const int8_t BORS[128];
        static const int8_t MBORS[128];
        static const int8_t BXORV[128];
        static const int8_t MBXORV[128];
        static const int8_t BXORS[128];
        static const int8_t MBXORS[128];
        static const int8_t BNOT[128];
        static const int8_t MBNOT[128];

        static const int8_t HBAND[128];
        static const int8_t MHBAND[128];
        static const int8_t HBANDS[128];
        static const int8_t MHBANDS[128];
        static const int8_t HBOR[128];
        static const int8_t MHBOR[128];
        static const int8_t HBORS[128];
        static const int8_t MHBORS[128];
        static const int8_t HBXOR[128];
        static const int8_t MHBXOR[128];
        static const int8_t HBXORS[128];
        static const int8_t MHBXORS[128];

        static const int8_t FMULADDV[128];
        static const int8_t MFMULADDV[128];
        static const int8_t FMULSUBV[128];
        static const int8_t MFMULSUBV[128];
        static const int8_t FADDMULV[128];
        static const int8_t MFADDMULV[128];
        static const int8_t FSUBMULV[128];
        static const int8_t MFSUBMULV[128];
        static const int8_t MAXV[128];
        static const int8_t MMAXV[128];
        static const int8_t MAXS[128];
        static const int8_t MMAXS[128];
        static const int8_t MINV[128];
        static const int8_t MMINV[128];
        static const int8_t MINS[128];
        static const int8_t MMINS[128];
        static const int8_t HMAX[128];
        static const int8_t MHMAX[128];
        static const int8_t HMIN[128];
        static const int8_t MHMIN[128];
        static const int8_t SQR[128];
        static const int8_t MSQR[128];
        static const int8_t SQRT[128];
        static const int8_t MSQRT[128];

        // define as uint8_t, but load as int8_t
        static const uint8_t LSHV[128];
        static const uint8_t MLSHV[128];
        static const uint8_t LSHS[128];
        static const uint8_t MLSHS[128];
        static const uint8_t RSHV[128];
        static const uint8_t MRSHV[128];
        static const uint8_t RSHS[128];
        static const uint8_t MRSHS[128];
        static const uint8_t ROLV[128];
        static const uint8_t MROLV[128];
        static const uint8_t ROLS[128];
        static const uint8_t MROLS[128];
        static const uint8_t RORV[128];
        static const uint8_t MRORV[128];
        static const uint8_t RORS[128];
        static const uint8_t MRORS[128];

        static const int8_t NEG[128];
        static const int8_t MNEG[128];
        static const int8_t ABS[128];
        static const int8_t MABS[128];

        static const uint8_t ITOU[128];
        //static const float8_t    ITOF[128];
    };

};

#endif
