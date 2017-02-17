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
#ifndef UME_UNIT_TEST_DATA_SETS_64_H_
#define UME_UNIT_TEST_DATA_SETS_64_H_

#include <cstdint>

struct DataSet_1_64u {
    struct inputs {
        static const uint64_t inputA[16];
        static const uint64_t inputB[16];
        static const uint64_t inputC[16];
        static const uint64_t inputShiftA[16];
        static const uint64_t scalarA;
        static const uint64_t inputShiftScalarA;
        static const bool  maskA[16];
    };

    struct outputs {
        static const uint64_t ADDV[16];
        static const uint64_t MADDV[16];
        static const uint64_t ADDS[16];
        static const uint64_t MADDS[16];
        static const uint64_t POSTPREFINC[16];
        static const uint64_t MPOSTPREFINC[16];
        static const uint64_t SUBV[16];
        static const uint64_t MSUBV[16];
        static const uint64_t SUBS[16];
        static const uint64_t MSUBS[16];
        static const uint64_t SUBFROMV[16];
        static const uint64_t MSUBFROMV[16];
        static const uint64_t SUBFROMS[16];
        static const uint64_t MSUBFROMS[16];
        static const uint64_t POSTPREFDEC[16];
        static const uint64_t MPOSTPREFDEC[16];
        static const uint64_t MULV[16];
        static const uint64_t MMULV[16];
        static const uint64_t MULS[16];
        static const uint64_t MMULS[16];
        static const uint64_t DIVV[16];
        static const uint64_t MDIVV[16];
        static const uint64_t DIVS[16];
        static const uint64_t MDIVS[16];
        static const uint64_t RCP[16];
        static const uint64_t MRCP[16];
        static const uint64_t RCPS[16];
        static const uint64_t MRCPS[16];
        static const bool  CMPEQV[16];
        static const bool  CMPEQS[16];
        static const bool  CMPNEV[16];
        static const bool  CMPNES[16];
        static const bool  CMPGTV[16];
        static const bool  CMPGTS[16];
        static const bool  CMPLTV[16];
        static const bool  CMPLTS[16];
        static const bool  CMPGEV[16];
        static const bool  CMPGES[16];
        static const bool  CMPLEV[16];
        static const bool  CMPLES[16];
        static const bool  CMPEV;
        static const bool  CMPES;

        static const uint64_t HADD[16];
        static const uint64_t MHADD[16];
        static const uint64_t HMUL[16];
        static const uint64_t MHMUL[16];

        static const uint64_t BANDV[16];
        static const uint64_t MBANDV[16];
        static const uint64_t BANDS[16];
        static const uint64_t MBANDS[16];
        static const uint64_t BORV[16];
        static const uint64_t MBORV[16];
        static const uint64_t BORS[16];
        static const uint64_t MBORS[16];
        static const uint64_t BXORV[16];
        static const uint64_t MBXORV[16];
        static const uint64_t BXORS[16];
        static const uint64_t MBXORS[16];
        static const uint64_t BNOT[16];
        static const uint64_t MBNOT[16];

        static const uint64_t HBAND[16];
        static const uint64_t MHBAND[16];
        static const uint64_t HBANDS[16];
        static const uint64_t MHBANDS[16];
        static const uint64_t HBOR[16];
        static const uint64_t MHBOR[16];
        static const uint64_t HBORS[16];
        static const uint64_t MHBORS[16];
        static const uint64_t HBXOR[16];
        static const uint64_t MHBXOR[16];
        static const uint64_t HBXORS[16];
        static const uint64_t MHBXORS[16];

        static const uint64_t FMULADDV[16];
        static const uint64_t MFMULADDV[16];
        static const uint64_t FMULSUBV[16];
        static const uint64_t MFMULSUBV[16];
        static const uint64_t FADDMULV[16];
        static const uint64_t MFADDMULV[16];
        static const uint64_t FSUBMULV[16];
        static const uint64_t MFSUBMULV[16];
        static const uint64_t MAXV[16];
        static const uint64_t MMAXV[16];
        static const uint64_t MAXS[16];
        static const uint64_t MMAXS[16];
        static const uint64_t MINV[16];
        static const uint64_t MMINV[16];
        static const uint64_t MINS[16];
        static const uint64_t MMINS[16];
        static const uint64_t HMAX[16];
        static const uint64_t MHMAX[16];
        static const uint64_t HMIN[16];
        static const uint64_t MHMIN[16];
        static const uint64_t SQR[16];
        static const uint64_t MSQR[16];
        static const uint64_t SQRT[16];
        static const uint64_t MSQRT[16];

        static const uint64_t LSHV[16];
        static const uint64_t MLSHV[16];
        static const uint64_t LSHS[16];
        static const uint64_t MLSHS[16];
        static const uint64_t RSHV[16];
        static const uint64_t MRSHV[16];
        static const uint64_t RSHS[16];
        static const uint64_t MRSHS[16];
        static const uint64_t ROLV[16];
        static const uint64_t MROLV[16];
        static const uint64_t ROLS[16];
        static const uint64_t MROLS[16];
        static const uint64_t RORV[16];
        static const uint64_t MRORV[16];
        static const uint64_t RORS[16];
        static const uint64_t MRORS[16];

        static const int64_t UTOI[16];
        static const double   UTOF[16];
    };
};


struct DataSet_1_64i {
    struct inputs {
        static const int64_t inputA[16];
        static const int64_t inputB[16];
        static const int64_t inputC[16];
        static const uint64_t inputShiftA[16];
        static const int64_t scalarA;
        static const uint64_t inputShiftScalarA;
        static const bool  maskA[16];
    };

    struct outputs {
        static const int64_t ADDV[16];
        static const int64_t MADDV[16];
        static const int64_t ADDS[16];
        static const int64_t MADDS[16];
        static const int64_t POSTPREFINC[16];
        static const int64_t MPOSTPREFINC[16];
        static const int64_t SUBV[16];
        static const int64_t MSUBV[16];
        static const int64_t SUBS[16];
        static const int64_t MSUBS[16];
        static const int64_t SUBFROMV[16];
        static const int64_t MSUBFROMV[16];
        static const int64_t SUBFROMS[16];
        static const int64_t MSUBFROMS[16];
        static const int64_t POSTPREFDEC[16];
        static const int64_t MPOSTPREFDEC[16];
        static const int64_t MULV[16];
        static const int64_t MMULV[16];
        static const int64_t MULS[16];
        static const int64_t MMULS[16];
        static const int64_t DIVV[16];
        static const int64_t MDIVV[16];
        static const int64_t DIVS[16];
        static const int64_t MDIVS[16];
        static const int64_t RCP[16];
        static const int64_t MRCP[16];
        static const int64_t RCPS[16];
        static const int64_t MRCPS[16];
        static const bool  CMPEQV[16];
        static const bool  CMPEQS[16];
        static const bool  CMPNEV[16];
        static const bool  CMPNES[16];
        static const bool  CMPGTV[16];
        static const bool  CMPGTS[16];
        static const bool  CMPLTV[16];
        static const bool  CMPLTS[16];
        static const bool  CMPGEV[16];
        static const bool  CMPGES[16];
        static const bool  CMPLEV[16];
        static const bool  CMPLES[16];
        static const bool  CMPEV;
        static const bool  CMPES;

        static const int64_t HADD[16];
        static const int64_t MHADD[16];
        static const int64_t HMUL[16];
        static const int64_t MHMUL[16];

        static const int64_t BANDV[16];
        static const int64_t MBANDV[16];
        static const int64_t BANDS[16];
        static const int64_t MBANDS[16];
        static const int64_t BORV[16];
        static const int64_t MBORV[16];
        static const int64_t BORS[16];
        static const int64_t MBORS[16];
        static const int64_t BXORV[16];
        static const int64_t MBXORV[16];
        static const int64_t BXORS[16];
        static const int64_t MBXORS[16];
        static const int64_t BNOT[16];
        static const int64_t MBNOT[16];

        static const int64_t HBAND[16];
        static const int64_t MHBAND[16];
        static const int64_t HBANDS[16];
        static const int64_t MHBANDS[16];
        static const int64_t HBOR[16];
        static const int64_t MHBOR[16];
        static const int64_t HBORS[16];
        static const int64_t MHBORS[16];
        static const int64_t HBXOR[16];
        static const int64_t MHBXOR[16];
        static const int64_t HBXORS[16];
        static const int64_t MHBXORS[16];

        static const int64_t FMULADDV[16];
        static const int64_t MFMULADDV[16];
        static const int64_t FMULSUBV[16];
        static const int64_t MFMULSUBV[16];
        static const int64_t FADDMULV[16];
        static const int64_t MFADDMULV[16];
        static const int64_t FSUBMULV[16];
        static const int64_t MFSUBMULV[16];
        static const int64_t MAXV[16];
        static const int64_t MMAXV[16];
        static const int64_t MAXS[16];
        static const int64_t MMAXS[16];
        static const int64_t MINV[16];
        static const int64_t MMINV[16];
        static const int64_t MINS[16];
        static const int64_t MMINS[16];
        static const int64_t HMAX[16];
        static const int64_t MHMAX[16];
        static const int64_t HMIN[16];
        static const int64_t MHMIN[16];
        static const int64_t SQR[16];
        static const int64_t MSQR[16];
        static const int64_t SQRT[16];
        static const int64_t MSQRT[16];

        static const int64_t LSHV[16];
        static const int64_t MLSHV[16];
        static const int64_t LSHS[16];
        static const int64_t MLSHS[16];
        static const int64_t RSHV[16];
        static const int64_t MRSHV[16];
        static const int64_t RSHS[16];
        static const int64_t MRSHS[16];
        static const int64_t ROLV[16];
        static const int64_t MROLV[16];
        static const int64_t ROLS[16];
        static const int64_t MROLS[16];
        static const int64_t RORV[16];
        static const int64_t MRORV[16];
        static const int64_t RORS[16];
        static const int64_t MRORS[16];

        static const int64_t NEG[16];
        static const int64_t MNEG[16];
        static const int64_t ABS[16];
        static const int64_t MABS[16];

        static const uint64_t ITOU[16];
        static const double    ITOF[16];
    };
};

struct DataSet_1_64f {
    struct inputs {
        static const double inputA[16];
        static const double inputB[16];
        static const double inputC[16];
        static const uint64_t inputUintA[16];
        static const int64_t inputIntA[16];
        static const double scalarA;
        static const bool  maskA[16];
    };

    struct outputs {
        static const double ADDV[16];
        static const double MADDV[16];
        static const double ADDS[16];
        static const double MADDS[16];
        static const double POSTPREFINC[16];
        static const double MPOSTPREFINC[16];
        static const double SUBV[16];
        static const double MSUBV[16];
        static const double SUBS[16];
        static const double MSUBS[16];
        static const double SUBFROMV[16];
        static const double MSUBFROMV[16];
        static const double SUBFROMS[16];
        static const double MSUBFROMS[16];
        static const double POSTPREFDEC[16];
        static const double MPOSTPREFDEC[16];
        static const double MULV[16];
        static const double MMULV[16];
        static const double MULS[16];
        static const double MMULS[16];
        static const double DIVV[16];
        static const double MDIVV[16];
        static const double DIVS[16];
        static const double MDIVS[16];
        static const double RCP[16];
        static const double MRCP[16];
        static const double RCPS[16];
        static const double MRCPS[16];
        static const bool  CMPEQV[16];
        static const bool  CMPEQS[16];
        static const bool  CMPNEV[16];
        static const bool  CMPNES[16];
        static const bool  CMPGTV[16];
        static const bool  CMPGTS[16];
        static const bool  CMPLTV[16];
        static const bool  CMPLTS[16];
        static const bool  CMPGEV[16];
        static const bool  CMPGES[16];
        static const bool  CMPLEV[16];
        static const bool  CMPLES[16];
        static const bool  CMPEV;
        static const bool  CMPES;

        static const double HADD[16];
        static const double MHADD[16];
        static const double HMUL[16];
        static const double MHMUL[16];

        static const double FMULADDV[16];
        static const double MFMULADDV[16];
        static const double FMULSUBV[16];
        static const double MFMULSUBV[16];
        static const double FADDMULV[16];
        static const double MFADDMULV[16];
        static const double FSUBMULV[16];
        static const double MFSUBMULV[16];
        static const double MAXV[16];
        static const double MMAXV[16];
        static const double MAXS[16];
        static const double MMAXS[16];
        static const double MINV[16];
        static const double MMINV[16];
        static const double MINS[16];
        static const double MMINS[16];
        static const double HMAX[16];
        static const double MHMAX[16];
        static const double HMIN[16];
        static const double MHMIN[16];

        static const double NEG[16];
        static const double MNEG[16];
        static const double ABS[16];
        static const double MABS[16];

        static const double SQR[16];
        static const double MSQR[16];
        static const double SQRT[16];
        static const double MSQRT[16];
        static const double ROUND[16];
        static const double MROUND[16];
        static const int64_t TRUNC[16];
        static const int64_t MTRUNC[16];
        static const double FLOOR[16];
        static const double MFLOOR[16];
        static const double CEIL[16];
        static const double MCEIL[16];
        static const bool ISFIN[16];
        static const bool ISINF[16];
        static const bool ISAN[16];
        static const bool ISNAN[16];
        static const bool ISNORM[16];
        static const bool ISSUB[16];
        static const bool ISZERO[16];
        static const bool ISZEROSUB[16];

        static const double SIN[16];
        static const double MSIN[16];
        static const double COS[16];
        static const double MCOS[16];
        static const double TAN[16];
        static const double MTAN[16];
        static const double CTAN[16];
        static const double MCTAN[16];

        static const uint64_t FTOU[16];
        static const int64_t  FTOI[16];
    };
};

#endif
