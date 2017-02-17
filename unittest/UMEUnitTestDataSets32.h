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
#ifndef UME_UNIT_TEST_DATA_SETS_32_H_
#define UME_UNIT_TEST_DATA_SETS_32_H_

#include <cstdint>

struct DataSet_1_mask {
    struct inputs {
        static const bool maskA[128];
        static const bool maskB[128];

        static const bool scalarA;
        static const bool scalarB;
    };

    struct outputs {
        static const bool LANDV[128];
        static const bool LANDS_A[128]; // LANDS with scalarA
        static const bool LANDS_B[128]; // LANDS with scalarB
        static const bool LORV[128];
        static const bool LORS_A[128];  // LORS with scalarA
        static const bool LORS_B[128];  // LORS with scalarB
        static const bool LXORV[128];
        static const bool LXORS_A[128]; // LXORS with scalarA
        static const bool LXORS_B[128]; // LXORS with scalarB
        static const bool LNOT[128];
        static const bool HLAND[128];
        static const bool HLOR[128];
        static const bool HLXOR[128];
    };
};

struct DataSet_1_32u {
    struct inputs {
        static const uint32_t inputA[32];
        static const uint32_t inputB[32];
        static const uint32_t inputC[32];
        static const uint32_t inputShiftA[32];
        static const uint32_t scalarA;
        static const uint32_t inputShiftScalarA;
        static const bool  maskA[32];
    };

    struct outputs {
        static const uint32_t ADDV[32];
        static const uint32_t MADDV[32];
        static const uint32_t ADDS[32];
        static const uint32_t MADDS[32];
        static const uint32_t POSTPREFINC[32];
        static const uint32_t MPOSTPREFINC[32];
        static const uint32_t SUBV[32];
        static const uint32_t MSUBV[32];
        static const uint32_t SUBS[32];
        static const uint32_t MSUBS[32];
        static const uint32_t SUBFROMV[32];
        static const uint32_t MSUBFROMV[32];
        static const uint32_t SUBFROMS[32];
        static const uint32_t MSUBFROMS[32];
        static const uint32_t POSTPREFDEC[32];
        static const uint32_t MPOSTPREFDEC[32];
        static const uint32_t MULV[32];
        static const uint32_t MMULV[32];
        static const uint32_t MULS[32];
        static const uint32_t MMULS[32];
        static const uint32_t DIVV[32];
        static const uint32_t MDIVV[32];
        static const uint32_t DIVS[32];
        static const uint32_t MDIVS[32];
        static const uint32_t RCP[32];
        static const uint32_t MRCP[32];
        static const uint32_t RCPS[32];
        static const uint32_t MRCPS[32];
        static const bool  CMPEQV[32];
        static const bool  CMPEQS[32];
        static const bool  CMPNEV[32];
        static const bool  CMPNES[32];
        static const bool  CMPGTV[32];
        static const bool  CMPGTS[32];
        static const bool  CMPLTV[32];
        static const bool  CMPLTS[32];
        static const bool  CMPGEV[32];
        static const bool  CMPGES[32];
        static const bool  CMPLEV[32];
        static const bool  CMPLES[32];
        static const bool  CMPEV;
        static const bool  CMPES;

        static const uint32_t HADD[32];
        static const uint32_t MHADD[32];
        static const uint32_t HMUL[32];
        static const uint32_t MHMUL[32];

        static const uint32_t BANDV[32];
        static const uint32_t MBANDV[32];
        static const uint32_t BANDS[32];
        static const uint32_t MBANDS[32];
        static const uint32_t BORV[32];
        static const uint32_t MBORV[32];
        static const uint32_t BORS[32];
        static const uint32_t MBORS[32];
        static const uint32_t BXORV[32];
        static const uint32_t MBXORV[32];
        static const uint32_t BXORS[32];
        static const uint32_t MBXORS[32];
        static const uint32_t BNOT[32];
        static const uint32_t MBNOT[32];

        static const uint32_t HBAND[32];
        static const uint32_t MHBAND[32];
        static const uint32_t HBANDS[32];
        static const uint32_t MHBANDS[32];
        static const uint32_t HBOR[32];
        static const uint32_t MHBOR[32];
        static const uint32_t HBORS[32];
        static const uint32_t MHBORS[32];
        static const uint32_t HBXOR[32];
        static const uint32_t MHBXOR[32];
        static const uint32_t HBXORS[32];
        static const uint32_t MHBXORS[32];

        static const uint32_t FMULADDV[32];
        static const uint32_t MFMULADDV[32];
        static const uint32_t FMULSUBV[32];
        static const uint32_t MFMULSUBV[32];
        static const uint32_t FADDMULV[32];
        static const uint32_t MFADDMULV[32];
        static const uint32_t FSUBMULV[32];
        static const uint32_t MFSUBMULV[32];
        static const uint32_t MAXV[32];
        static const uint32_t MMAXV[32];
        static const uint32_t MAXS[32];
        static const uint32_t MMAXS[32];
        static const uint32_t MINV[32];
        static const uint32_t MMINV[32];
        static const uint32_t MINS[32];
        static const uint32_t MMINS[32];
        static const uint32_t HMAX[32];
        static const uint32_t MHMAX[32];
        static const uint32_t HMIN[32];
        static const uint32_t MHMIN[32];
        static const uint32_t SQR[32];
        static const uint32_t MSQR[32];
        static const uint32_t SQRT[32];
        static const uint32_t MSQRT[32];

        static const uint32_t LSHV[32];
        static const uint32_t MLSHV[32];
        static const uint32_t LSHS[32];
        static const uint32_t MLSHS[32];
        static const uint32_t RSHV[32];
        static const uint32_t MRSHV[32];
        static const uint32_t RSHS[32];
        static const uint32_t MRSHS[32];
        static const uint32_t ROLV[32];
        static const uint32_t MROLV[32];
        static const uint32_t ROLS[32];
        static const uint32_t MROLS[32];
        static const uint32_t RORV[32];
        static const uint32_t MRORV[32];
        static const uint32_t RORS[32];
        static const uint32_t MRORS[32];

        static const int32_t UTOI[32];
        static const float   UTOF[32];
    };
};


struct DataSet_1_32i {
    struct inputs {
        static const int32_t inputA[32];
        static const int32_t inputB[32];
        static const int32_t inputC[32];
        static const uint32_t inputShiftA[32];
        static const int32_t scalarA;
        static const uint32_t inputShiftScalarA;
        static const bool  maskA[32];
    };

    struct outputs {
        static const int32_t ADDV[32];
        static const int32_t MADDV[32];
        static const int32_t ADDS[32];
        static const int32_t MADDS[32];
        static const int32_t POSTPREFINC[32];
        static const int32_t MPOSTPREFINC[32];
        static const int32_t SUBV[32];
        static const int32_t MSUBV[32];
        static const int32_t SUBS[32];
        static const int32_t MSUBS[32];
        static const int32_t SUBFROMV[32];
        static const int32_t MSUBFROMV[32];
        static const int32_t SUBFROMS[32];
        static const int32_t MSUBFROMS[32];
        static const int32_t POSTPREFDEC[32];
        static const int32_t MPOSTPREFDEC[32];
        static const int32_t MULV[32];
        static const int32_t MMULV[32];
        static const int32_t MULS[32];
        static const int32_t MMULS[32];
        static const int32_t DIVV[32];
        static const int32_t MDIVV[32];
        static const int32_t DIVS[32];
        static const int32_t MDIVS[32];
        static const int32_t RCP[32];
        static const int32_t MRCP[32];
        static const int32_t RCPS[32];
        static const int32_t MRCPS[32];
        static const bool  CMPEQV[32];
        static const bool  CMPEQS[32];
        static const bool  CMPNEV[32];
        static const bool  CMPNES[32];
        static const bool  CMPGTV[32];
        static const bool  CMPGTS[32];
        static const bool  CMPLTV[32];
        static const bool  CMPLTS[32];
        static const bool  CMPGEV[32];
        static const bool  CMPGES[32];
        static const bool  CMPLEV[32];
        static const bool  CMPLES[32];
        static const bool  CMPEV;
        static const bool  CMPES;

        static const int32_t HADD[32];
        static const int32_t MHADD[32];
        static const int32_t HMUL[32];
        static const int32_t MHMUL[32];

        static const int32_t BANDV[32];
        static const int32_t MBANDV[32];
        static const int32_t BANDS[32];
        static const int32_t MBANDS[32];
        static const int32_t BORV[32];
        static const int32_t MBORV[32];
        static const int32_t BORS[32];
        static const int32_t MBORS[32];
        static const int32_t BXORV[32];
        static const int32_t MBXORV[32];
        static const int32_t BXORS[32];
        static const int32_t MBXORS[32];
        static const int32_t BNOT[32];
        static const int32_t MBNOT[32];

        static const int32_t HBAND[32];
        static const int32_t MHBAND[32];
        static const int32_t HBANDS[32];
        static const int32_t MHBANDS[32];
        static const int32_t HBOR[32];
        static const int32_t MHBOR[32];
        static const int32_t HBORS[32];
        static const int32_t MHBORS[32];
        static const int32_t HBXOR[32];
        static const int32_t MHBXOR[32];
        static const int32_t HBXORS[32];
        static const int32_t MHBXORS[32];

        static const int32_t FMULADDV[32];
        static const int32_t MFMULADDV[32];
        static const int32_t FMULSUBV[32];
        static const int32_t MFMULSUBV[32];
        static const int32_t FADDMULV[32];
        static const int32_t MFADDMULV[32];
        static const int32_t FSUBMULV[32];
        static const int32_t MFSUBMULV[32];
        static const int32_t MAXV[32];
        static const int32_t MMAXV[32];
        static const int32_t MAXS[32];
        static const int32_t MMAXS[32];
        static const int32_t MINV[32];
        static const int32_t MMINV[32];
        static const int32_t MINS[32];
        static const int32_t MMINS[32];
        static const int32_t HMAX[32];
        static const int32_t MHMAX[32];
        static const int32_t HMIN[32];
        static const int32_t MHMIN[32];
        static const int32_t SQR[32];
        static const int32_t MSQR[32];
        static const int32_t SQRT[32];
        static const int32_t MSQRT[32];

        // define as uint32_t, but load as int32_t
        static const uint32_t LSHV[32];
        static const uint32_t MLSHV[32];
        static const uint32_t LSHS[32];
        static const uint32_t MLSHS[32];
        static const uint32_t RSHV[32];
        static const uint32_t MRSHV[32];
        static const uint32_t RSHS[32];
        static const uint32_t MRSHS[32];
        static const uint32_t ROLV[32];
        static const uint32_t MROLV[32];
        static const uint32_t ROLS[32];
        static const uint32_t MROLS[32];
        static const uint32_t RORV[32];
        static const uint32_t MRORV[32];
        static const uint32_t RORS[32];
        static const uint32_t MRORS[32];

        static const int32_t NEG[32];
        static const int32_t MNEG[32];
        static const int32_t ABS[32];
        static const int32_t MABS[32];

        static const uint32_t ITOU[32];
        static const float    ITOF[32];
    };
};

struct DataSet_1_32f {
    struct inputs {
        static const float inputA[32];
        static const float inputB[32];
        static const float inputC[32];
        static const uint32_t inputUintA[32];
        static const int32_t inputIntA[32];
        static const float scalarA;
        static const bool  maskA[32];
    };

    struct outputs {
        static const float ADDV[32];
        static const float MADDV[32];
        static const float ADDS[32];
        static const float MADDS[32];
        static const float POSTPREFINC[32];
        static const float MPOSTPREFINC[32];
        static const float SUBV[32];
        static const float MSUBV[32];
        static const float SUBS[32];
        static const float MSUBS[32];
        static const float SUBFROMV[32];
        static const float MSUBFROMV[32];
        static const float SUBFROMS[32];
        static const float MSUBFROMS[32];
        static const float POSTPREFDEC[32];
        static const float MPOSTPREFDEC[32];
        static const float MULV[32];
        static const float MMULV[32];
        static const float MULS[32];
        static const float MMULS[32];
        static const float DIVV[32];
        static const float MDIVV[32];
        static const float DIVS[32];
        static const float MDIVS[32];
        static const float RCP[32];
        static const float MRCP[32];
        static const float RCPS[32];
        static const float MRCPS[32];
        static const bool  CMPEQV[32];
        static const bool  CMPEQS[32];
        static const bool  CMPNEV[32];
        static const bool  CMPNES[32];
        static const bool  CMPGTV[32];
        static const bool  CMPGTS[32];
        static const bool  CMPLTV[32];
        static const bool  CMPLTS[32];
        static const bool  CMPGEV[32];
        static const bool  CMPGES[32];
        static const bool  CMPLEV[32];
        static const bool  CMPLES[32];
        static const bool  CMPEV;
        static const bool  CMPES;

        static const float HADD[32];
        static const float MHADD[32];
        static const float HMUL[32];
        static const float MHMUL[32];

        static const float FMULADDV[32];
        static const float MFMULADDV[32];
        static const float FMULSUBV[32];
        static const float MFMULSUBV[32];
        static const float FADDMULV[32];
        static const float MFADDMULV[32];
        static const float FSUBMULV[32];
        static const float MFSUBMULV[32];
        static const float MAXV[32];
        static const float MMAXV[32];
        static const float MAXS[32];
        static const float MMAXS[32];
        static const float MINV[32];
        static const float MMINV[32];
        static const float MINS[32];
        static const float MMINS[32];
        static const float HMAX[32];
        static const float MHMAX[32];
        static const float HMIN[32];
        static const float MHMIN[32];

        static const float NEG[32];
        static const float MNEG[32];
        static const float ABS[32];
        static const float MABS[32];

        static const float SQR[32];
        static const float MSQR[32];
        static const float SQRT[32];
        static const float MSQRT[32];
        static const float ROUND[32];
        static const float MROUND[32];
        static const int32_t TRUNC[32];
        static const int32_t MTRUNC[32];
        static const float FLOOR[32];
        static const float MFLOOR[32];
        static const float CEIL[32];
        static const float MCEIL[32];
        static const bool ISFIN[32];
        static const bool ISINF[32];
        static const bool ISAN[32];
        static const bool ISNAN[32];
        static const bool ISNORM[32];
        static const bool ISSUB[32];
        static const bool ISZERO[32];
        static const bool ISZEROSUB[32];

        static const float SIN[32];
        static const float MSIN[32];
        static const float COS[32];
        static const float MCOS[32];
        static const float TAN[32];
        static const float MTAN[32];
        static const float CTAN[32];
        static const float MCTAN[32];

        static const uint32_t FTOU[32];
        static const int32_t  FTOI[32];
    };
};

#endif
