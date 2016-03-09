#ifndef UME_SIMD_INTERFACE_FUNCTIONS_H_
#define UME_SIMD_INTERFACE_FUNCTIONS_H_

#include "UMESimdTraits.h"

namespace UME
{
namespace SIMD
{
namespace FUNCTIONS
{
    // ADDV
    template<typename VEC_T>
    inline VEC_T add(VEC_T const & src1, VEC_T const & src2) { return src1.add(src2); }
    // ADDS, rhs scalar
    template<typename VEC_T>
    inline VEC_T add(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.add(src2); }
    // ADDS, lhs scalar
    template<typename VEC_T>
    inline VEC_T add(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.add(src1); }
    // MADDV
    template<typename VEC_T>
    inline VEC_T add(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, VEC_T const & src2) { return src1.add(mask, src2); }
    // MADDS, rhs scalar
    template<typename VEC_T>
    inline VEC_T add(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.add(mask, src2); };
    // MADDS, lhs scalar
    template<typename VEC_T>
    inline VEC_T add(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return VEC_T(src1).add(mask, src2); };

    // SADDV
    template<typename VEC_T>
    inline VEC_T sadd(VEC_T const & src1, VEC_T const & src2) { return src1.sadd(src2); }
    // SADDS, rhs scalar
    template<typename VEC_T>
    inline VEC_T sadd(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.sadd(src2); }
    // SADDS, lhs scalar
    template<typename VEC_T>
    inline VEC_T sadd(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.sadd(src1); }
    // MSADDV
    template<typename VEC_T>
    inline VEC_T sadd(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, VEC_T const & src2) { return src1.sadd(mask, src2); }
    // MSADDS, rhs scalar
    template<typename VEC_T>
    inline VEC_T sadd(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.sadd(mask, src2); };
    // MSADDS, lhs scalar
    template<typename VEC_T>
    inline VEC_T sadd(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return VEC_T(src1).sadd(mask, src2); };

    // POSTINC
    template<typename VEC_T>
    inline VEC_T postinc(VEC_T & src1) { return src1.postinc(); }
    // MPOSTINC
    template<typename VEC_T>
    inline VEC_T postinc(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T & src1) { return src1.postinc(mask); }

    // PREFINC
    template<typename VEC_T>
    inline VEC_T & prefinc(VEC_T & src1) { return src1.prefinc(); }
    // MPREFINC
    template<typename VEC_T>
    inline VEC_T & prefinc(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T & src1) { return src1.prefinc(mask); }

    // SUBV
    template<typename VEC_T>
    inline VEC_T sub(VEC_T const & src1, VEC_T const & src2) { return src1.sub(src2); }
    // SUBS, rhs scalar
    template<typename VEC_T>
    inline VEC_T sub(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.sub(src2); }
    // SUBS, lhs scalar
    template<typename VEC_T>
    inline VEC_T sub(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.subfrom(src1); }
    // MSUBV
    template<typename VEC_T>
    inline VEC_T sub(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, VEC_T const & src2) { return src1.sub(mask, src2); }
    // MSUBS, rhs scalar
    template<typename VEC_T>
    inline VEC_T sub(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.sub(mask, src2); }
    // MSUBS, lhs scalar
    template<typename VEC_T>
    inline VEC_T sub(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.subfrom(mask, src1); }

    // SSUBV
    template<typename VEC_T>
    inline VEC_T ssub(VEC_T const & src1, VEC_T const & src2) { return src1.ssub(src2); }
    // SSUBS, rhs scalar
    template<typename VEC_T>
    inline VEC_T ssub(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.ssub(src2); }
    // SSUBS, lhs scalar
    template<typename VEC_T>
    inline VEC_T ssub(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return (VEC_T(src1)).ssub(src2); }
    // MSSUBV
    template<typename VEC_T>
    inline VEC_T ssub(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, VEC_T const & src2) { return src1.ssub(mask, src2); }
    // MSSUBS, rhs scalar
    template<typename VEC_T>
    inline VEC_T ssub(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.ssub(mask, src2); }
    // MSSUBS, lhs scalar
    template<typename VEC_T>
    inline VEC_T ssub(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return (VEC_T(src1)).ssub(mask, src2); }

    // POSTDEC
    template<typename VEC_T>
    inline VEC_T postdec(VEC_T & src1) { return src1.postdec(); }
    // MPOSTDEC
    template<typename VEC_T>
    inline VEC_T postdec(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T & src1) { return src1.postdec(mask); }

    // PREFDEC
    template<typename VEC_T>
    inline VEC_T & prefdec(VEC_T & src1) { return src1.prefdec(); }
    // MPREFDEC
    template<typename VEC_T>
    inline VEC_T & prefdec(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T & src1) { return src1.prefdec(mask); }

    // MULV
    template<typename VEC_T>
    inline VEC_T mul(VEC_T const & src1, VEC_T const & src2) { return src1.mul(src2); }
    // MULS, rhs scalar
    template<typename VEC_T>
    inline VEC_T mul(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.mul(src2); }
    // MULS, lhs scalar
    template<typename VEC_T>
    inline VEC_T mul(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.mul(src1); }
    // MMULV
    template<typename VEC_T>
    inline VEC_T mul(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, VEC_T const & src2) { return src1.mul(mask, src2); }
    // MMULS, rhs scalar
    template<typename VEC_T>
    inline VEC_T mul(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.mul(mask, src2); };
    // MMULS, lhs scalar
    template<typename VEC_T>
    inline VEC_T mul(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return VEC_T(src1).mul(mask, src2); };

    // DIVV
    template<typename VEC_T>
    inline VEC_T div(VEC_T const & src1, VEC_T const & src2) { return src1.div(src2); }
    // DIVS, rhs scalar
    template<typename VEC_T>
    inline VEC_T div(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.div(src2); }
    // DIVS, lhs scalar
    template<typename VEC_T>
    inline VEC_T div(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.rcp(src1); }
    // MDIVV
    template<typename VEC_T>
    inline VEC_T div(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, VEC_T const & src2) { return src1.div(mask, src2); }
    // MDIVS, rhs scalar
    template<typename VEC_T>
    inline VEC_T div(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.div(mask, src2); }
    // MDIVS, lhs scalar
    template<typename VEC_T>
    inline VEC_T div(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.rcp(mask, src1); }

    // RCP
    template<typename VEC_T>
    inline VEC_T rcp(VEC_T const & src1) { return src1.rcp(); }
    // RCP - scalar numerator
    template<typename VEC_T>
    inline VEC_T rcp(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.rcp(src2); }

    // MRCP
    template<typename VEC_T>
    inline VEC_T rcp(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.rcp(mask); }
    // MRCP - scalar numerator
    template<typename VEC_T>
    inline VEC_T rcp(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.rcp(mask, src2); }

    // CMPEQV
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmpeq(VEC_T const & src1, VEC_T const & src2) { return src1.cmpeq(src2); }
    // CMPEQS - rhs scalar
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmpeq(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.cmpeq(src2); }
    // CMPEQS - lhs scalar
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmpeq(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.cmpeq(src1); }

    // CMPNEV
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmpne(VEC_T const & src1, VEC_T const & src2) { return src1.cmpne(src2); }
    // CMPNES - rhs scalar
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmpne(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.cmpne(src2); }
    // CMPNES - lhs scalar
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmpne(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.cmpne(src1); }

    // CMPGTV
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmpgt(VEC_T const & src1, VEC_T const & src2) { return src1.cmpgt(src2); }
    // CMPGTS - rhs scalar
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmpgt(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.cmpgt(src2); }
    // CMPGTS - lhs scalar
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmpgt(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.cmplt(src1); }

    // CMPLTV
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmplt(VEC_T const & src1, VEC_T const & src2) { return src1.cmplt(src2); }
    // CMPLTS - rhs scalar
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmplt(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.cmplt(src2); }
    // CMPLTS - lhs scalar
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmplt(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.cmpgt(src1); }

    // CMPGEV
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmpge(VEC_T const & src1, VEC_T const & src2) { return src1.cmpge(src2); }
    // CMPGES - rhs scalar
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmpge(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.cmpge(src2); }
    // CMPGES - lhs scalar
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmpge(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.cmple(src1); }

    // CMPLEV
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmple(VEC_T const & src1, VEC_T const & src2) { return src1.cmple(src2); }
    // CMPLES - rhs scalar
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmple(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.cmple(src2); }
    // CMPLES - lhs scalar
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T cmple(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.cmpge(src1); }

    // CMPLEV
    template<typename VEC_T>
    inline bool cmpe(VEC_T const & src1, VEC_T const & src2) { return src1.cmpe(src2); }
    // CMPLES - rhs scalar
    template<typename VEC_T>
    inline bool cmpe(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.cmpe(src2); }
    // CMPLES - lhs scalar
    template<typename VEC_T>
    inline bool cmpe(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.cmpe(src1); }

    // UNIQUE
    template<typename VEC_T>
    inline bool unique(VEC_T const & src1) { return src1.unique(); }

    // HADD
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hadd(VEC_T const & src1) { return src1.hadd(); }
    // MHADD
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hadd(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.hadd(mask); }

    // HADDS
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hadd(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCLAR_T const & src2) { return src1.hadd(src2); }
    // MHADDS
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hadd(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCLAR_T const & src2) { return src1.hadd(mask, src2); }

    // HADD
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hmul(VEC_T const & src1) { return src1.hmul(); }
    // MHADD
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hmul(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.hmul(mask); }

    // HMULS
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hmul(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCLAR_T const & src2) { return src1.hmul(src2); }
    // MHMULS
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hmul(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCLAR_T const & src2) { return src1.hmul(mask, src2); }

    // FMULADDV
    template<typename VEC_T>
    inline VEC_T fmuladd(VEC_T const & src1, VEC_T const & src2, VEC_T const & src3) { return src1.fmuladd(src2, src3); }

    // MFMULADDV
    template<typename VEC_T>
    inline VEC_T fmuladd(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, VEC_T const & src2, VEC_T const & src3) { return src1.fmuladd(mask, src2, src3); }

    // FMULSUBV
    template<typename VEC_T>
    inline VEC_T fmulsub(VEC_T const & src1, VEC_T const & src2, VEC_T const & src3) { return src1.fmulsub(src2, src3); }

    // MFMULSUBV
    template<typename VEC_T>
    inline VEC_T fmulsub(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, VEC_T const & src2, VEC_T const & src3) { return src1.fmulsub(mask, src2, src3); }

    // FADDMULV
    template<typename VEC_T>
    inline VEC_T faddmul(VEC_T const & src1, VEC_T const & src2, VEC_T const & src3) { return src1.faddmul(src2, src3); }

    // MFADDMULV
    template<typename VEC_T>
    inline VEC_T faddmul(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, VEC_T const & src2, VEC_T const & src3) { return src1.faddmul(mask, src2, src3); }

    // FSUBMULV
    template<typename VEC_T>
    inline VEC_T fsubmul(VEC_T const & src1, VEC_T const & src2, VEC_T const & src3) { return src1.fsubmul(src2, src3); }

    // MFSUBMULV
    template<typename VEC_T>
    inline VEC_T fsubmul(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, VEC_T const & src2, VEC_T const & src3) { return src1.fsubmul(mask, src2, src3); }

    // MAXV
    template<typename VEC_T>
    inline VEC_T max(VEC_T const & src1, VEC_T const & src2) { return src1.max(src2); }
    // MAXS - rhs scalar
    template<typename VEC_T>
    inline VEC_T max(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.max(src2); }
    // MAXS - lhs scalar
    template<typename VEC_T>
    inline VEC_T max(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.max(src1); }
    // MMAXV
    template<typename VEC_T>
    inline VEC_T max(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, VEC_T const & src2) { return src1.max(mask, src2); }
    // MMAXS - rhs scalar
    template<typename VEC_T>
    inline VEC_T max(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.max(mask, src2); }
    // MMAXS - lhs scalar
    template<typename VEC_T>
    inline VEC_T max(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.max(mask, src1); }


    // MINV
    template<typename VEC_T>
    inline VEC_T min(VEC_T const & src1, VEC_T const & src2) { return src1.min(src2); }
    // MINS - rhs scalar
    template<typename VEC_T>
    inline VEC_T min(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.min(src2); }
    // MINS - lhs scalar
    template<typename VEC_T>
    inline VEC_T min(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.min(src1); }
    // MMINV
    template<typename VEC_T>
    inline VEC_T min(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, VEC_T const & src2) { return src1.min(mask, src2); }
    // MMINS - rhs scalar
    template<typename VEC_T>
    inline VEC_T min(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.min(mask, src2); }
    // MMINS - lhs scalar
    template<typename VEC_T>
    inline VEC_T min(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.min(mask, src1); }

    // HMAX
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hmax(VEC_T const & src1) { return src1.hmax(); }
    // MHMAX
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hmax(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.hmax(mask); }
    // IMAX
    template<typename VEC_T>
    inline uint32_t imax(VEC_T const & src1) { return src1.imax(); }
    // MIMAX
    template<typename VEC_T>
    inline uint32_t imax(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.imax(mask); }

    // HMIN
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hmin(VEC_T const & src1) { return src1.hmin(); }
    // MHMIN
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hmin(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.hmin(mask); }
    // IMIN
    template<typename VEC_T>
    inline uint32_t imin(VEC_T const & src1) { return src1.imin(); }
    // MIMIN
    template<typename VEC_T>
    inline uint32_t imin(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.imin(mask); }

    // BANDV
    template<typename VEC_T>
    inline VEC_T band(VEC_T const & src1, VEC_T const & src2) { return src1.band(src2); }
    // BANDS, rhs scalar
    template<typename VEC_T>
    inline VEC_T band(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.band(src2); }
    // BANDS, lhs scalar
    template<typename VEC_T>
    inline VEC_T band(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.band(src1); }
    // MBANDV
    template<typename VEC_T>
    inline VEC_T band(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, VEC_T const & src2) { return src1.band(mask, src2); }
    // MBANDS, rhs scalar
    template<typename VEC_T>
    inline VEC_T band(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.band(mask, src2); };
    // MBANDS, lhs scalar
    template<typename VEC_T>
    inline VEC_T band(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return VEC_T(src1).band(mask, src2); };

    // BORV
    template<typename VEC_T>
    inline VEC_T bor(VEC_T const & src1, VEC_T const & src2) { return src1.bor(src2); }
    // BORS, rhs scalar
    template<typename VEC_T>
    inline VEC_T bor(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.bor(src2); }
    // BORS, lhs scalar
    template<typename VEC_T>
    inline VEC_T bor(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.bor(src1); }
    // MBORV
    template<typename VEC_T>
    inline VEC_T bor(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, VEC_T const & src2) { return src1.bor(mask, src2); }
    // MBORS, rhs scalar
    template<typename VEC_T>
    inline VEC_T bor(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.bor(mask, src2); };
    // MBORS, lhs scalar
    template<typename VEC_T>
    inline VEC_T bor(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return VEC_T(src1).bor(mask, src2); };

    // BXORV
    template<typename VEC_T>
    inline VEC_T bxor(VEC_T const & src1, VEC_T const & src2) { return src1.bxor(src2); }
    // BXORS, rhs scalar
    template<typename VEC_T>
    inline VEC_T bxor(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.bxor(src2); }
    // BXORS, lhs scalar
    template<typename VEC_T>
    inline VEC_T bxor(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return src2.bxor(src1); }
    // MBXORV
    template<typename VEC_T>
    inline VEC_T bxor(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, VEC_T const & src2) { return src1.bxor(mask, src2); }
    // MBXORS, rhs scalar
    template<typename VEC_T>
    inline VEC_T bxor(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src2) { return src1.bxor(mask, src2); };
    // MBXORS, lhs scalar
    template<typename VEC_T>
    inline VEC_T bxor(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T const & src1, VEC_T const & src2) { return VEC_T(src1).bxor(mask, src2); };

    // BNOT
    template<typename VEC_T>
    inline VEC_T bnot(VEC_T const & src1) { return src1.bnot(); }
    // MBNOT
    template<typename VEC_T>
    inline VEC_T bnot(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.bnot(mask); }

    // HBAND
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hband(VEC_T const & src1) { return src1.hband(); }
    // MHBAND
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hband(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.hband(mask); }

    // HBOR
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hbor(VEC_T const & src1) { return src1.hbor(); }
    // MHBOR
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hbor(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.hbor(mask); }

    // HBXOR
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hbxor(VEC_T const & src1) { return src1.hbxor(); }
    // MHBXOR
    template<typename VEC_T>
    inline typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T hbxor(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.hbxor(mask); }

    // GATHERS
    template<typename VEC_T>
    inline VEC_T & gather(
        VEC_T & dst,
        typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T* baseAddr,
        typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T* indices)
    {
        return dst.gather(baseAddr, indices);
    }
    // MGATHERS
    template<typename VEC_T>
    inline VEC_T & gather(
        typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask,
        VEC_T & dst,
        typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T* baseAddr,
        typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T* indices)
    {
        return dst.gather(mask, baseAddr, indices);
    }
    // GATHERV
    template<typename VEC_T>
    inline VEC_T & gather(
        VEC_T & dst,
        typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T* baseAddr,
        typename UME::SIMD::SIMDTraits<VEC_T>::UINT_VEC_T* indices)
    {
        return dst.gather(baseAddr, indices);
    }
    // MGATHERV
    template<typename VEC_T>
    inline VEC_T & gather(
        typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask,
        VEC_T & dst,
        typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T* baseAddr,
        typename UME::SIMD::SIMDTraits<VEC_T>::UINT_VEC_T* indices)
    {
        return dst.gather(mask, baseAddr, indices);
    }

    // SCATTERS
    template<typename VEC_T>
    typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T* scatter(
        VEC_T & src1,
        typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T* baseAddr,
        typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T* indices)
    {
        return src1.scatter(baseAddr, indices);
    }

    // MSCATTERS
    template<typename VEC_T>
    typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T* scatter(
        typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T* mask,
        VEC_T & src1,
        typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T* baseAddr,
        typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T* indices)
    {
        return src1.scatter(baseAddr, indices);
    }

    // SCATTERV
    template<typename VEC_T>
    typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T* scatter(
        VEC_T & src1,
        typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T* baseAddr,
        typename UME::SIMD::SIMDTraits<VEC_T>::UINT_VEC_T* indices)
    {
        return src1.scatter(baseAddr, indices);
    }

    // MSCATTERV
    template<typename VEC_T>
    typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T* scatter(
        typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T* mask,
        VEC_T & src1,
        typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T* baseAddr,
        typename UME::SIMD::SIMDTraits<VEC_T>::UINT_VEC_T* indices)
    {
        return src1.scatter(baseAddr, indices);
    }

    // LSHV
    template<typename VEC_T>
    VEC_T lsh(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::UINT_VEC_T const & src2) { return src1.lsh(src2); }
    // MLSHV
    template<typename VEC_T>
    VEC_T lsh(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::UINT_VEC_T const & src2) { return src1.lsh(mask, src2); }
    // LSHS - rhs scalar
    template<typename VEC_T>
    VEC_T lsh(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T src2) { return src1.lsh(src2); }
    // LSHS - lhs scalar
    template<typename VEC_T>
    VEC_T lsh(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T src1, VEC_T const & src2 ) { return VEC_T(src1).lsh(src2); }
    // MLSHS - rhs scalar
    template<typename VEC_T>
    VEC_T lsh(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T src2) { return src1.lsh(mask, src2); }
    // MLSHS - lhs scalar
    template<typename VEC_T>
    VEC_T lsh(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T src1, VEC_T const & src2) { return VEC_T(src1).lsh(mask, src2); }

    // RSHV
    template<typename VEC_T>
    VEC_T rsh(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::UINT_VEC_T const & src2) { return src1.rsh(src2); }
    // MRSHV
    template<typename VEC_T>
    VEC_T rsh(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::UINT_VEC_T const & src2) { return src1.rsh(mask, src2); }
    // RSHS - rhs scalar
    template<typename VEC_T>
    VEC_T rsh(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T src2) { return src1.rsh(src2); }
    // RSHS - lhs scalar
    template<typename VEC_T>
    VEC_T rsh(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T src1, VEC_T const & src2) { return VEC_T(src1).rsh(src2); }
    // MRSHS - rhs scalar
    template<typename VEC_T>
    VEC_T rsh(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T src2) { return src1.rsh(mask, src2); }
    // MRSHS - lhs scalar
    template<typename VEC_T>
    VEC_T rsh(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T src1, VEC_T const & src2) { return VEC_T(src1).rsh(mask, src2); }

    // ROLV
    template<typename VEC_T>
    VEC_T rol(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::UINT_VEC_T const & src2) { return src1.rol(src2); }
    // MROLV
    template<typename VEC_T>
    VEC_T rol(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::UINT_VEC_T const & src2) { return src1.rol(mask, src2); }
    // ROLS - rhs scalar
    template<typename VEC_T>
    VEC_T rol(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T src2) { return src1.rol(src2); }
    // ROLS - lhs scalar
    template<typename VEC_T>
    VEC_T rol(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T src1, VEC_T const & src2) { return VEC_T(src1).rol(src2); }
    // MROLS - rhs scalar
    template<typename VEC_T>
    VEC_T rol(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T src2) { return src1.rol(mask, src2); }
    // MROLS - lhs scalar
    template<typename VEC_T>
    VEC_T rol(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T src1, VEC_T const & src2) { return VEC_T(src1).rol(mask, src2); }

    // RORV
    template<typename VEC_T>
    VEC_T ror(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::UINT_VEC_T const & src2) { return src1.ror(src2); }
    // MRORV
    template<typename VEC_T>
    VEC_T ror(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::UINT_VEC_T const & src2) { return src1.ror(mask, src2); }
    // RORS - rhs scalar
    template<typename VEC_T>
    VEC_T ror(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T src2) { return src1.ror(src2); }
    // RORS - lhs scalar
    template<typename VEC_T>
    VEC_T ror(typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T src1, VEC_T const & src2) { return VEC_T(src1).ror(src2); }
    // MRORS - rhs scalar
    template<typename VEC_T>
    VEC_T ror(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T src2) { return src1.ror(mask, src2); }
    // MRORS - lhs scalar
    template<typename VEC_T>
    VEC_T ror(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_UINT_T src1, VEC_T const & src2) { return VEC_T(src1).ror(mask, src2); }


    // PACK
    template<typename VEC_T>
    VEC_T & pack(VEC_T & dst1, typename UME::SIMD::SIMDTraits<VEC_T>::HALF_LEN_VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::HALF_LEN_VEC_T const & src2) {
        return dst1.pack(src1, src2);
    }
    // PACKLO
    template<typename VEC_T>
    VEC_T & packlo(VEC_T & dst1, typename UME::SIMD::SIMDTraits<VEC_T>::HALF_LEN_VEC_T const & src1) {
        return dst1.packlo(src1);
    }
    // PACKHI
    template<typename VEC_T>
    VEC_T & packhi(VEC_T & dst1, typename UME::SIMD::SIMDTraits<VEC_T>::HALF_LEN_VEC_T const & src1) {
        return dst1.packhi(src1);
    }

    // UNPACK
    template<typename VEC_T>
    void unpack(VEC_T const & src1, typename UME::SIMD::SIMDTraits<VEC_T>::HALF_LEN_VEC_T & dst1, typename UME::SIMD::SIMDTraits<VEC_T>::HALF_LEN_VEC_T & dst2) {
        src1.unpack(dst1, dst2);
    }
    // UNPACKLO
    template<typename VEC_T>
    typename UME::SIMD::SIMDTraits<VEC_T>::HALF_LEN_VEC_T unpacklo(VEC_T const & src1) {
        return src1.unpacklo();
    }
    // UNPACKHI
    template<typename VEC_T>
    typename UME::SIMD::SIMDTraits<VEC_T>::HALF_LEN_VEC_T unpackhi(VEC_T const & src1) {
        return src1.unpackhi();
    }

    // NEG
    template<typename VEC_T>
    VEC_T neg(VEC_T const & src1) { return src1.neg(); }
    // MNEG
    template<typename VEC_T>
    VEC_T neg(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.neg(mask); }
    // ABS
    template<typename VEC_T>
    VEC_T abs(VEC_T const & src1) { return src1.abs(); }
    // MABS
    template<typename VEC_T>
    VEC_T abs(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.abs(mask); }

    // SQR
    template<typename VEC_T>
    VEC_T sqr(VEC_T const & src1) { return src1.sqr(); }
    // MSQR
    template<typename VEC_T>
    VEC_T sqr(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.sqr(mask); }

    // SQRT
    template<typename VEC_T>
    VEC_T sqrt(VEC_T const & src1) { return src1.sqrt(); }
    // MSQRT
    template<typename VEC_T>
    VEC_T sqrt(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.sqrt(mask); }

    // RSQRT
    template<typename VEC_T>
    VEC_T rsqrt(VEC_T const & src1) { return src1.rsqrt(); }
    // MRSQRT
    template<typename VEC_T>
    VEC_T rsqrt(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.rsqrt(mask); }

    // ROUND
    template<typename VEC_T>
    VEC_T round(VEC_T const & src1) { return src1.round(); }
    // MROUND
    template<typename VEC_T>
    VEC_T round(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.round(mask); }

    // TRUNC
    template<typename VEC_T>
    typename UME::SIMD::SIMDTraits<VEC_T>::INT_VEC_T trunc(VEC_T const & src1) { return src1.trunc(); }

    // MTRUNC
    template<typename VEC_T>
    typename UME::SIMD::SIMDTraits<VEC_T>::INT_VEC_T trunc(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.trunc(mask); }

    // FLOOR
    template<typename VEC_T>
    VEC_T floor(VEC_T const & src1) { return src1.floor(); }
    // MFLOOR
    template<typename VEC_T>
    VEC_T floor(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.floor(mask); }

    // CEIL
    template<typename VEC_T>
    VEC_T ceil(VEC_T const & src1) { return src1.ceil(); }
    // MCEIL
    template<typename VEC_T>
    VEC_T ceil(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.ceil(mask); }

    // ISFIN
    template<typename VEC_T>
    typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T isfin(VEC_T const & src1) { return src1.isfin(); }
    // ISINF
    template<typename VEC_T>
    typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T isinf(VEC_T const & src1) { return src1.isinf(); }
    // ISAN
    template<typename VEC_T>
    typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T isan(VEC_T const & src1) { return src1.isan(); }
    // ISNAN
    template<typename VEC_T>
    typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T isnan(VEC_T const & src1) { return src1.isnan(); }
    // ISNORM
    template<typename VEC_T>
    typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T isnorm(VEC_T const & src1) { return src1.isnorm(); }
    // ISSUB
    template<typename VEC_T>
    typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T issub(VEC_T const & src1) { return src1.issub(); }
    // ISZERO
    template<typename VEC_T>
    typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T iszero(VEC_T const & src1) { return src1.iszero(); }
    // ISZEROSUB
    template<typename VEC_T>
    typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T iszerosub(VEC_T const & src1) { return src1.iszerosub(); }

    // EXP
    template<typename VEC_T>
    inline VEC_T exp(VEC_T const & src1) { return src1.exp(); }
    // MEXP
    template<typename VEC_T>
    inline VEC_T exp(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.exp(mask); }

    // LOG
    template<typename VEC_T>
    inline VEC_T log(VEC_T const & src1) { return src1.log(); }
    // MLOG
    template<typename VEC_T>
    inline VEC_T log(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.log(mask); }

    // LOG10
    template<typename VEC_T>
    inline VEC_T log10(VEC_T const & src1) { return src1.log10(); }
    // MLOG10
    template<typename VEC_T>
    inline VEC_T log10(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.log10(mask); }

    // LOG2
    template<typename VEC_T>
    inline VEC_T log2(VEC_T const & src1) { return src1.log2(); }
    // MLOG2
    template<typename VEC_T>
    inline VEC_T log2(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.log2(mask); }

    // SIN
    template<typename VEC_T>
    inline VEC_T sin(VEC_T const & src1) { return src1.sin(); }
    // MSIN
    template<typename VEC_T>
    inline VEC_T sin(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.sin(mask); }

    // COS
    template<typename VEC_T>
    inline VEC_T cos(VEC_T const & src1) { return src1.cos(); }
    // MCOS
    template<typename VEC_T>
    inline VEC_T cos(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.cos(mask); }

    // SINCOS
    template<typename VEC_T>
    void sincos(VEC_T const & src1, VEC_T & dst1, VEC_T & dst2) { src1.sincos(dst1, dst2); }
    // MSINCOS
    template<typename VEC_T>
    void sincos(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1, VEC_T & dst1, VEC_T & dst2) { src1.sincos(mask, dst1, dst2); }

    // TAN
    template<typename VEC_T>
    inline VEC_T tan(VEC_T const & src1) { return src1.tan(); }
    // MTAN
    template<typename VEC_T>
    inline VEC_T tan(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.tan(mask); }

    // CTAN
    template<typename VEC_T>
    inline VEC_T ctan(VEC_T const & src1) { return src1.ctan(); }
    // MCTAN
    template<typename VEC_T>
    inline VEC_T ctan(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.ctan(mask); }

    // ATAN
    template<typename VEC_T>
    inline VEC_T atan(VEC_T const & src1) { return src1.atan(); }
    // MATAN
    template<typename VEC_T>
    inline VEC_T atan(typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T const & mask, VEC_T const & src1) { return src1.atan(mask); }

}
}
}

#endif
