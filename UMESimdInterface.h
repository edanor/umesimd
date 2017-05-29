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

#ifndef UME_SIMD_INTERFACE_H_
#define UME_SIMD_INTERFACE_H_

#include <cmath>
#include <limits>

#include "UMEBasicTypes.h"
#include "UMESimdScalarEmulation.h"
#include "UMESimdVectorEmulation.h"

namespace UME
{
namespace SIMD
{
    // **********************************************************************
    // *
    // *  Declaration of IndexVectorInterface class
    // *
    // **********************************************************************
 
    //// Checks if N is power of 2
    //template<unsigned int N>
    //struct isPow2
    //{
    //    enum {
    //        value = N && !(N & (N -1))
    //    };
    //};

    //// Calculates number of bits required to represent element of swizzle mask.
    //template<unsigned int N, unsigned int P=0>
    //struct SwizzleMaskBitsPerElement
    //{
    //    //static const unsigned int value = LogBase2<N/2, P+1>.value;
    //    enum {
    //        value = SwizzleMaskBitsPerElement<N/2 + !(isPow2<N>::value), P+1>::value
    //    };
    //};

    //// Partial specialization for base case
    //template<unsigned P>
    //struct SwizzleMaskBitsPerElement<0, P>
    //{
    //    enum {
    //        value = P
    //    };
    //};

    //template<unsigned P>
    //struct SwizzleMaskBitsPerElement<1, P>
    //{
    //    enum {
    //        value = P
    //    };
    //};
    
    template<class DERIVED_SWIZZLE_TYPE, uint32_t SMASK_LEN>
    class SIMDSwizzleMaskBaseInterface
    {
        // Declarations only. These operators should be overriden in derived types.
        // EXTRACT
        UME_FUNC_ATTRIB bool extract(uint32_t index);
        // EXTRACT
        UME_FUNC_ATTRIB bool operator[] (uint32_t index);
        // INSERT
        UME_FUNC_ATTRIB void insert(uint32_t index, uint32_t value);

    protected:
        ~SIMDSwizzleMaskBaseInterface() {};

    public:
        // LENGTH
        constexpr static uint32_t length () { return SMASK_LEN; };

        // LOAD
        UME_FUNC_ATTRIB DERIVED_SWIZZLE_TYPE & load(uint32_t const * addr) {
            return SCALAR_EMULATION::load<DERIVED_SWIZZLE_TYPE, uint32_t>(static_cast<DERIVED_SWIZZLE_TYPE &>(*this), addr);
        }

        UME_FUNC_ATTRIB DERIVED_SWIZZLE_TYPE & load(uint64_t const * addr) {
            return SCALAR_EMULATION::load<DERIVED_SWIZZLE_TYPE, uint64_t>(static_cast<DERIVED_SWIZZLE_TYPE &>(*this), addr);
        }

        // ALIGNMENT
        static int alignment () { return SMASK_LEN*sizeof(uint32_t); };
    };

    // This class represents a vector of VEC_LEN scalars and is used for emulation.
    template<typename SCALAR_TYPE, uint32_t VEC_LEN> 
    class SIMDVecEmuRegister
    {
    private:
        SCALAR_TYPE reg[VEC_LEN];
    public:
        SIMDVecEmuRegister() {
            UME_EMULATION_WARNING();
            for(unsigned int i = 0; i < VEC_LEN; i++) { reg[i] = 0; }
        }

        SIMDVecEmuRegister(SCALAR_TYPE x) {
            UME_EMULATION_WARNING();
            for(unsigned int i = 0; i < VEC_LEN; i++) { reg[i] = x; }
        }

        SIMDVecEmuRegister(SIMDVecEmuRegister const & x) {
            UME_EMULATION_WARNING();
            for(unsigned int i = 0; i < VEC_LEN; i++) { reg[i] = x.reg[i]; }
        }

        // Also define a non-modifying access operator
        UME_FUNC_ATTRIB SCALAR_TYPE operator[] (uint32_t index) const { 
            SCALAR_TYPE temp = reg[index];    
            return temp; 
        }
            
        UME_FUNC_ATTRIB void insert(uint32_t index, SCALAR_TYPE value){
            reg[index] = value; 
        }
    };

    template<uint32_t MASK_LEN>
    struct MaskAsInt{
        uint64_t m0;
    };
    
    template<>
    struct MaskAsInt<128> {
        uint64_t m0;
        uint64_t m1;
    };
    
    // **********************************************************************
    // *
    // *  Declaration of SIMDMaskBaseInterface class 
    // *
    // *    This class should be used as a basic class for all masks. 
    // *    All masks should implement interface contained in 
    // *    SIMDMaskBaseInterface. If the derived class does not provide an 
    // *    overload for given operation, this class will default 
    // *    to scalar emulation, thus providing interface coherence over
    // *    different plugins.
    // *
    // **********************************************************************

    template<class DERIVED_MASK_TYPE, 
            typename MASK_BASE_TYPE, 
            uint32_t MASK_LEN>
    class SIMDMaskBaseInterface {
        // Declarations only. These operators should be overriden in derived types.
        // EXTRACT
        UME_FUNC_ATTRIB bool extract(uint32_t index);
        // EXTRACT
        UME_FUNC_ATTRIB bool operator[] (uint32_t index);
        // INSERT
        UME_FUNC_ATTRIB void insert(uint32_t index, bool value);

    protected:
        ~SIMDMaskBaseInterface() {}

    public:
        // LENGTH
        constexpr static uint32_t length() { return MASK_LEN; }

        // ALIGNMENT
        constexpr static int alignment() { return MASK_LEN*sizeof(MASK_BASE_TYPE); }

        // LOAD
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & load(bool const * addr) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::load<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE &>(*this), addr);
        }

        // LOADA
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & loada(bool const * addrAligned) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::loadAligned<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE &>(*this), addrAligned);
        }

        // STORE
        UME_FUNC_ATTRIB bool* store(bool* addr) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::store<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE const &>(*this), addr);
        }

        // STOREA
        UME_FUNC_ATTRIB bool* storea(bool* addrAligned) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::storeAligned<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE const &>(*this), addrAligned);
        }

        // GATHERU
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & gatheru (bool const * baseAddr, uint32_t stride) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::gatheru<DERIVED_MASK_TYPE, bool> (static_cast<DERIVED_MASK_TYPE &>(*this), baseAddr, stride);
        }

        // SCATTERU
        UME_FUNC_ATTRIB bool* scatteru (bool * baseAddr, uint32_t stride) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::scatteru<DERIVED_MASK_TYPE, bool> (static_cast<DERIVED_MASK_TYPE &>(*this), baseAddr, stride);
        }

        // ASSIGNV
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & assign(DERIVED_MASK_TYPE const & maskOp) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::assign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & operator= (DERIVED_MASK_TYPE const & maskOp) {
            return assign(maskOp);
        }

        // MASSIGNV
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & assign(DERIVED_MASK_TYPE const & mask, DERIVED_MASK_TYPE const & maskOp) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::assign<DERIVED_MASK_TYPE, DERIVED_MASK_TYPE>(mask, static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        }

        // ASSIGNS
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & assign(bool scalarOp) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::assign<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE &>(*this), scalarOp);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & operator= (bool scalarOp) {
            return assign(scalarOp);
        }

        // MASSIGNS
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & assign(DERIVED_MASK_TYPE const & mask, bool scalarOp) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::assign<DERIVED_MASK_TYPE, bool, DERIVED_MASK_TYPE>(mask, static_cast<DERIVED_MASK_TYPE &>(*this), scalarOp);
        }

        // LANDV
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE land(DERIVED_MASK_TYPE const & maskOp) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAnd<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), maskOp);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE operator& (DERIVED_MASK_TYPE const & maskOp) const {
            return land(maskOp);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE operator&& (DERIVED_MASK_TYPE const & maskOp) const {
            return land(maskOp);
        }

        // LANDS
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE land(bool value) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAnd<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), value);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE operator& (bool value) const {
            return land(value);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE operator&& (bool value) const {
            return land(value);
        }

        // LANDVA
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & landa(DERIVED_MASK_TYPE const & maskOp) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAndAssign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & operator&= (DERIVED_MASK_TYPE const & maskOp) {
            return landa(maskOp);
        }

        // LANDSA
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & landa(bool value) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAndAssign<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE &>(*this), value);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & operator&= (bool value) {
            return landa(value);
        }

        // LORV
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE lor(DERIVED_MASK_TYPE const & maskOp) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryOr<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), maskOp);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE operator| (DERIVED_MASK_TYPE const & maskOp) const {
            return lor(maskOp);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE operator|| (DERIVED_MASK_TYPE const & maskOp) const {
            return lor(maskOp);
        }

        // LORS
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE lor(bool value) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryOr<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), value);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE operator| (bool value) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryOr<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE const &>(*this), value);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE operator|| (bool value) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryOr<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE const &>(*this), value);
        }

        // LORVA
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & lora(DERIVED_MASK_TYPE const & maskOp) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryOrAssign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & operator|= (DERIVED_MASK_TYPE const & maskOp) {
            return lora(maskOp);
        }

        // LORSA
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & lora(bool value) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryOrAssign<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE &>(*this), value);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & operator|= (bool value) {
            return lora(value);
        }

        // LXORV
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE lxor(DERIVED_MASK_TYPE const & maskOp) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryXor<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), maskOp);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE operator^ (DERIVED_MASK_TYPE const & maskOp) const {
            return lxor(maskOp);
        }

        // LXORS
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE lxor(bool value) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryXor<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE const &>(*this), value);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE operator^ (bool value) const {
            return lxor(value);
        }

        // LXORVA
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & lxora(DERIVED_MASK_TYPE const & maskOp) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryXorAssign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this), maskOp);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & operator^= (DERIVED_MASK_TYPE const & maskOp) {
            return lxora(maskOp);
        }

        // LXORSA
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & lxora(bool value) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryXorAssign<DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE &>(*this), value);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & operator^= (bool value) {
            return lxora(value);
        }

        // LNOT
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE lnot () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::logicalNot<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this));
        }
        
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE operator!() const {
            return lnot();
        }

        // LNOTA
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE & lnota () {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::logicalNotAssign<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE &>(*this));
        }

        // LANDNOTV
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE landnot(DERIVED_MASK_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::logicalAndNot<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), b);
        }

        // LANDNOTS
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE landnot(bool b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::logicalAndNot<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), b);
        }

        // CMPEQV
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE cmpeq(DERIVED_MASK_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isEqual<DERIVED_MASK_TYPE, DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE operator== (DERIVED_MASK_TYPE const & b) const {
            return cmpeq(b);
        }

        // CMPEQS
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE cmpeq(bool b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isEqual<DERIVED_MASK_TYPE, DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE operator== (bool b) const {
            return cmpeq(b);
        }

        // CMPNEV
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE cmpne(DERIVED_MASK_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isNotEqual<DERIVED_MASK_TYPE, DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE operator!= (DERIVED_MASK_TYPE const & b) const {
            return cmpne(b);
        }

        // CMPNES
        UME_FUNC_ATTRIB DERIVED_MASK_TYPE cmpne(bool b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isNotEqual<DERIVED_MASK_TYPE, DERIVED_MASK_TYPE, bool>(static_cast<DERIVED_MASK_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_MASK_TYPE operator!= (bool b) const {
            return  cmpne(b);
        }

        // HLAND
        UME_FUNC_ATTRIB bool hland() const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceLogicalAnd<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this));
        }

        // HLOR
        UME_FUNC_ATTRIB bool hlor() const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceLogicalOr<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this));
        }

        // HLXOR
        UME_FUNC_ATTRIB bool hlxor() const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceLogicalXor<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this));
        }

        // CMPEV
        UME_FUNC_ATTRIB bool cmpe(DERIVED_MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isExact<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), mask);
        }

        // CMPES
        UME_FUNC_ATTRIB bool cmpe(bool b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isExact<DERIVED_MASK_TYPE>(static_cast<DERIVED_MASK_TYPE const &>(*this), DERIVED_MASK_TYPE(b));
        }
    };

    // **********************************************************************
    // *
    // *  Declaration of IntermediateMask class 
    // *
    // *    This class is a helper class used in masked version of
    // *    operator[]. This object is not copyable and can only be created
    // *    from its vector type (VEC_TYPE) for temporary use. 
    // *
    // **********************************************************************
    template<class VEC_TYPE, class SCALAR_TYPE, class MASK_TYPE>
    class IntermediateMask {
    public:
        // MASSIGNV
        UME_FUNC_ATTRIB void operator=(VEC_TYPE const & vecRhs) const {
            mVecRef.assign(mMaskRef, vecRhs);
        }

        // MASSIGNS
        UME_FUNC_ATTRIB void operator=(SCALAR_TYPE scalarRhs) const {
            mVecRef.assign(mMaskRef, scalarRhs);
        }

        // MADDVA
        UME_FUNC_ATTRIB void operator+=(VEC_TYPE const & vecRhs) const {
            mVecRef.adda(mMaskRef, vecRhs);
        }

        // MADDSA
        UME_FUNC_ATTRIB void operator+=(SCALAR_TYPE scalarRhs) const {
            mVecRef.adda(mMaskRef, scalarRhs);
        }

        // MSUBVA
        UME_FUNC_ATTRIB void operator-= (VEC_TYPE const & vecRhs) const {
            mVecRef.suba(mMaskRef, vecRhs);
        }

        // MSUBSA
        UME_FUNC_ATTRIB void operator-=(SCALAR_TYPE scalarRhs) const {
            mVecRef.suba(mMaskRef, scalarRhs);
        }

        // MMULVA
        UME_FUNC_ATTRIB void operator*= (VEC_TYPE const & vecRhs) const {
            mVecRef.mula(mMaskRef, vecRhs);
        }

        // MMULSA
        UME_FUNC_ATTRIB void operator*=(SCALAR_TYPE scalarRhs) const {
            mVecRef.mula(mMaskRef, scalarRhs);
        }

        // MDIVVA
        UME_FUNC_ATTRIB void operator/= (VEC_TYPE const & vecRhs) const {
            mVecRef.diva(mMaskRef, vecRhs);
        }

        // MDIVSA
        UME_FUNC_ATTRIB void operator/=(SCALAR_TYPE scalarRhs) const {
            mVecRef.diva(mMaskRef, scalarRhs);
        }

        // MBANDVA
        UME_FUNC_ATTRIB void operator&= (VEC_TYPE const & vecRhs) const {
            mVecRef.banda(mMaskRef, vecRhs);
        }

        // MBANDSA
        UME_FUNC_ATTRIB void operator&=(SCALAR_TYPE scalarRhs) const {
            mVecRef.banda(mMaskRef, scalarRhs);
        }

        // MBORVA
        UME_FUNC_ATTRIB void operator|= (VEC_TYPE const & vecRhs) const {
            mVecRef.bora(mMaskRef, vecRhs);
        }

        // MBORSA
        UME_FUNC_ATTRIB void operator|=(SCALAR_TYPE scalarRhs) const {
            mVecRef.bora(mMaskRef, scalarRhs);
        }

        // MBXORVA
        UME_FUNC_ATTRIB void operator^= (VEC_TYPE const & vecRhs) const {
            mVecRef.bxora(mMaskRef, vecRhs);
        }

        // MBXORSA
        UME_FUNC_ATTRIB void operator^=(SCALAR_TYPE scalarRhs) const {
            mVecRef.bxora(mMaskRef, scalarRhs);
        }

        // This object should be only constructible by the
        // vector type using it.
        IntermediateMask();
        IntermediateMask(IntermediateMask const &);
        IntermediateMask & operator= (IntermediateMask const &); 

        explicit IntermediateMask(uint32_t);
#ifndef __NVCC__
        // NVCC has a problem handling this friendship relation.
        // Make it public only when compiling CUDA
        private:
        friend VEC_TYPE;
#endif

        UME_FUNC_ATTRIB explicit IntermediateMask(MASK_TYPE const & mask, VEC_TYPE & vec) : mMaskRef(mask), mVecRef(vec) {}

        MASK_TYPE const & mMaskRef;
        VEC_TYPE & mVecRef;
    };

    // **********************************************************************
    // *
    // *  Declaration of IntermediateIndex class 
    // *
    // *    This class is a helper class used in assignment version of
    // *    operator[SCALAR]. This object is not copyable and can only be created
    // *    from its vector type (VEC_TYPE) for temporary use. It's purpose is
    // *    to allow LHS assignments to expressions of form:
    // *
    // *     <vec>[index] <assignment_operator> <RHS scalar value>
    // *
    // **********************************************************************
    template<class VEC_TYPE, class SCALAR_TYPE>
    class IntermediateIndex {
    public:
        IntermediateIndex(IntermediateIndex const & x) : mIndexRef(x.mIndexRef), mVecRef_RW(x.mVecRef_RW) {}
        IntermediateIndex & operator= (IntermediateIndex const & x) {
            mVecRef_RW.insert(mIndexRef, x.mVecRef_RW.extract(x.mIndexRef));
            return *this;
        }

        // MASSIGNS
        UME_FUNC_ATTRIB void operator= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, scalarRhs);
        }

        UME_FUNC_ATTRIB void operator+= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, mVecRef_RW[mIndexRef] + scalarRhs);
        }

        UME_FUNC_ATTRIB void operator-= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, mVecRef_RW[mIndexRef] - scalarRhs);
        }

        UME_FUNC_ATTRIB void operator*= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, mVecRef_RW[mIndexRef] * scalarRhs);
        }

        UME_FUNC_ATTRIB void operator/= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, mVecRef_RW[mIndexRef] / scalarRhs);
        }

        UME_FUNC_ATTRIB void operator%= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, mVecRef_RW[mIndexRef] % scalarRhs);
        }

        UME_FUNC_ATTRIB void operator&= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, mVecRef_RW[mIndexRef] & scalarRhs);
        }

        UME_FUNC_ATTRIB void operator|= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, mVecRef_RW[mIndexRef] | scalarRhs);
        }

        UME_FUNC_ATTRIB void operator^= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, mVecRef_RW[mIndexRef] ^ scalarRhs);
        }

        UME_FUNC_ATTRIB void operator<<= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, mVecRef_RW[mIndexRef] << scalarRhs);
        }

        UME_FUNC_ATTRIB void operator>>= (SCALAR_TYPE scalarRhs) {
            mVecRef_RW.insert(mIndexRef, mVecRef_RW[mIndexRef] >> scalarRhs);
        }

        UME_FUNC_ATTRIB operator SCALAR_TYPE() const { return mVecRef_RW.extract(mIndexRef); }

        // Comparison operators accept any type of scalar to allow mixing 
        // scalar types.
        template<
            typename T,
            typename = typename std::enable_if<std::is_fundamental<T>::value, void*>::type
            >
        UME_FUNC_ATTRIB bool operator==(
                T const & rhs) const {
            return mVecRef_RW.extract(mIndexRef) == SCALAR_TYPE(rhs);
        }
        UME_FUNC_ATTRIB bool operator== (IntermediateIndex const & x) const {
            return mVecRef_RW.extract(mIndexRef) ==
                x.mVecRef_RW.extract(x.mIndexRef);
        }
        template<typename T>
        UME_FUNC_ATTRIB bool operator!=(T const & rhs) const {
            return mVecRef_RW.extract(mIndexRef) != SCALAR_TYPE(rhs);
        }
        UME_FUNC_ATTRIB bool operator!= (IntermediateIndex const & x) const {
            return mVecRef_RW.extract(mIndexRef) !=
                x.mVecRef_RW.extract(x.mIndexRef);
        }
        template<typename T>
        UME_FUNC_ATTRIB SCALAR_TYPE operator+ (T const & x) const {
            return mVecRef_RW.extract(mIndexRef) + SCALAR_TYPE(x);
        }
        UME_FUNC_ATTRIB SCALAR_TYPE operator+ (IntermediateIndex const & x) const {
            return mVecRef_RW.extract(mIndexRef) +
                x.mVecRef_RW.extract(x.mIndexRef);
        }
        template<typename T>
        UME_FUNC_ATTRIB SCALAR_TYPE operator- (T const & x) const {
            return mVecRef_RW.extract(mIndexRef) - SCALAR_TYPE(x);
        }
        UME_FUNC_ATTRIB SCALAR_TYPE operator- (IntermediateIndex const & x) const {
            return mVecRef_RW.extract(mIndexRef) -
                x.mVecRef_RW.extract(x.mIndexRef);
        }
        template<typename T>
        UME_FUNC_ATTRIB SCALAR_TYPE operator* (T const & x) const {
            return mVecRef_RW.extract(mIndexRef) * SCALAR_TYPE(x);
        }
        UME_FUNC_ATTRIB SCALAR_TYPE operator* (IntermediateIndex const & x) const {
            return mVecRef_RW.extract(mIndexRef) *
                x.mVecRef_RW.extract(x.mIndexRef);
        }
        template<typename T>
        UME_FUNC_ATTRIB SCALAR_TYPE operator/ (T const & x) const {
            return mVecRef_RW.extract(mIndexRef) / SCALAR_TYPE(x);
        }
        UME_FUNC_ATTRIB SCALAR_TYPE operator/ (IntermediateIndex const & x) const {
            return mVecRef_RW.extract(mIndexRef) /
                x.mVecRef_RW.extract(x.mIndexRef);
        }
        template<typename T>
        UME_FUNC_ATTRIB SCALAR_TYPE operator% (T const & x) const {
            return mVecRef_RW.extract(mIndexRef) % SCALAR_TYPE(x);
        }
        UME_FUNC_ATTRIB SCALAR_TYPE operator% (IntermediateIndex const & x) const {
            return mVecRef_RW.extract(mIndexRef) %
                x.mVecRef_RW.extract(x.mIndexRef);
        }
        template<typename T>
        UME_FUNC_ATTRIB SCALAR_TYPE operator& (T const & x) const {
            return mVecRef_RW.extract(mIndexRef) & SCALAR_TYPE(x);
        }
        UME_FUNC_ATTRIB SCALAR_TYPE operator& (IntermediateIndex const & x) const {
            return mVecRef_RW.extract(mIndexRef) &
                x.mVecRef_RW.extract(x.mIndexRef);
        }
        template<typename T>
        UME_FUNC_ATTRIB SCALAR_TYPE operator| (T const & x) const {
            return mVecRef_RW.extract(mIndexRef) | SCALAR_TYPE(x);
        }
        UME_FUNC_ATTRIB SCALAR_TYPE operator| (IntermediateIndex const & x) const {
            return mVecRef_RW.extract(mIndexRef) |
                x.mVecRef_RW.extract(x.mIndexRef);
        }
        template<typename T>
        UME_FUNC_ATTRIB SCALAR_TYPE operator^ (T const & x) const {
            return mVecRef_RW.extract(mIndexRef) ^ SCALAR_TYPE(x);
        }
        UME_FUNC_ATTRIB SCALAR_TYPE operator^ (IntermediateIndex const & x) const {
            return mVecRef_RW.extract(mIndexRef) ^
                x.mVecRef_RW.extract(x.mIndexRef);
        }
        template<typename T>
        UME_FUNC_ATTRIB SCALAR_TYPE operator<< (T const & x) const {
            return mVecRef_RW.extract(mIndexRef) << SCALAR_TYPE(x);
        }
        UME_FUNC_ATTRIB SCALAR_TYPE operator<< (IntermediateIndex const & x) const {
            return mVecRef_RW.extract(mIndexRef) <<
                x.mVecRef_RW.extract(x.mIndexRef);
        }
        template<typename T>
        UME_FUNC_ATTRIB SCALAR_TYPE operator>> (T const & x) const {
            return mVecRef_RW.extract(mIndexRef) >> SCALAR_TYPE(x);
        }
        UME_FUNC_ATTRIB SCALAR_TYPE operator>> (IntermediateIndex const & x) const {
            return mVecRef_RW.extract(mIndexRef) >>
                x.mVecRef_RW.extract(x.mIndexRef);
        }

#ifndef __NVCC__
// NVCC has a problem handling this friendship relation.
// Make it public only when compiling CUDA
    private:
        friend VEC_TYPE;
#endif
        // This object should be only constructible by the
        // vector type using it.
        IntermediateIndex() {}


        UME_FUNC_ATTRIB explicit IntermediateIndex(uint32_t index, VEC_TYPE & vec) : mIndexRef(index), mVecRef_RW(vec) {}

        VEC_TYPE & mVecRef_RW;
        uint32_t mIndexRef;
    };

    // **********************************************************************
    // *
    // *  Declaration of SIMDVecBaseInterface class 
    // *
    // *    This class should be used as a basic class for all integer and 
    // *    floating point vector types. All vectors should implement interface
    // *    contained in SIMDVecBaseInterface. If the derived class does not
    // *    provide an overload for given operation, this class will default 
    // *    to scalar emulation, thus providing interface coherence over
    // *    different plugins. This class should not be used directly in
    // *    plugins since it encapsulates only a common part of all vector
    // *    types. Plugins should use:
    // *     - "SIMDVecUnsignedInterface" for unsigned integer vectors,
    // *     - "SIMDVecSignedInterface" for signed integer vectors,
    // *     - "SIMDVecFloatInterface" for floating point vectors
    // *
    // **********************************************************************

    // DERIVED_VEC_TYPE - this is a derived class to be used as a part of 'Curiously Recurring Design Pattern (CRTP)'
    // SCALAR_TYPE - basic type of scalar elements packed in DERIVED_VEC_TYPE
    // VEC_LEN - number of SIMD elements in vector
    // MASK_TYPE - exact type of the mask to be used with this vector
    template<class DERIVED_VEC_TYPE, 
             typename SCALAR_TYPE, 
             uint32_t VEC_LEN,
             typename MASK_TYPE,
             typename SWIZZLE_MASK_TYPE>
    class SIMDVecBaseInterface
    {
        // Other vector types necessary for this class
        typedef SIMDVecBaseInterface< 
            DERIVED_VEC_TYPE, 
            SCALAR_TYPE, 
            VEC_LEN, 
            MASK_TYPE,
            SWIZZLE_MASK_TYPE> VEC_TYPE;

    protected:
        // Making destructor protected prohibits this class from being instantiated. Effectively this class can only be used as a base class.
        UME_FUNC_ATTRIB ~SIMDVecBaseInterface() {};
    public:
   
        // TODO: can be marked as constexpr?
        constexpr static uint32_t length() { return VEC_LEN; }

        constexpr static uint32_t alignment() { return VEC_LEN*sizeof(SCALAR_TYPE); }
        
        // ZERO-VEC
        static DERIVED_VEC_TYPE zero() { return DERIVED_VEC_TYPE(SCALAR_TYPE(0)); }

        // ONE-VEC
        static DERIVED_VEC_TYPE one() { return DERIVED_VEC_TYPE(SCALAR_TYPE(1)); }

#include "utilities/ignore_warnings_push.h"
#include "utilities/ignore_warnings_unused_parameter.h"

        // PREFETCH0
        static UME_FUNC_ATTRIB void prefetch0(SCALAR_TYPE const *p) {
            // DO NOTHING!
        }

        // PREFETCH1
        static UME_FUNC_ATTRIB void prefetch1(SCALAR_TYPE const *p) {
            // DO NOTHING!
        }

        // PREFETCH2
        static UME_FUNC_ATTRIB void prefetch2(SCALAR_TYPE const *p) {
            // DO NOTHING!
        }

#include "utilities/ignore_warnings_pop.h"

        // ASSIGNV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & assign (DERIVED_VEC_TYPE const & src) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::assign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), src);
        }
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator= (DERIVED_VEC_TYPE const & src) {
            return assign(src);
        }

        // MASSIGNV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & assign (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & src) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::assign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), src);
        }

        // ASSIGNS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & assign (SCALAR_TYPE value) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::assign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), value);
        }
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator= (SCALAR_TYPE value) {
            return assign(value);
        }

        // MASSIGNS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & assign (MASK_TYPE const & mask, SCALAR_TYPE value) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::assign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), value);
        }

        // LOAD
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & load (SCALAR_TYPE const *p) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::load<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }

        // MLOAD
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & load (MASK_TYPE const & mask, SCALAR_TYPE const * p) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::load<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }

        // LOADA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & loada (SCALAR_TYPE const * p) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::loadAligned<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }

        // MLOADA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & loada (MASK_TYPE const & mask, SCALAR_TYPE const *p) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::loadAligned<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }

        // SLOAD
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & sload(SCALAR_TYPE const *p) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::load<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }

        // MSLOAD
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & sload(MASK_TYPE const & mask, SCALAR_TYPE const *p) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::load<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), p);
        }

        // STORE
        UME_FUNC_ATTRIB SCALAR_TYPE* store (SCALAR_TYPE* p) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::store<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), p);
        }

        // MSTORE
        UME_FUNC_ATTRIB SCALAR_TYPE* store (MASK_TYPE const & mask, SCALAR_TYPE* p) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::store<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), p);
        }

        // STOREA
        UME_FUNC_ATTRIB SCALAR_TYPE* storea (SCALAR_TYPE* p) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::store<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), p);
        }

        // MSTOREA
        UME_FUNC_ATTRIB SCALAR_TYPE* storea (MASK_TYPE const & mask, SCALAR_TYPE* p) const {
            UME_EMULATION_WARNING();
           return SCALAR_EMULATION::store<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), p);
        }

        // SSTORE
        UME_FUNC_ATTRIB SCALAR_TYPE* sstore(SCALAR_TYPE *p) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::store<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), p);
        }

        // MSSTORE
        UME_FUNC_ATTRIB SCALAR_TYPE* sstore(MASK_TYPE const & mask, SCALAR_TYPE *p) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::store<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), p);
        }

        // EXTRACT
        // This method should be provided for all derived classes and cannot be defined
        // as generic.
        UME_FUNC_ATTRIB SCALAR_TYPE extract(uint32_t index) const;

        // INSERT
        // This method should be provided for all derived classes and cannot be defined
        // as generic.
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & insert(uint32_t index, SCALAR_TYPE value);

        // BLENDV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE blend (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::blend<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BLENDS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE blend (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::blend<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SWIZZLE
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE swizzle (SWIZZLE_MASK_TYPE const & sMask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::swizzle<DERIVED_VEC_TYPE, SWIZZLE_MASK_TYPE> (sMask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        template<int i0>
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE swizzle() const {
            UME_EMULATION_WARNING();
            static_assert(VEC_LEN == 1, "Invalid number of template parameters.");
            SWIZZLE_MASK_TYPE sMask(i0);
            return SCALAR_EMULATION::swizzle<DERIVED_VEC_TYPE, SWIZZLE_MASK_TYPE> (sMask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        template<int i0, int i1>
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE swizzle() const {
            UME_EMULATION_WARNING();
            static_assert(VEC_LEN == 2, "Invalid number of template parameters.");
            SWIZZLE_MASK_TYPE sMask(i0, i1);
            return SCALAR_EMULATION::swizzle<DERIVED_VEC_TYPE, SWIZZLE_MASK_TYPE> (sMask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        template<int i0, int i1, int i2, int i3>
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE swizzle() const {
            UME_EMULATION_WARNING();
            static_assert(VEC_LEN == 4, "Invalid number of template parameters.");
            SWIZZLE_MASK_TYPE sMask(i0, i1, i2, i3);
            return SCALAR_EMULATION::swizzle<DERIVED_VEC_TYPE, SWIZZLE_MASK_TYPE> (sMask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE swizzle() const {
            UME_EMULATION_WARNING();
            static_assert(VEC_LEN == 8, "Invalid number of template parameters.");
            SWIZZLE_MASK_TYPE sMask(i0, i1, i2, i3, i4, i5, i6, i7);
            return SCALAR_EMULATION::swizzle<DERIVED_VEC_TYPE, SWIZZLE_MASK_TYPE> (sMask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        template<
            int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, 
            int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15>
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE swizzle() const {
            UME_EMULATION_WARNING();
            static_assert(VEC_LEN == 16, "Invalid number of template parameters.");
            SWIZZLE_MASK_TYPE sMask(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15);
            return SCALAR_EMULATION::swizzle<DERIVED_VEC_TYPE, SWIZZLE_MASK_TYPE> (sMask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        template<
            int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, 
            int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15,
            int i16, int i17, int i18, int i19, int i20, int i21, int i22, int i23,
            int i24, int i25, int i26, int i27, int i28, int i29, int i30, int i31>
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE swizzle() const {
            UME_EMULATION_WARNING();
            static_assert(VEC_LEN == 31, "Invalid number of template parameters.");
            SWIZZLE_MASK_TYPE sMask(
                i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15,
                i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31);
            return SCALAR_EMULATION::swizzle<DERIVED_VEC_TYPE, SWIZZLE_MASK_TYPE> (sMask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        template<
            int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, 
            int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15,
            int i16, int i17, int i18, int i19, int i20, int i21, int i22, int i23,
            int i24, int i25, int i26, int i27, int i28, int i29, int i30, int i31,
            int i32, int i33, int i34, int i35, int i36, int i37, int i38, int i39,
            int i40, int i41, int i42, int i43, int i44, int i45, int i46, int i47,
            int i48, int i49, int i50, int i51, int i52, int i53, int i54, int i55,
            int i56, int i57, int i58, int i59, int i60, int i61, int i62, int i63>
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE swizzle() const {
            UME_EMULATION_WARNING();
            static_assert(VEC_LEN == 64, "Invalid number of template parameters.");
            SWIZZLE_MASK_TYPE sMask(
                i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15,
                i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31,
                i32, i33, i34, i35, i36, i37, i38, i39, i40, i41, i42, i43, i44, i45, i46, i47,
                i48, i49, i50, i51, i52, i53, i54, i55, i56, i57, i58, i59, i60, i61, i62, i63);
            return SCALAR_EMULATION::swizzle<DERIVED_VEC_TYPE, SWIZZLE_MASK_TYPE> (sMask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        template<
            int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, 
            int i8, int i9, int i10, int i11, int i12, int i13, int i14, int i15,
            int i16, int i17, int i18, int i19, int i20, int i21, int i22, int i23,
            int i24, int i25, int i26, int i27, int i28, int i29, int i30, int i31,
            int i32, int i33, int i34, int i35, int i36, int i37, int i38, int i39,
            int i40, int i41, int i42, int i43, int i44, int i45, int i46, int i47,
            int i48, int i49, int i50, int i51, int i52, int i53, int i54, int i55,
            int i56, int i57, int i58, int i59, int i60, int i61, int i62, int i63,
            int i64, int i65, int i66, int i67, int i68, int i69, int i70, int i71,
            int i72, int i73, int i74, int i75, int i76, int i77, int i78, int i79,
            int i80, int i81, int i82, int i83, int i84, int i85, int i86, int i87,
            int i88, int i89, int i90, int i91, int i92, int i93, int i94, int i95,
            int i96, int i97, int i98, int i99, int i100, int i101, int i102, int i103,
            int i104, int i105, int i106, int i107, int i108, int i109, int i110, int i111,
            int i112, int i113, int i114, int i115, int i116, int i117, int i118, int i119,
            int i120, int i121, int i122, int i123, int i124, int i125, int i126, int i127>
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE swizzle() const {
            UME_EMULATION_WARNING();
            static_assert(VEC_LEN == 128, "Invalid number of template parameters.");
            SWIZZLE_MASK_TYPE sMask(
                i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15,
                i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31,
                i32, i33, i34, i35, i36, i37, i38, i39, i40, i41, i42, i43, i44, i45, i46, i47,
                i48, i49, i50, i51, i52, i53, i54, i55, i56, i57, i58, i59, i60, i61, i62, i63,
                i64, i65, i66, i67, i68, i69, i70, i71, i72, i73, i74, i75, i76, i77, i78, i79,
                i80, i81, i82, i83, i84, i85, i86, i87, i88, i89, i90, i91, i92, i93, i94, i95,
                i96, i97, i98, i99, i100, i101, i102, i103, i104, i105, i106, i107, i108, i109, i110, i111,
                i112, i113, i114, i115, i116, i117, i118, i119, i120, i121, i122, i123, i124, i125, i126, i127);
            return SCALAR_EMULATION::swizzle<DERIVED_VEC_TYPE, SWIZZLE_MASK_TYPE> (sMask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SWIZZLEA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & swizzlea (SWIZZLE_MASK_TYPE const & sMask) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::swizzleAssign<DERIVED_VEC_TYPE, SWIZZLE_MASK_TYPE> (sMask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // SORTA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE sorta() {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::sortAscending<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SORTD
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE sortd() {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::sortDescending<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ADDV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE add (DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::add<DERIVED_VEC_TYPE> ( static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator+ (DERIVED_VEC_TYPE const & b) const {
            return add(b);
        }

        // MADDV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE add (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::add<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ADDS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE add (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::addScalar<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator+ (SCALAR_TYPE b) const {
            return add(b);
        }

        // MADDS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE add(MASK_TYPE const & mask, SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::addScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ADDVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & adda (DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::addAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator+= (DERIVED_VEC_TYPE const & b) {
            return adda(b);
        }

        // MADDVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & adda (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::addAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // ADDSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & adda (SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::addAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator+= (SCALAR_TYPE b) {
            return adda(b);
        }

        // MADDSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & adda (MASK_TYPE const & mask, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::addAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // SADDV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE sadd(DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::addSaturated<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        } 

        // MSADDV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE sadd(MASK_TYPE const & mask, DERIVED_VEC_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::addSaturated<DERIVED_VEC_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SADDS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE sadd(SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::addSaturatedScalar<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MSADDS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE sadd(MASK_TYPE const & mask, SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::addSaturatedScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SADDVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & sadda(DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::addSaturatedAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MSADDVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & sadda(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::addSaturatedAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // SADDSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & sadda(SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::addSaturatedScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MSADDSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & sadda(MASK_TYPE const & mask, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::addSaturatedScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // POSTINC
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE postinc () {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::postfixIncrement<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator++ (int) {
            return postinc();
        }

        // MPOSTINC
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE postinc (MASK_TYPE const & mask) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::postfixIncrement<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // PREFINC
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & prefinc () {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::prefixIncrement<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator++ () {
            return prefinc();
        }

        // MPREFINC
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & prefinc (MASK_TYPE const & mask) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::prefixIncrement<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // SUBV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE sub (DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::sub<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MSUBV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE sub (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::sub<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SUBS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE sub (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subScalar<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MSUBS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE sub (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SUBVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & suba (DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator-= (DERIVED_VEC_TYPE const & b) {
            return suba(b);
        }

        // MSUBVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & suba (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // SUBSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & suba (SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator-= (SCALAR_TYPE b) {
            return suba(b);
        }

        // MSUBSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & suba (MASK_TYPE const & mask, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // SSUBV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE ssub (DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subSaturated<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MSSUBV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE ssub (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subSaturated<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SSUBS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE ssub (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subSaturated<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MSSUBS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE ssub (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subSaturated<DERIVED_VEC_TYPE, SCALAR_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // SSUBVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & ssuba (DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subSaturatedAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MSSUBVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & ssuba (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subSaturatedAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // SSUBSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & ssuba (SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subSaturatedScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MSSUBSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & ssuba (MASK_TYPE const & mask, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subSaturatedScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // SUBFROMV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE subfrom (DERIVED_VEC_TYPE const & a) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subFrom<DERIVED_VEC_TYPE>(a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MSUBFROMV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE subfrom (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & a) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subFrom<DERIVED_VEC_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SUBFROMS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE subfrom (SCALAR_TYPE a) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subFromScalar<DERIVED_VEC_TYPE, SCALAR_TYPE>(a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MSUBFROMS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE subfrom (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subFromScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SUBFROMVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & subfroma (DERIVED_VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subFromAssign<DERIVED_VEC_TYPE>(a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MSUBFROMVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & subfroma (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subFromAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // SUBFROMSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & subfroma (SCALAR_TYPE a) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subFromScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MSUBFROMSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & subfroma (MASK_TYPE const & mask, SCALAR_TYPE a) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::subFromScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // POSTDEC
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE postdec () {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::postfixDecrement<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator-- (int) {
            return postdec();
        }

        // MPOSTDEC
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE postdec (MASK_TYPE const & mask) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::postfixDecrement<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // PREFDEC
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & prefdec() {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::prefixDecrement<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }
        
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator-- () {
            return prefdec();
        }

        // MPREFDEC
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & prefdec (MASK_TYPE const & mask) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::prefixDecrement<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MULV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE mul (DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::mult<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator* (DERIVED_VEC_TYPE const & b) const {
            return mul(b);
        }

        // MMULV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE mul (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::mult<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MULS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE mul (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::mult<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator* (SCALAR_TYPE b) const {
            return mul(b);
        }

        // MMULS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE mul (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::mult<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MULVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & mula (DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::multAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator*= (DERIVED_VEC_TYPE const & b) {
            return mula(b);
        }

        // MMULVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & mula (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::multAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MULSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & mula (SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::multAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator*= (SCALAR_TYPE b) {
            return mula(b);
        }

        // MMULSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & mula (MASK_TYPE const & mask, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::multAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // DIVV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE div (DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::div<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator/ (DERIVED_VEC_TYPE const & b) const {
            return div(b);
        }

        // MDIVV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE div (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::div<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // DIVS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE div (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::div<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator/ (SCALAR_TYPE b) const {
            return div(b);
        }

        // MDIVS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE div (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::div<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // DIVVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & diva (DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::divAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator/= (DERIVED_VEC_TYPE const & b) {
            return diva(b);
        }

        // MDIVVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & diva (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::divAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // DIVSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & diva (SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::divAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator/= (SCALAR_TYPE b) {
            return diva(b);
        }

        // MDIVSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & diva (MASK_TYPE const & mask, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::divAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // RCP
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rcp () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rcp<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MRCP
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rcp (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rcp<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // RCPS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rcp (SCALAR_TYPE a) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rcpScalar<DERIVED_VEC_TYPE, SCALAR_TYPE> (a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MRCPS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rcp (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rcpScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // RCPA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & rcpa () {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rcpAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MRCPA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & rcpa (MASK_TYPE const & mask) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rcpAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // RCPSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & rcpa (SCALAR_TYPE a) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rcpScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MRCPSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & rcpa (MASK_TYPE const & mask, SCALAR_TYPE a) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rcpScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, a, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // CMPEQV
        UME_FUNC_ATTRIB MASK_TYPE cmpeq (DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isEqual<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB MASK_TYPE operator== (DERIVED_VEC_TYPE const & b) const {
            return cmpeq(b);
        }

        // CMPEQS
        UME_FUNC_ATTRIB MASK_TYPE cmpeq (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isEqual<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB MASK_TYPE operator== (SCALAR_TYPE b) const {
            return cmpeq(b);
        }

        // CMPNEV
        UME_FUNC_ATTRIB MASK_TYPE cmpne (DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isNotEqual<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB MASK_TYPE operator!= (DERIVED_VEC_TYPE const & b) const {
            return cmpne(b);
        }

        // CMPNES
        UME_FUNC_ATTRIB MASK_TYPE cmpne (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isNotEqual<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB MASK_TYPE operator!= (SCALAR_TYPE b) const {
            return cmpne(b);
        }

        // CMPGTV
        UME_FUNC_ATTRIB MASK_TYPE cmpgt (DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isGreater<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB MASK_TYPE operator> (DERIVED_VEC_TYPE const & b) const {
            return cmpgt(b);
        }

        // CMPGTS
        UME_FUNC_ATTRIB MASK_TYPE cmpgt (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isGreater<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB MASK_TYPE operator> (SCALAR_TYPE b) const {
            return cmpgt(b);
        }

        // CMPLTV
        UME_FUNC_ATTRIB MASK_TYPE cmplt (DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isLesser<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB MASK_TYPE operator< (DERIVED_VEC_TYPE const & b) const {
            return cmplt(b);
        }

        // CMPLTS
        UME_FUNC_ATTRIB MASK_TYPE cmplt (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isLesser<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB MASK_TYPE operator< (SCALAR_TYPE b) const {
            return cmplt(b);
        }

        // CMPGEV
        UME_FUNC_ATTRIB MASK_TYPE cmpge (DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isGreaterEqual<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB MASK_TYPE operator>= (DERIVED_VEC_TYPE const & b) const {
            return cmpge(b);
        }

        // CMPGES
        UME_FUNC_ATTRIB MASK_TYPE cmpge (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isGreaterEqual<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB MASK_TYPE operator>= (SCALAR_TYPE b) const {
            return cmpge(b);
        }

        // CMPLEV
        UME_FUNC_ATTRIB MASK_TYPE cmple (DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isLesserEqual<MASK_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB MASK_TYPE operator<= (DERIVED_VEC_TYPE const & b) const {
            return cmple(b);
        }

        // CMPLES
        UME_FUNC_ATTRIB MASK_TYPE cmple (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isLesserEqual<MASK_TYPE, DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB MASK_TYPE operator<= (SCALAR_TYPE b) const {
            return cmple(b);
        }

        // CMPEV
        UME_FUNC_ATTRIB bool cmpe (DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isExact<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // CMPES
        UME_FUNC_ATTRIB bool cmpe (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::isExact<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), DERIVED_VEC_TYPE(b));
        }

        // UNIQUE
        UME_FUNC_ATTRIB bool unique() const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::unique<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HADD
        UME_FUNC_ATTRIB SCALAR_TYPE hadd () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceAdd<SCALAR_TYPE, DERIVED_VEC_TYPE>( static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHADD
        UME_FUNC_ATTRIB SCALAR_TYPE hadd (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceAdd<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &> (*this));
        }

        // HADDS
        UME_FUNC_ATTRIB SCALAR_TYPE hadd (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceAdd<SCALAR_TYPE, DERIVED_VEC_TYPE>(b, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHADDS
        UME_FUNC_ATTRIB SCALAR_TYPE hadd (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceAdd<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, b, static_cast<DERIVED_VEC_TYPE const &> (*this));
        }

        // HMUL
        UME_FUNC_ATTRIB SCALAR_TYPE hmul () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceMult<SCALAR_TYPE, DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHMUL
        UME_FUNC_ATTRIB SCALAR_TYPE hmul (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceMult<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HMULS
        UME_FUNC_ATTRIB SCALAR_TYPE hmul (SCALAR_TYPE a) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceMultScalar<SCALAR_TYPE, DERIVED_VEC_TYPE>(a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHMULS
        UME_FUNC_ATTRIB SCALAR_TYPE hmul (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceMultScalar<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ******************************************************************
        // * Fused arithmetics
        // ******************************************************************

        // FMULADDV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE fmuladd(DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::fmuladd<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // MFMULADDV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE fmuladd(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::fmuladd<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // FMULSUBV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE fmulsub(DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::fmulsub<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // MFMULSUBV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE fmulsub(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::fmulsub<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // FADDMULV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE faddmul(DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::faddmul<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // MFADDMULV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE faddmul(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::faddmul<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }
        
        // FSUBMULV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE fsubmul(DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::fsubmul<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // MFSUBMULV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE fsubmul(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b, DERIVED_VEC_TYPE const & c) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::fsubmul<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b, c);
        }

        // ******************************************************************
        // * Additional math functions
        // ******************************************************************

        // MAXV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE max (DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::max<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMAXV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE max (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::max<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MAXS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE max (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::maxScalar<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMAXS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE max (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::maxScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MAXVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & maxa (DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::maxAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MMAXVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & maxa (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::maxAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MAXSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & maxa (SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::maxScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MMAXSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & maxa (MASK_TYPE const & mask, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::maxScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MINV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE min (DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::min<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMINV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE min (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::min<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MINS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE min (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::minScalar<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MMINS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE min (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::minScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MINVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & mina (DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::minAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MMINVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & mina (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::minAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MINSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & mina (SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::minScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MMINSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & mina (MASK_TYPE const & mask, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::minScalarAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // HMAX
        UME_FUNC_ATTRIB SCALAR_TYPE hmax () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::reduceMax<SCALAR_TYPE, DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHMAX
        UME_FUNC_ATTRIB SCALAR_TYPE hmax (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::reduceMax<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HMAXS
        UME_FUNC_ATTRIB SCALAR_TYPE hmax (SCALAR_TYPE a) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::reduceMax<SCALAR_TYPE, DERIVED_VEC_TYPE>(a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHMAXS
        UME_FUNC_ATTRIB SCALAR_TYPE hmax (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::reduceMax<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // IMAX
        UME_FUNC_ATTRIB uint32_t imax() const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::indexMax<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MIMAX
        UME_FUNC_ATTRIB uint32_t imax(MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::indexMax<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HMIN
        UME_FUNC_ATTRIB SCALAR_TYPE hmin() const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::reduceMin<SCALAR_TYPE, DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHMIN
        UME_FUNC_ATTRIB SCALAR_TYPE hmin(MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::reduceMin<SCALAR_TYPE, DERIVED_VEC_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // IMIN
        UME_FUNC_ATTRIB uint32_t imin() const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::indexMin<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MIMIN
        UME_FUNC_ATTRIB uint32_t imin(MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::indexMin<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
    };

    // ***************************************************************************
    // *
    // *    Definition of Bitwise Interface. Bitwise operations can only be
    // *    performed on integer (signed and unsigned) data types in C++.
    // *    While making bitwise operations on floating points is sometimes
    // *    necessary, it is not safe and not portable.
    // *
    // ***************************************************************************
    template<typename DERIVED_VEC_TYPE,
             typename SCALAR_TYPE,
             typename MASK_TYPE>
    class SIMDVecBitwiseInterface {
        
        typedef SIMDVecBitwiseInterface< 
            DERIVED_VEC_TYPE, 
            SCALAR_TYPE,
            MASK_TYPE> VEC_TYPE;
 
    public:
        // BANDV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE band (DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAnd<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator& (DERIVED_VEC_TYPE const & b) const {
            return band(b);
        }

        // MBANDV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE band (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAnd<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BANDS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE band (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAnd<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator& (SCALAR_TYPE b) const {
            return band(b);
        }

        // MBANDS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE band (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAnd<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BANDVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & banda (DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAndAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator&= (DERIVED_VEC_TYPE const & b) {
            return banda(b);
        }

        // MBANDVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & banda (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAndAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        // BANDSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & banda (SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAndAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator&= (bool b) {
            return banda(b);
        }

        // MBANDSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & banda (MASK_TYPE const & mask, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAndAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // BORV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE bor ( DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryOr<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator| ( DERIVED_VEC_TYPE const & b) const {
            return bor(b);
        }

        // MBORV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE bor ( MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryOr<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BORS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE bor (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryOr<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator| (SCALAR_TYPE b) const {
            return bor(b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator|| (SCALAR_TYPE b) const {
            return bor(b);
        }

        // MBORS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE bor (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryOr<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BORVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & bora (DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryOrAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator|= (DERIVED_VEC_TYPE const & b) {
            return bora(b);
        }

        // MBORVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & bora (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryOrAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // BORSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & bora (SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryOrAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator|= (SCALAR_TYPE b) {
            return bora(b);
        }

        // MBORSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & bora (MASK_TYPE const & mask, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryOrAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // BXORV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE bxor (DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryXor<DERIVED_VEC_TYPE> ( static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator^ (DERIVED_VEC_TYPE const & b) const {
            return bxor(b);
        }

        // MBXORV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE bxor (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryXor<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BXORS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE bxor (SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryXor<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator^ (SCALAR_TYPE b) const {
            return bxor(b);
        }

        // MBXORS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE bxor (MASK_TYPE const & mask, SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryXor<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // BXORVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & bxora (DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryXorAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator^= (DERIVED_VEC_TYPE const & b) {
            return bxora(b);
        }

        // MBXORVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & bxora (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryXorAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // BXORSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & bxora (SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryXorAssign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & operator^= (SCALAR_TYPE b) {
            return bxora(b);
        }

        // MBXORSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & bxora (MASK_TYPE const & mask, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryXorAssign<DERIVED_VEC_TYPE,SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // BNOT
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE bnot () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryNot<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
    
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator~ () const {
            return bnot();
        }

        // MBNOT
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE bnot (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryNot<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // BNOTA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & bnota () {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryNotAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // BANDNOTV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE bandnot(DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAndNot<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        // MBANDNOTV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE bandnot(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAndNot<DERIVED_VEC_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        // BANDNOTS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE bandnot(SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAndNot<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        // MBANDNOTS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE bandnot(MASK_TYPE const & mask, SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAndNot<DERIVED_VEC_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        // BANDNOTVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & bandnota(DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAndNotAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        // MBANDNOTVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & bandnota(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAndNotAssign<DERIVED_VEC_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        // BANDNOTSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & bandnota(SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAndNotAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        // MBANDNOTSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & bandnota(MASK_TYPE const & mask, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryAndNotAssign<DERIVED_VEC_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MBNOTA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & bnota (MASK_TYPE const & mask) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::binaryNotAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // HBAND
        UME_FUNC_ATTRIB SCALAR_TYPE hband ()const  {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceBinaryAnd<SCALAR_TYPE, DERIVED_VEC_TYPE>( static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBAND
        UME_FUNC_ATTRIB SCALAR_TYPE hband (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceBinaryAnd<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HBANDS
        UME_FUNC_ATTRIB SCALAR_TYPE hband (SCALAR_TYPE a) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceBinaryAndScalar<SCALAR_TYPE, DERIVED_VEC_TYPE>(a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBANDS
        UME_FUNC_ATTRIB SCALAR_TYPE hband (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceBinaryAndScalar<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE>(mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HBOR
        UME_FUNC_ATTRIB SCALAR_TYPE hbor () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceBinaryOr<SCALAR_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBOR
        UME_FUNC_ATTRIB SCALAR_TYPE hbor (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceBinaryOr<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HBORS
        UME_FUNC_ATTRIB SCALAR_TYPE hbor (SCALAR_TYPE a) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceBinaryOrScalar<SCALAR_TYPE, DERIVED_VEC_TYPE> (a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBORS
        UME_FUNC_ATTRIB SCALAR_TYPE hbor (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceBinaryOrScalar<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // HBXOR
        UME_FUNC_ATTRIB SCALAR_TYPE hbxor () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceBinaryXor<SCALAR_TYPE, DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBXOR
        UME_FUNC_ATTRIB SCALAR_TYPE hbxor (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceBinaryXor<SCALAR_TYPE, DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // HBXORS
        UME_FUNC_ATTRIB SCALAR_TYPE hbxor (SCALAR_TYPE a) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceBinaryXorScalar<SCALAR_TYPE, DERIVED_VEC_TYPE> (a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MHBXORS
        UME_FUNC_ATTRIB SCALAR_TYPE hbxor (MASK_TYPE const & mask, SCALAR_TYPE a) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reduceBinaryXorScalar<SCALAR_TYPE, DERIVED_VEC_TYPE> (mask, a, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
    };


    // ***************************************************************************
    // *
    // *    Definition of Integer Interface. Integer operations can only be
    // *    performed on integer (signed and unsigned) data types in C++.
    // *    While making certain operations (such as bitwise) on floating points is sometimes
    // *    necessary, it is not safe and not portable. 
    // *
    // ***************************************************************************
    template<typename DERIVED_VEC_TYPE,
             typename SCALAR_TYPE,
             typename MASK_TYPE>
    class SIMDVecIntegerInterface : 
        public SIMDVecBitwiseInterface<
            DERIVED_VEC_TYPE,
            SCALAR_TYPE,
            MASK_TYPE>
    {
        typedef SIMDVecIntegerInterface<
            DERIVED_VEC_TYPE, 
            SCALAR_TYPE,
            MASK_TYPE> VEC_TYPE;
 
    public:
        // REMV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rem(DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reminder<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator% (DERIVED_VEC_TYPE const & b) const {
            return rem(b);
        }

        // MREMV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rem(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reminder<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // REMS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rem(SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reminder<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator% (SCALAR_TYPE b) const {
            return rem(b);
        }

        // MREMS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rem(MASK_TYPE const & mask, SCALAR_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reminder<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // REMVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rema(DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reminderAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator%= (DERIVED_VEC_TYPE const & b) {
            return rema(b);
        }

        // MREMVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rema(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reminderAssign<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // REMSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rema(SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reminderAssign<DERIVED_VEC_TYPE, SCALAR_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator%= (SCALAR_TYPE b) {
            return rema(b);
        }
        // MREMSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rema(MASK_TYPE const & mask, SCALAR_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::reminderAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // LANDV
        UME_FUNC_ATTRIB MASK_TYPE land(DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            // C++ standard says:
            // "A prvalue of arithmetic, unscoped enumeration, pointer, or pointer to member type can be converted to a
            //    prvalue of type bool.A zero value, null pointer value, or null member pointer value is converted to false;
            //    any other value is converted to true.A prvalue of type std::nullptr_t can be converted to a prvalue of
            //    type bool; the resulting value is false."
            MASK_TYPE t0 = SCALAR_EMULATION::isNotEqual<MASK_TYPE, DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), SCALAR_TYPE(0));
            MASK_TYPE t1 = SCALAR_EMULATION::isNotEqual<MASK_TYPE, DERIVED_VEC_TYPE>(b, SCALAR_TYPE(0));
            MASK_TYPE t2 = SCALAR_EMULATION::binaryAnd<MASK_TYPE>(t0, t1);
            return t2;
        }
        UME_FUNC_ATTRIB MASK_TYPE operator&& (DERIVED_VEC_TYPE const & b) const {
            return land(b);
        }

        // LANDS
        UME_FUNC_ATTRIB MASK_TYPE land(bool b) const {
            UME_EMULATION_WARNING();
            // LAND with scalar operators can simply use booleans. C++ standard should take
            // care of boolean conversions from any other expressions.
            MASK_TYPE t0 = SCALAR_EMULATION::isNotEqual<MASK_TYPE, DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), SCALAR_TYPE(0));
            MASK_TYPE t1 = SCALAR_EMULATION::binaryAnd<MASK_TYPE>(t0, b);
            return t1;
        }
        UME_FUNC_ATTRIB MASK_TYPE operator&& (bool b) const {
            return land(b);
        }

        // LORV
        UME_FUNC_ATTRIB MASK_TYPE lor(DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            // C++ standard says:
            // "A prvalue of arithmetic, unscoped enumeration, pointer, or pointer to member type can be converted to a
            //    prvalue of type bool.A zero value, null pointer value, or null member pointer value is converted to false;
            //    any other value is converted to true.A prvalue of type std::nullptr_t can be converted to a prvalue of
            //    type bool; the resulting value is false."
            MASK_TYPE t0 = SCALAR_EMULATION::isNotEqual<MASK_TYPE, DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), SCALAR_TYPE(0));
            MASK_TYPE t1 = SCALAR_EMULATION::isNotEqual<MASK_TYPE, DERIVED_VEC_TYPE>(b, SCALAR_TYPE(0));
            MASK_TYPE t2 = SCALAR_EMULATION::binaryOr<MASK_TYPE>(t0, t1);
            return t2;
        }
        UME_FUNC_ATTRIB MASK_TYPE operator|| (DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return lor(b);
        }

        // LORS
        UME_FUNC_ATTRIB MASK_TYPE lor(bool b) const {
            UME_EMULATION_WARNING();
            // LAND with scalar operators can simply use booleans. C++ standard should take
            // care of boolean conversions from any other expressions.
            MASK_TYPE t0 = SCALAR_EMULATION::isNotEqual<MASK_TYPE, DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), SCALAR_TYPE(0));
            MASK_TYPE t1 = SCALAR_EMULATION::binaryOr<MASK_TYPE>(t0, b);
            return t1;
        }
        UME_FUNC_ATTRIB MASK_TYPE operator|| (bool b) const {
            return lor(b);
        }
    };

    // ***************************************************************************
    // *
    // *    Definition of Gather/Scatter interface. This interface creates
    // *    an abstraction for gather and scatter operations. It needs to be
    // *    separate from base interface, because it is aware of unsigned
    // *    types (used for indexing).
    // *
    // ***************************************************************************
    template<typename DERIVED_VEC_TYPE,
             typename DERIVED_UINT_VEC_TYPE,
             typename SCALAR_TYPE,
             typename SCALAR_UINT_TYPE,
             typename MASK_TYPE>
    class SIMDVecGatherScatterInterface
    {
        typedef SIMDVecGatherScatterInterface< 
            DERIVED_VEC_TYPE, 
            DERIVED_UINT_VEC_TYPE,
            SCALAR_TYPE,
            SCALAR_UINT_TYPE,
            MASK_TYPE> VEC_TYPE;
 
    public:
        // GATHERU
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & gatheru (SCALAR_TYPE const * baseAddr, uint32_t stride) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::gatheru<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, stride);
        }

        // MGATHERU
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & gatheru (MASK_TYPE const & mask, SCALAR_TYPE const * baseAddr, uint32_t stride) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::gatheru<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, stride);
        }

        // GATHERS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & gather (SCALAR_TYPE const * baseAddr, SCALAR_UINT_TYPE const * indices) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::gather<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }

        // MGATHERS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & gather (MASK_TYPE const & mask, SCALAR_TYPE const * baseAddr, SCALAR_UINT_TYPE const * indices) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::gather<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }

        // GATHERV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & gather (SCALAR_TYPE const * baseAddr, DERIVED_UINT_VEC_TYPE const & indices) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::gather<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }

        // MGATHERV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & gather (MASK_TYPE const & mask, SCALAR_TYPE const * baseAddr, DERIVED_UINT_VEC_TYPE const & indices) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::gather<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }

        // SCATTERU
        UME_FUNC_ATTRIB SCALAR_TYPE* scatteru (SCALAR_TYPE * baseAddr, uint32_t stride) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::scatteru<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, stride);
        }

        // MSCATTERU
        UME_FUNC_ATTRIB SCALAR_TYPE* scatteru (MASK_TYPE const & mask, SCALAR_TYPE * baseAddr, uint32_t stride) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::scatteru<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, stride);
        }

        // SCATTERS
        UME_FUNC_ATTRIB SCALAR_TYPE* scatter (SCALAR_TYPE* baseAddr, SCALAR_UINT_TYPE* indices) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::scatter<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }

        // MSCATTERS
        UME_FUNC_ATTRIB SCALAR_TYPE*  scatter (MASK_TYPE const & mask, SCALAR_TYPE* baseAddr, SCALAR_UINT_TYPE* indices) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::scatter<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }

        // SCATTERV
        UME_FUNC_ATTRIB SCALAR_TYPE*  scatter (SCALAR_TYPE* baseAddr, DERIVED_UINT_VEC_TYPE const & indices) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::scatter<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }

        // MSCATTERV
        UME_FUNC_ATTRIB SCALAR_TYPE*  scatter (MASK_TYPE const & mask, SCALAR_TYPE* baseAddr, DERIVED_UINT_VEC_TYPE const & indices) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::scatter<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), baseAddr, indices);
        }       
    };
    
    // ***************************************************************************
    // *
    // *    Definition of Shift/Rotate interface. This interface creates
    // *    an abstraction for bitwise shift and rotation operations. These
    // *    operations should only be used on signed and unsigned integer
    // *    vector types.
    // *
    // ***************************************************************************
    template<typename DERIVED_VEC_TYPE,
             typename DERIVED_UINT_VEC_TYPE,
             typename SCALAR_TYPE,
             typename SCALAR_UINT_TYPE,
             typename MASK_TYPE>
    class SIMDVecShiftRotateInterface
    {
        typedef SIMDVecShiftRotateInterface< 
            DERIVED_VEC_TYPE, 
            DERIVED_UINT_VEC_TYPE,
            SCALAR_TYPE,
            SCALAR_UINT_TYPE,
            MASK_TYPE> VEC_TYPE;
 
    public:
        // LSHV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE lsh (DERIVED_UINT_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::shiftBitsLeft<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator<< (DERIVED_UINT_VEC_TYPE const & b) const {
            return lsh(b);
        }

        // MLSHV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE lsh (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::shiftBitsLeft<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // LSHS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE lsh (SCALAR_UINT_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::shiftBitsLeftScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator<< (SCALAR_UINT_TYPE b) const {
            return lsh(b);
        }

        // MLSHS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE lsh (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::shiftBitsLeftScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // LSHVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & lsha (DERIVED_UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::shiftBitsLeftAssign<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator<<= (DERIVED_UINT_VEC_TYPE const & b) {
            return lsha(b);
        }

        // MLSHVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & lsha (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::shiftBitsLeftAssign<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // LSHSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & lsha (SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::shiftBitsLeftAssignScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator<<= (SCALAR_UINT_TYPE b) {
            return lsha(b);
        }

        // MLSHSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & lsha (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::shiftBitsLeftAssignScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // RSHV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rsh (DERIVED_UINT_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::shiftBitsRight<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator>> (DERIVED_UINT_VEC_TYPE const & b) const {
            return rsh(b);
        }

        // MRSHV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rsh (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::shiftBitsRight<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // RSHS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rsh (SCALAR_UINT_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::shiftBitsRightScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator>> (SCALAR_UINT_TYPE b) const {
            return rsh(b);
        }

        // MRSHS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rsh (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::shiftBitsRightScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // RSHVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & rsha (DERIVED_UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::shiftBitsRightAssign<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator>>= (DERIVED_UINT_VEC_TYPE const & b) {
            return rsha(b);
        }

        // MRSHVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & rsha (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::shiftBitsRightAssign<DERIVED_VEC_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // RSHSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & rsha (SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::shiftBitsRightAssignScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator>>= (SCALAR_UINT_TYPE b) {
            return rsha(b);
        }

        // MRSHSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & rsha (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::shiftBitsRightAssignScalar<DERIVED_VEC_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // ROLV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rol (DERIVED_UINT_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rotateBitsLeft<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, SCALAR_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MROLV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rol (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rotateBitsLeft<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ROLS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rol (SCALAR_UINT_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rotateBitsLeftScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MROLS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rol (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rotateBitsLeftScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // ROLVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & rola (DERIVED_UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rotateBitsLeftAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MROLVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & rola (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rotateBitsLeftAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // ROLSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & rola (SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rotateBitsLeftAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MROLSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & rola (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rotateBitsLeftAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // RORV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE ror (DERIVED_UINT_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rotateBitsRight<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MRORV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE ror (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rotateBitsRight<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // RORS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE ror (SCALAR_UINT_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rotateBitsRightScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // MRORS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE ror (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rotateBitsRightScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

        // RORVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rora (DERIVED_UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rotateBitsRightAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MRORVA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rora (MASK_TYPE const & mask, DERIVED_UINT_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rotateBitsRightAssign<DERIVED_VEC_TYPE, SCALAR_TYPE, DERIVED_UINT_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // RORSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rora (SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rotateBitsRightAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }

        // MRORSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rora (MASK_TYPE const & mask, SCALAR_UINT_TYPE b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::rotateBitsRightAssignScalar<DERIVED_VEC_TYPE, SCALAR_TYPE, SCALAR_UINT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this), b);
        }
    };

    // ***************************************************************************
    // *
    // *    Definition of Packable Interface. Pack operations can only be 
    // *    performed on SIMD vector with lengths higher than 1 and being
    // *    powers of 2. Vectors of such lengths have to derive from one of type
    // *    interfaces: signed, unsigned or float and from packable interface.
    // *    SIMD vectors of length 1 should only use type interface.
    // *
    // ***************************************************************************
    template<class DERIVED_VEC_TYPE,
    class DERIVED_HALF_VEC_TYPE>
    class SIMDVecPackableInterface
    {
        // Other vector types necessary for this class
        typedef SIMDVecPackableInterface<
            DERIVED_VEC_TYPE,
            DERIVED_HALF_VEC_TYPE> VEC_TYPE;

    public:

        // PACK
        DERIVED_VEC_TYPE & pack(DERIVED_HALF_VEC_TYPE const & a, DERIVED_HALF_VEC_TYPE const & b) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::pack<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE>(
                static_cast<DERIVED_VEC_TYPE &>(*this),
                static_cast<DERIVED_HALF_VEC_TYPE const &>(a),
                static_cast<DERIVED_HALF_VEC_TYPE const &>(b)
                );
        }

        // PACKLO
        DERIVED_VEC_TYPE & packlo(DERIVED_HALF_VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::packLow<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE>(
                static_cast<DERIVED_VEC_TYPE &>(*this),
                static_cast<DERIVED_HALF_VEC_TYPE const &>(a)
                );
        }

        // PACKHI
        DERIVED_VEC_TYPE & packhi(DERIVED_HALF_VEC_TYPE const & a) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::packHigh<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE>(
                static_cast<DERIVED_VEC_TYPE &>(*this),
                static_cast<DERIVED_HALF_VEC_TYPE const &>(a)
                );
        }

        // UNPACK
        void unpack(DERIVED_HALF_VEC_TYPE & a, DERIVED_HALF_VEC_TYPE & b) const {
            UME_EMULATION_WARNING();
            SCALAR_EMULATION::unpack<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE>(
                static_cast<DERIVED_VEC_TYPE const &>(*this),
                static_cast<DERIVED_HALF_VEC_TYPE &>(a),
                static_cast<DERIVED_HALF_VEC_TYPE &>(b)
                );
        }

        // UNPACKLO
        DERIVED_HALF_VEC_TYPE unpacklo() const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::unpackLow<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE>(
                static_cast<DERIVED_VEC_TYPE const &> (*this)
                );
        }

        // UNPACKHI
        DERIVED_HALF_VEC_TYPE unpackhi() const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::unpackHigh<DERIVED_VEC_TYPE, DERIVED_HALF_VEC_TYPE>(
                static_cast<DERIVED_VEC_TYPE const &> (*this)
                );
        }
    };

    // ***************************************************************************
    // *
    // *    Definition of Sign interface. This interface creates
    // *    an abstraction for operations that are aware of scalar types sign.
    // *    this interface should be reserved for signed integer and floating
    // *    point vector types.
    // *
    // ***************************************************************************
    template<typename DERIVED_VEC_TYPE, typename SCALAR_TYPE, typename MASK_TYPE>
    class SIMDVecSignInterface
    {        
        // Other vector types necessary for this class
        typedef SIMDVecSignInterface< 
            DERIVED_VEC_TYPE,
            SCALAR_TYPE,
            MASK_TYPE> VEC_TYPE;
 
    public:

        // NEG
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE neg () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::unaryMinus<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MNEG
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE neg (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::unaryMinus<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // NEGA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & nega() {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::unaryMinusAssign<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MNEGA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & nega(MASK_TYPE const & mask) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::unaryMinusAssign<DERIVED_VEC_TYPE>(mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // ABS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE abs () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::abs<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MABS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE abs (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::abs<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ABSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE absa () {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::absAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MABSA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE absa (MASK_TYPE const & mask) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::absAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // COPYSIGN
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE copysign(DERIVED_VEC_TYPE const & sign) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::copySign<DERIVED_VEC_TYPE, SCALAR_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), sign);
        }

        // MCOPYSIGN
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE copysign(MASK_TYPE const & mask, DERIVED_VEC_TYPE const & sign) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::copySign<DERIVED_VEC_TYPE, SCALAR_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), sign);
        }
    };
    
    // ***************************************************************************
    // *
    // *    Definition of interface for vectors using UNSIGNED INTEGER scalar types
    // *
    // ***************************************************************************
    template<typename DERIVED_UINT_VEC_TYPE,
             typename SCALAR_UINT_TYPE, 
             uint32_t VEC_LEN,
             typename MASK_TYPE,
             typename SWIZZLE_MASK_TYPE> 
    class SIMDVecUnsignedInterface : 
        public SIMDVecBaseInterface< 
            DERIVED_UINT_VEC_TYPE,
            SCALAR_UINT_TYPE, 
            VEC_LEN,
            MASK_TYPE,
            SWIZZLE_MASK_TYPE>,
        public SIMDVecIntegerInterface<
            DERIVED_UINT_VEC_TYPE,
            SCALAR_UINT_TYPE,
            MASK_TYPE>,
        public SIMDVecGatherScatterInterface<
            DERIVED_UINT_VEC_TYPE,   // DERIVED_VEC_TYPE
            DERIVED_UINT_VEC_TYPE,
            SCALAR_UINT_TYPE,
            SCALAR_UINT_TYPE,
            MASK_TYPE>,
        public SIMDVecShiftRotateInterface<
            DERIVED_UINT_VEC_TYPE,   // DERIVED_VEC_TYPE
            DERIVED_UINT_VEC_TYPE,
            SCALAR_UINT_TYPE,        // SCALAR_TYPE
            SCALAR_UINT_TYPE,
            MASK_TYPE>
    {
        // Other vector types necessary for this class
        typedef SIMDVecUnsignedInterface< 
            DERIVED_UINT_VEC_TYPE, 
            SCALAR_UINT_TYPE,
            VEC_LEN, 
            MASK_TYPE,
            SWIZZLE_MASK_TYPE> VEC_TYPE;
    private:

        // Forbid assignment-initialization of vector using scalar values
 
        //SCALAR_UINT_TYPE operator[] (SCALAR_UINT_TYPE index) const; // Declaration only! This operator has to be implemented in derived class.
        UME_FUNC_ATTRIB DERIVED_UINT_VEC_TYPE & insert(uint32_t index, SCALAR_UINT_TYPE value); // Declaration only! This operator has to be implemented in derived class.

    protected:
            
        // Making destructor protected prohibits this class from being instantiated. Effectively this class can only be used as a base class.
        ~SIMDVecUnsignedInterface() {};
    public:
        // SUBV
        UME_FUNC_ATTRIB DERIVED_UINT_VEC_TYPE operator- (DERIVED_UINT_VEC_TYPE const & b) const {
            return this->sub(b);
        }
        // SUBS
        UME_FUNC_ATTRIB DERIVED_UINT_VEC_TYPE operator- (SCALAR_UINT_TYPE b) const {
            return this->sub(b);
        }
    };

    // ***************************************************************************
    // *
    // *    Definition of interface for vectors using SIGNED INTEGER scalar types
    // *
    // ***************************************************************************
    template<typename DERIVED_VEC_TYPE,
             typename DERIVED_VEC_UINT_TYPE,
             typename SCALAR_TYPE, 
             uint32_t VEC_LEN,
             typename SCALAR_UINT_TYPE,
             typename MASK_TYPE,
             typename SWIZZLE_MASK_TYPE>
    class SIMDVecSignedInterface : 
        public SIMDVecBaseInterface< 
            DERIVED_VEC_TYPE,
            SCALAR_TYPE, 
            VEC_LEN,
            MASK_TYPE,
            SWIZZLE_MASK_TYPE>,
        public SIMDVecIntegerInterface<
            DERIVED_VEC_TYPE,
            SCALAR_TYPE,
            MASK_TYPE>,
        public SIMDVecGatherScatterInterface<
            DERIVED_VEC_TYPE,   // DERIVED_VEC_TYPE
            DERIVED_VEC_UINT_TYPE,   // DERIVEC_UINT_VEC_TYPE // TODO: replace this with DERIVED_VEC_TYPE when other types independant!
            SCALAR_TYPE,
            SCALAR_UINT_TYPE,
            MASK_TYPE>,
        public SIMDVecShiftRotateInterface<
            DERIVED_VEC_TYPE,
            DERIVED_VEC_UINT_TYPE,
            SCALAR_TYPE,
            SCALAR_UINT_TYPE,
            MASK_TYPE>,
        public SIMDVecSignInterface<
            DERIVED_VEC_TYPE,
            SCALAR_TYPE,
            MASK_TYPE>
    {
        // Other vector types necessary for this class
        typedef SIMDVecSignedInterface< DERIVED_VEC_TYPE,
                             DERIVED_VEC_UINT_TYPE,
                             SCALAR_TYPE,
                             VEC_LEN, 
                             SCALAR_UINT_TYPE,
                             MASK_TYPE,
                             SWIZZLE_MASK_TYPE> VEC_TYPE;

        SCALAR_TYPE operator[] (SCALAR_UINT_TYPE index) const; // Declaration only! This operator has to be implemented in derived class.
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & insert (uint32_t index, SCALAR_TYPE value); // Declaration only! This operator has to be implemented in derived class.
    protected:
            
        // Making destructor protected prohibits this class from being instantiated. Effectively this class can only be used as a base class.
        ~SIMDVecSignedInterface() {};
    public:
        // Everything already handled by other interface classes

        // SUBV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator- (DERIVED_VEC_TYPE const & b) const {
            return this->sub(b);
        }

        // SUBS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator- (SCALAR_TYPE const & b) const {
            return this->sub(b);
        }

        // NEG
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator- () const {
            return this->neg();
        }
    };

    // ***************************************************************************
    // *
    // *    Definition of interface for vectors using FLOATING POINT scalar types
    // *
    // ***************************************************************************
    template<typename DERIVED_VEC_TYPE,
             typename DERIVED_VEC_UINT_TYPE,
             typename DERIVED_VEC_INT_TYPE, // corresponding integer type
             typename SCALAR_FLOAT_TYPE, 
             uint32_t VEC_LEN,
             typename SCALAR_UINT_TYPE,
             typename SCALAR_INT_TYPE,
             typename MASK_TYPE,
             typename SWIZZLE_MASK_TYPE>
    class SIMDVecFloatInterface :  
        public SIMDVecBaseInterface< 
            DERIVED_VEC_TYPE,
            SCALAR_FLOAT_TYPE, 
            VEC_LEN,
            MASK_TYPE,
            SWIZZLE_MASK_TYPE>,
        public SIMDVecGatherScatterInterface<
            DERIVED_VEC_TYPE,   // DERIVED_VEC_TYPE
            DERIVED_VEC_UINT_TYPE,   // DERIVEC_UINT_VEC_TYPE // TODO: replace this with DERIVED_VEC_TYPE when other types independant!
            SCALAR_FLOAT_TYPE,
            SCALAR_UINT_TYPE,
            MASK_TYPE>,
        public SIMDVecSignInterface<
            DERIVED_VEC_TYPE,
            SCALAR_FLOAT_TYPE,
            MASK_TYPE>
    {
        // Other vector types necessary for this class
        typedef SIMDVecFloatInterface< DERIVED_VEC_TYPE,
                    DERIVED_VEC_UINT_TYPE,
                    DERIVED_VEC_INT_TYPE,
                    SCALAR_FLOAT_TYPE,
                    VEC_LEN, 
                    SCALAR_UINT_TYPE,
                    SCALAR_INT_TYPE,
                    MASK_TYPE,
                    SWIZZLE_MASK_TYPE> VEC_TYPE;

    protected:
        // Making destructor protected prohibits this class from being instantiated. Effectively this class can only be used as a base class.
        UME_FUNC_ATTRIB ~SIMDVecFloatInterface() {};
        
        SCALAR_FLOAT_TYPE operator[] (SCALAR_UINT_TYPE index) const; // Declaration only! This operator has to be implemented in derived class.
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & insert(uint32_t index, SCALAR_FLOAT_TYPE value); // Declaration only! This operator has to be implemented in derived class.
    public:

        // SUBV
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator- (DERIVED_VEC_TYPE const & b) const {
            return this->sub(b);
        }
        
        // SUBS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator- (SCALAR_FLOAT_TYPE b) const {
            return this->sub(b);
        }
        // NEG
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE operator- () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::unaryMinus<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ********************************************************************
        // * MATH FUNCTIONS
        // ********************************************************************

        // SQR
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE sqr () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::sqr<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MSQR
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE sqr (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::sqr<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SQRA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & sqra () {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::sqrAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MSQRA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & sqra (MASK_TYPE const & mask) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::sqrAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // SQRT
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE sqrt () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::sqrt<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // MSQRT
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE sqrt (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::sqrt<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // SQRTA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & sqrta () {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::sqrtAssign<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }
        
        // MSQRTA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & sqrta (MASK_TYPE const & mask) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::sqrtAssign<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // RSQRT
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rsqrt () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::rsqrt<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MRSQRT
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE rsqrt (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::rsqrt<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SQRTA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & rsqrta () {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::rsqrtAssign<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE> (static_cast<DERIVED_VEC_TYPE &>(*this));
        }

        // MSQRTA
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE & rsqrta (MASK_TYPE const & mask) {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::rsqrtAssign<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE &>(*this));
        }
        
        // POWV
        // Disabled, see Issue #10
        //UME_FUNC_ATTRIB DERIVED_VEC_TYPE pow (DERIVED_VEC_TYPE const & b) const {
        //    return SCALAR_EMULATION::MATH::pow<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        // }

        // MPOWV    
        // Disabled, see Issue #10    
        //UME_FUNC_ATTRIB DERIVED_VEC_TYPE pow (MASK_TYPE const & mask, DERIVED_VEC_TYPE const & b) const {
        //    return SCALAR_EMULATION::MATH::pow<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        //}

        // POWS
        // Disabled, see Issue #10
        //UME_FUNC_ATTRIB DERIVED_VEC_TYPE pow (SCALAR_FLOAT_TYPE b) const {
        //    return SCALAR_EMULATION::MATH::pows<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        //}

        // MPOWS
        // Disabled, see Issue #10
        //UME_FUNC_ATTRIB DERIVED_VEC_TYPE pow (MASK_TYPE const & mask, SCALAR_FLOAT_TYPE b) const {
        //    return SCALAR_EMULATION::MATH::pows<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        //}

        // ROUND
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE round () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::round<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // MROUND
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE round (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::round<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }
        
        // TRUNC
        UME_FUNC_ATTRIB DERIVED_VEC_INT_TYPE trunc () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::truncToInt<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE, DERIVED_VEC_INT_TYPE, SCALAR_INT_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MTRUNC
        UME_FUNC_ATTRIB DERIVED_VEC_INT_TYPE trunc (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::truncToInt<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE, DERIVED_VEC_INT_TYPE, SCALAR_INT_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // FLOOR
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE floor () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::floor<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MFLOOR
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE floor (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::floor<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // CEIL
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE ceil () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::ceil<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MCEIL
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE ceil (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::ceil<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISFIN
        UME_FUNC_ATTRIB MASK_TYPE isfin () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::isfin<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISINF
        UME_FUNC_ATTRIB MASK_TYPE isinf () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::isinf<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISAN
        UME_FUNC_ATTRIB MASK_TYPE isan () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::isan<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISNAN
        UME_FUNC_ATTRIB MASK_TYPE isnan () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::isnan<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISNORM
        UME_FUNC_ATTRIB MASK_TYPE isnorm() const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::isnorm<DERIVED_VEC_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISSUB
        UME_FUNC_ATTRIB MASK_TYPE issub () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::issub<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISZERO
        UME_FUNC_ATTRIB MASK_TYPE iszero () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::iszero<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ISZEROSUB
        UME_FUNC_ATTRIB MASK_TYPE iszerosub () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::iszerosub<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE, MASK_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // EXP
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE exp () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::exp<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MEXP
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE exp (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::exp<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // LOG
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE log() const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::log<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // LOG10
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE log10() const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::log10<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // LOG2
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE log2() const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::log2<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SIN
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE sin () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::sin<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MSIN
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE sin (MASK_TYPE const & mask)const  {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::sin<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // COS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE cos () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::cos<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MCOS
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE cos (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::cos<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // SINCOS
        UME_FUNC_ATTRIB void sincos(DERIVED_VEC_TYPE & sinvec, DERIVED_VEC_TYPE & cosvec) const {
            UME_EMULATION_WARNING();
            sinvec = SCALAR_EMULATION::MATH::sin<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
            cosvec = SCALAR_EMULATION::MATH::cos<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MSINCOS
        UME_FUNC_ATTRIB void sincos(MASK_TYPE const & mask, DERIVED_VEC_TYPE & sinvec, DERIVED_VEC_TYPE & cosvec) const {
            UME_EMULATION_WARNING();
            sinvec = SCALAR_EMULATION::MATH::sin<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
            cosvec = SCALAR_EMULATION::MATH::cos<DERIVED_VEC_TYPE, MASK_TYPE>(mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // TAN
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE tan () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::tan<DERIVED_VEC_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MTAN
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE tan (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::tan<DERIVED_VEC_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // CTAN
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE ctan () const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::ctan<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE> (static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // MCTAN
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE ctan (MASK_TYPE const & mask) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::ctan<DERIVED_VEC_TYPE, SCALAR_FLOAT_TYPE, MASK_TYPE> (mask, static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ATAN
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE atan() const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::atan<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this));
        }

        // ATAN2
        UME_FUNC_ATTRIB DERIVED_VEC_TYPE atan2(DERIVED_VEC_TYPE const & b) const {
            UME_EMULATION_WARNING();
            return SCALAR_EMULATION::MATH::atan2<DERIVED_VEC_TYPE>(static_cast<DERIVED_VEC_TYPE const &>(*this), b);
        }

    };

} // namespace UME::SIMD
} // namespace UME

#endif
