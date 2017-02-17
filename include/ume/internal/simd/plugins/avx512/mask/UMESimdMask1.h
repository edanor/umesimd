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

#ifndef UME_SIMD_MASK_1_H_
#define UME_SIMD_MASK_1_H_

#include "UMESimdMaskPrototype.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVecMask<1> :
        public SIMDMaskBaseInterface<
        SIMDVecMask<1>,
        uint32_t,
        1>
    {
        friend class SIMDVec_u<uint32_t, 1>;
        friend class SIMDVec_u<uint64_t, 1>;
        friend class SIMDVec_i<int32_t, 1>;
        friend class SIMDVec_i<int64_t, 1>;
        friend class SIMDVec_f<float, 1>;
        friend class SIMDVec_f<double, 1>;
    private:
        bool mMask;

    public:
        UME_FORCE_INLINE SIMDVecMask() {}

        UME_FORCE_INLINE SIMDVecMask(SIMDVecMask const & mask) {
            mMask = mask.mMask;
        }
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVecMask(bool m) {
            mMask = m;
        }
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVecMask(bool const * p) {
            mMask = p[0];
        }

#include "../../../utilities/ignore_warnings_push.h"
#include "../../../utilities/ignore_warnings_unused_parameter.h"

        // EXTRACT
        UME_FORCE_INLINE bool extract(uint32_t index) const {
            return mMask;
        }

        // A non-modifying element-wise access operator
        UME_FORCE_INLINE bool operator[] (uint32_t index) const {
            return mMask;
        }
        // INSERT
        UME_FORCE_INLINE void insert(uint32_t index, bool x) {
            mMask = x;
        }

#include "../../../utilities/ignore_warnings_pop.h"

        // LOAD
        UME_FORCE_INLINE SIMDVecMask & load(bool * p) {
            mMask = p[0];
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVecMask & loada(bool * p) {
            mMask = p[0];
            return *this;
        }
        // STORE
        UME_FORCE_INLINE bool* store(bool * p) const {
            p[0] = mMask;
            return p;
        }
        // STOREA
        UME_FORCE_INLINE bool* storea(bool * p) const {
            p[0] = mMask;
            return p;
        }
        // ASSIGNV
        UME_FORCE_INLINE SIMDVecMask & assign(SIMDVecMask const & b) {
            mMask = b.mMask;
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator= (SIMDVecMask const & b) {
            mMask = b.mMask;
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVecMask & assign(bool b) {
            mMask = b;
            return *this;
        }
        // LANDV
        UME_FORCE_INLINE SIMDVecMask land(SIMDVecMask const & b) const {
            bool t0 = mMask && b.mMask;
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator& (SIMDVecMask const & b) const {
            return land(b);
        }
        UME_FORCE_INLINE SIMDVecMask operator&& (SIMDVecMask const & b) const {
            return land(b);
        }
        // LANDS
        UME_FORCE_INLINE SIMDVecMask land(bool b) const {
            bool t0 = mMask && b;
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator& (bool b) const {
            return land(b);
        }
        UME_FORCE_INLINE SIMDVecMask operator&& (bool b) const {
            return land(b);
        }
        // LANDVA
        UME_FORCE_INLINE SIMDVecMask & landa(SIMDVecMask const & b) {
            mMask = mMask && b.mMask;
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator&= (SIMDVecMask const & b) {
            return landa(b);
        }
        // LANDSA
        UME_FORCE_INLINE SIMDVecMask & landa(bool b) {
            mMask = mMask && b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator&= (bool b) {
            return landa(b);
        }
        // LORV
        UME_FORCE_INLINE SIMDVecMask lor(SIMDVecMask const & b) const {
            bool t0 = mMask || b.mMask;
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator| (SIMDVecMask const & b) const {
            return lor(b);
        }
        UME_FORCE_INLINE SIMDVecMask operator|| (SIMDVecMask const & b) const {
            return lor(b);
        }
        // LORS
        UME_FORCE_INLINE SIMDVecMask lor(bool b) const {
            bool t0 = mMask || b;
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator| (bool b) const {
            return lor(b);
        }
        UME_FORCE_INLINE SIMDVecMask operator|| (bool b) const {
            return lor(b);
        }
        // LORVA
        UME_FORCE_INLINE SIMDVecMask & lora(SIMDVecMask const & b) {
            mMask = mMask || b.mMask;
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator|= (SIMDVecMask const & b) {
            return lora(b);
        }
        // LORSA
        UME_FORCE_INLINE SIMDVecMask & lora(bool b) {
            mMask = mMask || b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator|= (bool b) {
            return lora(b);
        }
        // LXORV
        UME_FORCE_INLINE SIMDVecMask lxor(SIMDVecMask const & b) const {
            bool t0 = mMask ^ b.mMask;
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator^ (SIMDVecMask const & b) const {
            return lxor(b);
        }
        // LXORS
        UME_FORCE_INLINE SIMDVecMask lxor(bool b) const {
            bool t0 = mMask ^ b;
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator^ (bool b) const {
            return lxor(b);
        }
        // LXORVA
        UME_FORCE_INLINE SIMDVecMask & lxora(SIMDVecMask const & b) {
            mMask = mMask ^ b.mMask;
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator^= (SIMDVecMask const & b) {
            return lxora(b);
        }
        // LXORSA
        UME_FORCE_INLINE SIMDVecMask & lxora(bool b) {
            mMask = mMask ^ b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator^= (bool b) {
            return lxora(b);
        }
        // LNOT
        UME_FORCE_INLINE SIMDVecMask lnot() const {
            bool t0 = !mMask;
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator! () const {
            return lnot();
        }
        // LNOTA
        UME_FORCE_INLINE SIMDVecMask & lnota() {
            mMask = !mMask;
            return *this;
        }
        // LANDNOTV
        UME_FORCE_INLINE SIMDVecMask landnot(SIMDVecMask const & b) const {
            bool t0 = !mMask && b.mMask;
            return SIMDVecMask(t0);
        }
        // LANDNOTS
        UME_FORCE_INLINE SIMDVecMask landnot(bool b) const {
            bool t0 = !mMask && b;
            return SIMDVecMask(t0);
        }
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask cmpeq(SIMDVecMask const & b) const {
            bool t0 = mMask == b.mMask;
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator== (SIMDVecMask const & b) const {
            return cmpeq(b);
        }        
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask cmpeq(bool b) const {
            bool t0 = mMask == b;
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator== (bool b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask cmpne(SIMDVecMask const & b) const {
            bool t0 = mMask != b.mMask;
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator!= (SIMDVecMask const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask cmpne(bool b) const {
            bool t0 = mMask != b;
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator!= (bool b) const {
            return cmpne(b);
        }
        // HLAND
        UME_FORCE_INLINE bool hland() const {
            return mMask;
        }
        // HLOR
        UME_FORCE_INLINE bool hlor() const {
            return mMask;
        }
        // HLXOR
        UME_FORCE_INLINE bool hlxor() const {
            return mMask;
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVecMask const & b) const {
            return mMask == b.mMask;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(bool b) const {
            return mMask == b;
        }
    };

}
}

#endif
