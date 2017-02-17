#ifndef UME_SIMD_SCALAR_OPERATORS_H_
#define UME_SIMD_SCALAR_OPERATORS_H_

// Operators that take scalar left LHS operand have to be defined outside interface. The scalar type cannot be
// made a template parameter because it will cause problems with operators having scalar RHS operand. This
// requires explicit declaration of operators for every scalar type.
namespace UME {
    namespace SIMD {
        // LANDS
        UME_FORCE_INLINE SIMDMask1 operator& (bool a, SIMDMask1 const &b) { return b.land(a); }
        UME_FORCE_INLINE SIMDMask2 operator& (bool a, SIMDMask2 const &b) { return b.land(a); }
        UME_FORCE_INLINE SIMDMask4 operator& (bool a, SIMDMask4 const &b) { return b.land(a); }
        UME_FORCE_INLINE SIMDMask8 operator& (bool a, SIMDMask8 const &b) { return b.land(a); }
        UME_FORCE_INLINE SIMDMask16 operator& (bool a, SIMDMask16 const &b) { return b.land(a); }
        UME_FORCE_INLINE SIMDMask32 operator& (bool a, SIMDMask32 const &b) { return b.land(a); }
        UME_FORCE_INLINE SIMDMask64 operator& (bool a, SIMDMask64 const &b) { return b.land(a); }
        UME_FORCE_INLINE SIMDMask128 operator& (bool a, SIMDMask128 const &b) { return b.land(a); }

        UME_FORCE_INLINE SIMDMask1 operator&& (bool a, SIMDMask1 const &b) { return b.land(a); }
        UME_FORCE_INLINE SIMDMask2 operator&& (bool a, SIMDMask2 const &b) { return b.land(a); }
        UME_FORCE_INLINE SIMDMask4 operator&& (bool a, SIMDMask4 const &b) { return b.land(a); }
        UME_FORCE_INLINE SIMDMask8 operator&& (bool a, SIMDMask8 const &b) { return b.land(a); }
        UME_FORCE_INLINE SIMDMask16 operator&& (bool a, SIMDMask16 const &b) { return b.land(a); }
        UME_FORCE_INLINE SIMDMask32 operator&& (bool a, SIMDMask32 const &b) { return b.land(a); }
        UME_FORCE_INLINE SIMDMask64 operator&& (bool a, SIMDMask64 const &b) { return b.land(a); }
        UME_FORCE_INLINE SIMDMask128 operator&& (bool a, SIMDMask128 const &b) { return b.land(a); }

        // LORS
        UME_FORCE_INLINE SIMDMask1 operator| (bool a, SIMDMask1 const &b) { return b.lor(a); }
        UME_FORCE_INLINE SIMDMask2 operator| (bool a, SIMDMask2 const &b) { return b.lor(a); }
        UME_FORCE_INLINE SIMDMask4 operator| (bool a, SIMDMask4 const &b) { return b.lor(a); }
        UME_FORCE_INLINE SIMDMask8 operator| (bool a, SIMDMask8 const &b) { return b.lor(a); }
        UME_FORCE_INLINE SIMDMask16 operator| (bool a, SIMDMask16 const &b) { return b.lor(a); }
        UME_FORCE_INLINE SIMDMask32 operator| (bool a, SIMDMask32 const &b) { return b.lor(a); }
        UME_FORCE_INLINE SIMDMask64 operator| (bool a, SIMDMask64 const &b) { return b.lor(a); }
        UME_FORCE_INLINE SIMDMask128 operator| (bool a, SIMDMask128 const &b) { return b.lor(a); }

        UME_FORCE_INLINE SIMDMask1 operator|| (bool a, SIMDMask1 const &b) { return b.lor(a); }
        UME_FORCE_INLINE SIMDMask2 operator|| (bool a, SIMDMask2 const &b) { return b.lor(a); }
        UME_FORCE_INLINE SIMDMask4 operator|| (bool a, SIMDMask4 const &b) { return b.lor(a); }
        UME_FORCE_INLINE SIMDMask8 operator|| (bool a, SIMDMask8 const &b) { return b.lor(a); }
        UME_FORCE_INLINE SIMDMask16 operator|| (bool a, SIMDMask16 const &b) { return b.lor(a); }
        UME_FORCE_INLINE SIMDMask32 operator|| (bool a, SIMDMask32 const &b) { return b.lor(a); }
        UME_FORCE_INLINE SIMDMask64 operator|| (bool a, SIMDMask64 const &b) { return b.lor(a); }
        UME_FORCE_INLINE SIMDMask128 operator|| (bool a, SIMDMask128 const &b) { return b.lor(a); }

        // LXORS
        UME_FORCE_INLINE SIMDMask1 operator^ (bool a, SIMDMask1 const &b) { return b.lxor(a); }
        UME_FORCE_INLINE SIMDMask2 operator^ (bool a, SIMDMask2 const &b) { return b.lxor(a); }
        UME_FORCE_INLINE SIMDMask4 operator^ (bool a, SIMDMask4 const &b) { return b.lxor(a); }
        UME_FORCE_INLINE SIMDMask8 operator^ (bool a, SIMDMask8 const &b) { return b.lxor(a); }
        UME_FORCE_INLINE SIMDMask16 operator^ (bool a, SIMDMask16 const &b) { return b.lxor(a); }
        UME_FORCE_INLINE SIMDMask32 operator^ (bool a, SIMDMask32 const &b) { return b.lxor(a); }
        UME_FORCE_INLINE SIMDMask64 operator^ (bool a, SIMDMask64 const &b) { return b.lxor(a); }
        UME_FORCE_INLINE SIMDMask128 operator^ (bool a, SIMDMask128 const &b) { return b.lxor(a); }

        // CMPEQS
        UME_FORCE_INLINE SIMDMask1 operator== (bool a, SIMDMask1 const &b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask2 operator== (bool a, SIMDMask2 const &b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask4 operator== (bool a, SIMDMask4 const &b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask8 operator== (bool a, SIMDMask8 const &b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask16 operator== (bool a, SIMDMask16 const &b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask32 operator== (bool a, SIMDMask32 const &b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask64 operator== (bool a, SIMDMask64 const &b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask128 operator== (bool a, SIMDMask128 const &b) { return b.cmpeq(a); }

        // CMPNES
        UME_FORCE_INLINE SIMDMask1 operator!= (bool a, SIMDMask1 const &b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask2 operator!= (bool a, SIMDMask2 const &b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask4 operator!= (bool a, SIMDMask4 const &b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask8 operator!= (bool a, SIMDMask8 const &b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask16 operator!= (bool a, SIMDMask16 const &b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask32 operator!= (bool a, SIMDMask32 const &b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask64 operator!= (bool a, SIMDMask64 const &b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask128 operator!= (bool a, SIMDMask128 const &b) { return b.cmpne(a); }

        // ADDS
        UME_FORCE_INLINE SIMD1_8u operator+ (uint8_t a, SIMD1_8u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD2_8u operator+ (uint8_t a, SIMD2_8u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD4_8u operator+ (uint8_t a, SIMD4_8u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD8_8u operator+ (uint8_t a, SIMD8_8u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD16_8u operator+ (uint8_t a, SIMD16_8u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD32_8u operator+ (uint8_t a, SIMD32_8u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD64_8u operator+ (uint8_t a, SIMD64_8u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD128_8u operator+ (uint8_t a, SIMD128_8u const & b) { return b.add(a); }

        UME_FORCE_INLINE SIMD1_16u operator+ (uint16_t a, SIMD1_16u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD2_16u operator+ (uint16_t a, SIMD2_16u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD4_16u operator+ (uint16_t a, SIMD4_16u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD8_16u operator+ (uint16_t a, SIMD8_16u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD16_16u operator+ (uint16_t a, SIMD16_16u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD32_16u operator+ (uint16_t a, SIMD32_16u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD64_16u operator+ (uint16_t a, SIMD64_16u const & b) { return b.add(a); }

        UME_FORCE_INLINE SIMD1_32u operator+ (uint32_t a, SIMD1_32u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD2_32u operator+ (uint32_t a, SIMD2_32u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD4_32u operator+ (uint32_t a, SIMD4_32u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD8_32u operator+ (uint32_t a, SIMD8_32u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD16_32u operator+ (uint32_t a, SIMD16_32u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD32_32u operator+ (uint32_t a, SIMD32_32u const & b) { return b.add(a); }

        UME_FORCE_INLINE SIMD1_64u operator+ (uint64_t a, SIMD1_64u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD2_64u operator+ (uint64_t a, SIMD2_64u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD4_64u operator+ (uint64_t a, SIMD4_64u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD8_64u operator+ (uint64_t a, SIMD8_64u const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD16_64u operator+ (uint64_t a, SIMD16_64u const & b) { return b.add(a); }

        UME_FORCE_INLINE SIMD1_8i operator+ (int8_t a, SIMD1_8i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD2_8i operator+ (int8_t a, SIMD2_8i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD4_8i operator+ (int8_t a, SIMD4_8i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD8_8i operator+ (int8_t a, SIMD8_8i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD16_8i operator+ (int8_t a, SIMD16_8i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD32_8i operator+ (int8_t a, SIMD32_8i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD64_8i operator+ (int8_t a, SIMD64_8i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD128_8i operator+ (int8_t a, SIMD128_8i const & b) { return b.add(a); }

        UME_FORCE_INLINE SIMD1_16i operator+ (int16_t a, SIMD1_16i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD2_16i operator+ (int16_t a, SIMD2_16i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD4_16i operator+ (int16_t a, SIMD4_16i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD8_16i operator+ (int16_t a, SIMD8_16i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD16_16i operator+ (int16_t a, SIMD16_16i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD32_16i operator+ (int16_t a, SIMD32_16i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD64_16i operator+ (int16_t a, SIMD64_16i const & b) { return b.add(a); }

        UME_FORCE_INLINE SIMD1_32i operator+ (int32_t a, SIMD1_32i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD2_32i operator+ (int32_t a, SIMD2_32i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD4_32i operator+ (int32_t a, SIMD4_32i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD8_32i operator+ (int32_t a, SIMD8_32i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD16_32i operator+ (int32_t a, SIMD16_32i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD32_32i operator+ (int32_t a, SIMD32_32i const & b) { return b.add(a); }

        UME_FORCE_INLINE SIMD1_64i operator+ (int64_t a, SIMD1_64i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD2_64i operator+ (int64_t a, SIMD2_64i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD4_64i operator+ (int64_t a, SIMD4_64i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD8_64i operator+ (int64_t a, SIMD8_64i const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD16_64i operator+ (int64_t a, SIMD16_64i const & b) { return b.add(a); }

        UME_FORCE_INLINE SIMD1_32f operator+ (float a, SIMD1_32f const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD2_32f operator+ (float a, SIMD2_32f const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD4_32f operator+ (float a, SIMD4_32f const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD8_32f operator+ (float a, SIMD8_32f const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD16_32f operator+ (float a, SIMD16_32f const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD32_32f operator+ (float a, SIMD32_32f const & b) { return b.add(a); }

        UME_FORCE_INLINE SIMD1_64f operator+ (double a, SIMD1_64f const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD2_64f operator+ (double a, SIMD2_64f const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD4_64f operator+ (double a, SIMD4_64f const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD8_64f operator+ (double a, SIMD8_64f const & b) { return b.add(a); }
        UME_FORCE_INLINE SIMD16_64f operator+ (double a, SIMD16_64f const & b) { return b.add(a); }

        // SUBFROMS
        UME_FORCE_INLINE SIMD1_8u operator- (uint8_t a, SIMD1_8u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD2_8u operator- (uint8_t a, SIMD2_8u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD4_8u operator- (uint8_t a, SIMD4_8u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD8_8u operator- (uint8_t a, SIMD8_8u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD16_8u operator- (uint8_t a, SIMD16_8u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD32_8u operator- (uint8_t a, SIMD32_8u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD64_8u operator- (uint8_t a, SIMD64_8u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD128_8u operator- (uint8_t a, SIMD128_8u const & b) { return b.subfrom(a); }

        UME_FORCE_INLINE SIMD1_16u operator- (uint16_t a, SIMD1_16u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD2_16u operator- (uint16_t a, SIMD2_16u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD4_16u operator- (uint16_t a, SIMD4_16u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD8_16u operator- (uint16_t a, SIMD8_16u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD16_16u operator- (uint16_t a, SIMD16_16u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD32_16u operator- (uint16_t a, SIMD32_16u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD64_16u operator- (uint16_t a, SIMD64_16u const & b) { return b.subfrom(a); }

        UME_FORCE_INLINE SIMD1_32u operator- (uint32_t a, SIMD1_32u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD2_32u operator- (uint32_t a, SIMD2_32u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD4_32u operator- (uint32_t a, SIMD4_32u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD8_32u operator- (uint32_t a, SIMD8_32u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD16_32u operator- (uint32_t a, SIMD16_32u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD32_32u operator- (uint32_t a, SIMD32_32u const & b) { return b.subfrom(a); }

        UME_FORCE_INLINE SIMD1_64u operator- (uint64_t a, SIMD1_64u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD2_64u operator- (uint64_t a, SIMD2_64u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD4_64u operator- (uint64_t a, SIMD4_64u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD8_64u operator- (uint64_t a, SIMD8_64u const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD16_64u operator- (uint64_t a, SIMD16_64u const & b) { return b.subfrom(a); }

        UME_FORCE_INLINE SIMD1_8i operator- (int8_t a, SIMD1_8i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD2_8i operator- (int8_t a, SIMD2_8i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD4_8i operator- (int8_t a, SIMD4_8i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD8_8i operator- (int8_t a, SIMD8_8i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD16_8i operator- (int8_t a, SIMD16_8i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD32_8i operator- (int8_t a, SIMD32_8i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD64_8i operator- (int8_t a, SIMD64_8i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD128_8i operator- (int8_t a, SIMD128_8i const & b) { return b.subfrom(a); }

        UME_FORCE_INLINE SIMD1_16i operator- (int16_t a, SIMD1_16i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD2_16i operator- (int16_t a, SIMD2_16i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD4_16i operator- (int16_t a, SIMD4_16i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD8_16i operator- (int16_t a, SIMD8_16i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD16_16i operator- (int16_t a, SIMD16_16i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD32_16i operator- (int16_t a, SIMD32_16i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD64_16i operator- (int16_t a, SIMD64_16i const & b) { return b.subfrom(a); }

        UME_FORCE_INLINE SIMD1_32i operator- (int32_t a, SIMD1_32i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD2_32i operator- (int32_t a, SIMD2_32i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD4_32i operator- (int32_t a, SIMD4_32i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD8_32i operator- (int32_t a, SIMD8_32i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD16_32i operator- (int32_t a, SIMD16_32i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD32_32i operator- (int32_t a, SIMD32_32i const & b) { return b.subfrom(a); }

        UME_FORCE_INLINE SIMD1_64i operator- (int64_t a, SIMD1_64i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD2_64i operator- (int64_t a, SIMD2_64i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD4_64i operator- (int64_t a, SIMD4_64i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD8_64i operator- (int64_t a, SIMD8_64i const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD16_64i operator- (int64_t a, SIMD16_64i const & b) { return b.subfrom(a); }

        UME_FORCE_INLINE SIMD1_32f operator- (float a, SIMD1_32f const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD2_32f operator- (float a, SIMD2_32f const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD4_32f operator- (float a, SIMD4_32f const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD8_32f operator- (float a, SIMD8_32f const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD16_32f operator- (float a, SIMD16_32f const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD32_32f operator- (float a, SIMD32_32f const & b) { return b.subfrom(a); }

        UME_FORCE_INLINE SIMD1_64f operator- (double a, SIMD1_64f const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD2_64f operator- (double a, SIMD2_64f const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD4_64f operator- (double a, SIMD4_64f const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD8_64f operator- (double a, SIMD8_64f const & b) { return b.subfrom(a); }
        UME_FORCE_INLINE SIMD16_64f operator- (double a, SIMD16_64f const & b) { return b.subfrom(a); }

        // MULS
        UME_FORCE_INLINE SIMD1_8u operator* (uint8_t a, SIMD1_8u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD2_8u operator* (uint8_t a, SIMD2_8u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD4_8u operator* (uint8_t a, SIMD4_8u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD8_8u operator* (uint8_t a, SIMD8_8u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD16_8u operator* (uint8_t a, SIMD16_8u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD32_8u operator* (uint8_t a, SIMD32_8u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD64_8u operator* (uint8_t a, SIMD64_8u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD128_8u operator* (uint8_t a, SIMD128_8u const & b) { return b.mul(a); }

        UME_FORCE_INLINE SIMD1_16u operator* (uint16_t a, SIMD1_16u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD2_16u operator* (uint16_t a, SIMD2_16u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD4_16u operator* (uint16_t a, SIMD4_16u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD8_16u operator* (uint16_t a, SIMD8_16u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD16_16u operator* (uint16_t a, SIMD16_16u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD32_16u operator* (uint16_t a, SIMD32_16u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD64_16u operator* (uint16_t a, SIMD64_16u const & b) { return b.mul(a); }

        UME_FORCE_INLINE SIMD1_32u operator* (uint32_t a, SIMD1_32u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD2_32u operator* (uint32_t a, SIMD2_32u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD4_32u operator* (uint32_t a, SIMD4_32u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD8_32u operator* (uint32_t a, SIMD8_32u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD16_32u operator* (uint32_t a, SIMD16_32u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD32_32u operator* (uint32_t a, SIMD32_32u const & b) { return b.mul(a); }

        UME_FORCE_INLINE SIMD1_64u operator* (uint64_t a, SIMD1_64u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD2_64u operator* (uint64_t a, SIMD2_64u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD4_64u operator* (uint64_t a, SIMD4_64u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD8_64u operator* (uint64_t a, SIMD8_64u const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD16_64u operator* (uint64_t a, SIMD16_64u const & b) { return b.mul(a); }

        UME_FORCE_INLINE SIMD1_8i operator* (int8_t a, SIMD1_8i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD2_8i operator* (int8_t a, SIMD2_8i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD4_8i operator* (int8_t a, SIMD4_8i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD8_8i operator* (int8_t a, SIMD8_8i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD16_8i operator* (int8_t a, SIMD16_8i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD32_8i operator* (int8_t a, SIMD32_8i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD64_8i operator* (int8_t a, SIMD64_8i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD128_8i operator* (int8_t a, SIMD128_8i const & b) { return b.mul(a); }

        UME_FORCE_INLINE SIMD1_16i operator* (int16_t a, SIMD1_16i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD2_16i operator* (int16_t a, SIMD2_16i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD4_16i operator* (int16_t a, SIMD4_16i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD8_16i operator* (int16_t a, SIMD8_16i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD16_16i operator* (int16_t a, SIMD16_16i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD32_16i operator* (int16_t a, SIMD32_16i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD64_16i operator* (int16_t a, SIMD64_16i const & b) { return b.mul(a); }

        UME_FORCE_INLINE SIMD1_32i operator* (int32_t a, SIMD1_32i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD2_32i operator* (int32_t a, SIMD2_32i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD4_32i operator* (int32_t a, SIMD4_32i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD8_32i operator* (int32_t a, SIMD8_32i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD16_32i operator* (int32_t a, SIMD16_32i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD32_32i operator* (int32_t a, SIMD32_32i const & b) { return b.mul(a); }

        UME_FORCE_INLINE SIMD1_64i operator* (int64_t a, SIMD1_64i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD2_64i operator* (int64_t a, SIMD2_64i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD4_64i operator* (int64_t a, SIMD4_64i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD8_64i operator* (int64_t a, SIMD8_64i const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD16_64i operator* (int64_t a, SIMD16_64i const & b) { return b.mul(a); }

        UME_FORCE_INLINE SIMD1_32f operator* (float a, SIMD1_32f const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD2_32f operator* (float a, SIMD2_32f const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD4_32f operator* (float a, SIMD4_32f const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD8_32f operator* (float a, SIMD8_32f const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD16_32f operator* (float a, SIMD16_32f const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD32_32f operator* (float a, SIMD32_32f const & b) { return b.mul(a); }

        UME_FORCE_INLINE SIMD1_64f operator* (double a, SIMD1_64f const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD2_64f operator* (double a, SIMD2_64f const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD4_64f operator* (double a, SIMD4_64f const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD8_64f operator* (double a, SIMD8_64f const & b) { return b.mul(a); }
        UME_FORCE_INLINE SIMD16_64f operator* (double a, SIMD16_64f const & b) { return b.mul(a); }

        // RCPS
        UME_FORCE_INLINE SIMD1_8u operator/ (uint8_t a, SIMD1_8u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD2_8u operator/ (uint8_t a, SIMD2_8u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD4_8u operator/ (uint8_t a, SIMD4_8u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD8_8u operator/ (uint8_t a, SIMD8_8u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD16_8u operator/ (uint8_t a, SIMD16_8u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD32_8u operator/ (uint8_t a, SIMD32_8u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD64_8u operator/ (uint8_t a, SIMD64_8u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD128_8u operator/ (uint8_t a, SIMD128_8u const & b) { return b.rcp(a); }

        UME_FORCE_INLINE SIMD1_16u operator/ (uint16_t a, SIMD1_16u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD2_16u operator/ (uint16_t a, SIMD2_16u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD4_16u operator/ (uint16_t a, SIMD4_16u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD8_16u operator/ (uint16_t a, SIMD8_16u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD16_16u operator/ (uint16_t a, SIMD16_16u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD32_16u operator/ (uint16_t a, SIMD32_16u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD64_16u operator/ (uint16_t a, SIMD64_16u const & b) { return b.rcp(a); }

        UME_FORCE_INLINE SIMD1_32u operator/ (uint32_t a, SIMD1_32u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD2_32u operator/ (uint32_t a, SIMD2_32u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD4_32u operator/ (uint32_t a, SIMD4_32u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD8_32u operator/ (uint32_t a, SIMD8_32u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD16_32u operator/ (uint32_t a, SIMD16_32u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD32_32u operator/ (uint32_t a, SIMD32_32u const & b) { return b.rcp(a); }

        UME_FORCE_INLINE SIMD1_64u operator/ (uint64_t a, SIMD1_64u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD2_64u operator/ (uint64_t a, SIMD2_64u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD4_64u operator/ (uint64_t a, SIMD4_64u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD8_64u operator/ (uint64_t a, SIMD8_64u const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD16_64u operator/ (uint64_t a, SIMD16_64u const & b) { return b.rcp(a); }

        UME_FORCE_INLINE SIMD1_8i operator/ (int8_t a, SIMD1_8i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD2_8i operator/ (int8_t a, SIMD2_8i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD4_8i operator/ (int8_t a, SIMD4_8i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD8_8i operator/ (int8_t a, SIMD8_8i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD16_8i operator/ (int8_t a, SIMD16_8i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD32_8i operator/ (int8_t a, SIMD32_8i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD64_8i operator/ (int8_t a, SIMD64_8i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD128_8i operator/ (int8_t a, SIMD128_8i const & b) { return b.rcp(a); }

        UME_FORCE_INLINE SIMD1_16i operator/ (int16_t a, SIMD1_16i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD2_16i operator/ (int16_t a, SIMD2_16i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD4_16i operator/ (int16_t a, SIMD4_16i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD8_16i operator/ (int16_t a, SIMD8_16i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD16_16i operator/ (int16_t a, SIMD16_16i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD32_16i operator/ (int16_t a, SIMD32_16i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD64_16i operator/ (int16_t a, SIMD64_16i const & b) { return b.rcp(a); }

        UME_FORCE_INLINE SIMD1_32i operator/ (int32_t a, SIMD1_32i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD2_32i operator/ (int32_t a, SIMD2_32i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD4_32i operator/ (int32_t a, SIMD4_32i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD8_32i operator/ (int32_t a, SIMD8_32i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD16_32i operator/ (int32_t a, SIMD16_32i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD32_32i operator/ (int32_t a, SIMD32_32i const & b) { return b.rcp(a); }

        UME_FORCE_INLINE SIMD1_64i operator/ (int64_t a, SIMD1_64i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD2_64i operator/ (int64_t a, SIMD2_64i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD4_64i operator/ (int64_t a, SIMD4_64i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD8_64i operator/ (int64_t a, SIMD8_64i const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD16_64i operator/ (int64_t a, SIMD16_64i const & b) { return b.rcp(a); }

        UME_FORCE_INLINE SIMD1_32f operator/ (float a, SIMD1_32f const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD2_32f operator/ (float a, SIMD2_32f const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD4_32f operator/ (float a, SIMD4_32f const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD8_32f operator/ (float a, SIMD8_32f const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD16_32f operator/ (float a, SIMD16_32f const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD32_32f operator/ (float a, SIMD32_32f const & b) { return b.rcp(a); }

        UME_FORCE_INLINE SIMD1_64f operator/ (double a, SIMD1_64f const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD2_64f operator/ (double a, SIMD2_64f const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD4_64f operator/ (double a, SIMD4_64f const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD8_64f operator/ (double a, SIMD8_64f const & b) { return b.rcp(a); }
        UME_FORCE_INLINE SIMD16_64f operator/ (double a, SIMD16_64f const & b) { return b.rcp(a); }

        // BANDS
        UME_FORCE_INLINE SIMD1_8u operator& (uint8_t a, SIMD1_8u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD2_8u operator& (uint8_t a, SIMD2_8u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD4_8u operator& (uint8_t a, SIMD4_8u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD8_8u operator& (uint8_t a, SIMD8_8u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD16_8u operator& (uint8_t a, SIMD16_8u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD32_8u operator& (uint8_t a, SIMD32_8u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD64_8u operator& (uint8_t a, SIMD64_8u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD128_8u operator& (uint8_t a, SIMD128_8u const & b) { return b.band(a); }

        UME_FORCE_INLINE SIMD1_16u operator& (uint16_t a, SIMD1_16u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD2_16u operator& (uint16_t a, SIMD2_16u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD4_16u operator& (uint16_t a, SIMD4_16u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD8_16u operator& (uint16_t a, SIMD8_16u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD16_16u operator& (uint16_t a, SIMD16_16u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD32_16u operator& (uint16_t a, SIMD32_16u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD64_16u operator& (uint16_t a, SIMD64_16u const & b) { return b.band(a); }

        UME_FORCE_INLINE SIMD1_32u operator& (uint32_t a, SIMD1_32u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD2_32u operator& (uint32_t a, SIMD2_32u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD4_32u operator& (uint32_t a, SIMD4_32u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD8_32u operator& (uint32_t a, SIMD8_32u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD16_32u operator& (uint32_t a, SIMD16_32u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD32_32u operator& (uint32_t a, SIMD32_32u const & b) { return b.band(a); }

        UME_FORCE_INLINE SIMD1_64u operator& (uint64_t a, SIMD1_64u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD2_64u operator& (uint64_t a, SIMD2_64u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD4_64u operator& (uint64_t a, SIMD4_64u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD8_64u operator& (uint64_t a, SIMD8_64u const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD16_64u operator& (uint64_t a, SIMD16_64u const & b) { return b.band(a); }

        UME_FORCE_INLINE SIMD1_8i operator& (int8_t a, SIMD1_8i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD2_8i operator& (int8_t a, SIMD2_8i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD4_8i operator& (int8_t a, SIMD4_8i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD8_8i operator& (int8_t a, SIMD8_8i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD16_8i operator& (int8_t a, SIMD16_8i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD32_8i operator& (int8_t a, SIMD32_8i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD64_8i operator& (int8_t a, SIMD64_8i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD128_8i operator& (int8_t a, SIMD128_8i const & b) { return b.band(a); }

        UME_FORCE_INLINE SIMD1_16i operator& (int16_t a, SIMD1_16i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD2_16i operator& (int16_t a, SIMD2_16i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD4_16i operator& (int16_t a, SIMD4_16i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD8_16i operator& (int16_t a, SIMD8_16i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD16_16i operator& (int16_t a, SIMD16_16i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD32_16i operator& (int16_t a, SIMD32_16i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD64_16i operator& (int16_t a, SIMD64_16i const & b) { return b.band(a); }

        UME_FORCE_INLINE SIMD1_32i operator& (int32_t a, SIMD1_32i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD2_32i operator& (int32_t a, SIMD2_32i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD4_32i operator& (int32_t a, SIMD4_32i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD8_32i operator& (int32_t a, SIMD8_32i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD16_32i operator& (int32_t a, SIMD16_32i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD32_32i operator& (int32_t a, SIMD32_32i const & b) { return b.band(a); }

        UME_FORCE_INLINE SIMD1_64i operator& (int64_t a, SIMD1_64i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD2_64i operator& (int64_t a, SIMD2_64i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD4_64i operator& (int64_t a, SIMD4_64i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD8_64i operator& (int64_t a, SIMD8_64i const & b) { return b.band(a); }
        UME_FORCE_INLINE SIMD16_64i operator& (int64_t a, SIMD16_64i const & b) { return b.band(a); }

        // BORS
        UME_FORCE_INLINE SIMD1_8u operator| (uint8_t a, SIMD1_8u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD2_8u operator| (uint8_t a, SIMD2_8u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD4_8u operator| (uint8_t a, SIMD4_8u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD8_8u operator| (uint8_t a, SIMD8_8u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD16_8u operator| (uint8_t a, SIMD16_8u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD32_8u operator| (uint8_t a, SIMD32_8u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD64_8u operator| (uint8_t a, SIMD64_8u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD128_8u operator| (uint8_t a, SIMD128_8u const & b) { return b.bor(a); }

        UME_FORCE_INLINE SIMD1_16u operator| (uint16_t a, SIMD1_16u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD2_16u operator| (uint16_t a, SIMD2_16u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD4_16u operator| (uint16_t a, SIMD4_16u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD8_16u operator| (uint16_t a, SIMD8_16u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD16_16u operator| (uint16_t a, SIMD16_16u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD32_16u operator| (uint16_t a, SIMD32_16u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD64_16u operator| (uint16_t a, SIMD64_16u const & b) { return b.bor(a); }

        UME_FORCE_INLINE SIMD1_32u operator| (uint32_t a, SIMD1_32u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD2_32u operator| (uint32_t a, SIMD2_32u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD4_32u operator| (uint32_t a, SIMD4_32u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD8_32u operator| (uint32_t a, SIMD8_32u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD16_32u operator| (uint32_t a, SIMD16_32u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD32_32u operator| (uint32_t a, SIMD32_32u const & b) { return b.bor(a); }

        UME_FORCE_INLINE SIMD1_64u operator| (uint64_t a, SIMD1_64u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD2_64u operator| (uint64_t a, SIMD2_64u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD4_64u operator| (uint64_t a, SIMD4_64u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD8_64u operator| (uint64_t a, SIMD8_64u const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD16_64u operator| (uint64_t a, SIMD16_64u const & b) { return b.bor(a); }

        UME_FORCE_INLINE SIMD1_8i operator| (int8_t a, SIMD1_8i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD2_8i operator| (int8_t a, SIMD2_8i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD4_8i operator| (int8_t a, SIMD4_8i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD8_8i operator| (int8_t a, SIMD8_8i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD16_8i operator| (int8_t a, SIMD16_8i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD32_8i operator| (int8_t a, SIMD32_8i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD64_8i operator| (int8_t a, SIMD64_8i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD128_8i operator| (int8_t a, SIMD128_8i const & b) { return b.bor(a); }

        UME_FORCE_INLINE SIMD1_16i operator| (int16_t a, SIMD1_16i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD2_16i operator| (int16_t a, SIMD2_16i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD4_16i operator| (int16_t a, SIMD4_16i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD8_16i operator| (int16_t a, SIMD8_16i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD16_16i operator| (int16_t a, SIMD16_16i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD32_16i operator| (int16_t a, SIMD32_16i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD64_16i operator| (int16_t a, SIMD64_16i const & b) { return b.bor(a); }

        UME_FORCE_INLINE SIMD1_32i operator| (int32_t a, SIMD1_32i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD2_32i operator| (int32_t a, SIMD2_32i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD4_32i operator| (int32_t a, SIMD4_32i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD8_32i operator| (int32_t a, SIMD8_32i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD16_32i operator| (int32_t a, SIMD16_32i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD32_32i operator| (int32_t a, SIMD32_32i const & b) { return b.bor(a); }

        UME_FORCE_INLINE SIMD1_64i operator| (int64_t a, SIMD1_64i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD2_64i operator| (int64_t a, SIMD2_64i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD4_64i operator| (int64_t a, SIMD4_64i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD8_64i operator| (int64_t a, SIMD8_64i const & b) { return b.bor(a); }
        UME_FORCE_INLINE SIMD16_64i operator| (int64_t a, SIMD16_64i const & b) { return b.bor(a); }

        // BXORS
        UME_FORCE_INLINE SIMD1_8u operator^ (uint8_t a, SIMD1_8u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD2_8u operator^ (uint8_t a, SIMD2_8u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD4_8u operator^ (uint8_t a, SIMD4_8u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD8_8u operator^ (uint8_t a, SIMD8_8u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD16_8u operator^ (uint8_t a, SIMD16_8u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD32_8u operator^ (uint8_t a, SIMD32_8u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD64_8u operator^ (uint8_t a, SIMD64_8u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD128_8u operator^ (uint8_t a, SIMD128_8u const & b) { return b.bxor(a); }

        UME_FORCE_INLINE SIMD1_16u operator^ (uint16_t a, SIMD1_16u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD2_16u operator^ (uint16_t a, SIMD2_16u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD4_16u operator^ (uint16_t a, SIMD4_16u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD8_16u operator^ (uint16_t a, SIMD8_16u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD16_16u operator^ (uint16_t a, SIMD16_16u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD32_16u operator^ (uint16_t a, SIMD32_16u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD64_16u operator^ (uint16_t a, SIMD64_16u const & b) { return b.bxor(a); }

        UME_FORCE_INLINE SIMD1_32u operator^ (uint32_t a, SIMD1_32u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD2_32u operator^ (uint32_t a, SIMD2_32u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD4_32u operator^ (uint32_t a, SIMD4_32u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD8_32u operator^ (uint32_t a, SIMD8_32u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD16_32u operator^ (uint32_t a, SIMD16_32u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD32_32u operator^ (uint32_t a, SIMD32_32u const & b) { return b.bxor(a); }

        UME_FORCE_INLINE SIMD1_64u operator^ (uint64_t a, SIMD1_64u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD2_64u operator^ (uint64_t a, SIMD2_64u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD4_64u operator^ (uint64_t a, SIMD4_64u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD8_64u operator^ (uint64_t a, SIMD8_64u const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD16_64u operator^ (uint64_t a, SIMD16_64u const & b) { return b.bxor(a); }

        UME_FORCE_INLINE SIMD1_8i operator^ (int8_t a, SIMD1_8i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD2_8i operator^ (int8_t a, SIMD2_8i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD4_8i operator^ (int8_t a, SIMD4_8i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD8_8i operator^ (int8_t a, SIMD8_8i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD16_8i operator^ (int8_t a, SIMD16_8i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD32_8i operator^ (int8_t a, SIMD32_8i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD64_8i operator^ (int8_t a, SIMD64_8i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD128_8i operator^ (int8_t a, SIMD128_8i const & b) { return b.bxor(a); }

        UME_FORCE_INLINE SIMD1_16i operator^ (int16_t a, SIMD1_16i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD2_16i operator^ (int16_t a, SIMD2_16i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD4_16i operator^ (int16_t a, SIMD4_16i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD8_16i operator^ (int16_t a, SIMD8_16i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD16_16i operator^ (int16_t a, SIMD16_16i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD32_16i operator^ (int16_t a, SIMD32_16i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD64_16i operator^ (int16_t a, SIMD64_16i const & b) { return b.bxor(a); }

        UME_FORCE_INLINE SIMD1_32i operator^ (int32_t a, SIMD1_32i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD2_32i operator^ (int32_t a, SIMD2_32i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD4_32i operator^ (int32_t a, SIMD4_32i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD8_32i operator^ (int32_t a, SIMD8_32i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD16_32i operator^ (int32_t a, SIMD16_32i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD32_32i operator^ (int32_t a, SIMD32_32i const & b) { return b.bxor(a); }

        UME_FORCE_INLINE SIMD1_64i operator^ (int64_t a, SIMD1_64i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD2_64i operator^ (int64_t a, SIMD2_64i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD4_64i operator^ (int64_t a, SIMD4_64i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD8_64i operator^ (int64_t a, SIMD8_64i const & b) { return b.bxor(a); }
        UME_FORCE_INLINE SIMD16_64i operator^ (int64_t a, SIMD16_64i const & b) { return b.bxor(a); }

        // LSHS
        // This can only be defined for RHS unsingned integer vectors The result type will depend on
        // the LHS scalar type then.
        UME_FORCE_INLINE SIMD1_8u operator<< (uint8_t a, SIMD1_8u const & b) { return (SIMD1_8u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD2_8u operator<< (uint8_t a, SIMD2_8u const & b) { return (SIMD2_8u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD4_8u operator<< (uint8_t a, SIMD4_8u const & b) { return (SIMD4_8u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD8_8u operator<< (uint8_t a, SIMD8_8u const & b) { return (SIMD8_8u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD16_8u operator<< (uint8_t a, SIMD16_8u const & b) { return (SIMD16_8u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD32_8u operator<< (uint8_t a, SIMD32_8u const & b) { return (SIMD32_8u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD64_8u operator<< (uint8_t a, SIMD64_8u const & b) { return (SIMD64_8u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD128_8u operator<< (uint8_t a, SIMD128_8u const & b) { return (SIMD128_8u(a)).lsh(b); }

        UME_FORCE_INLINE SIMD1_16u operator<< (uint16_t a, SIMD1_16u const & b) { return (SIMD1_16u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD2_16u operator<< (uint16_t a, SIMD2_16u const & b) { return (SIMD2_16u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD4_16u operator<< (uint16_t a, SIMD4_16u const & b) { return (SIMD4_16u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD8_16u operator<< (uint16_t a, SIMD8_16u const & b) { return (SIMD8_16u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD16_16u operator<< (uint16_t a, SIMD16_16u const & b) { return (SIMD16_16u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD32_16u operator<< (uint16_t a, SIMD32_16u const & b) { return (SIMD32_16u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD64_16u operator<< (uint16_t a, SIMD64_16u const & b) { return (SIMD64_16u(a)).lsh(b); }

        UME_FORCE_INLINE SIMD1_32u operator<< (uint32_t a, SIMD1_32u const & b) { return (SIMD1_32u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD2_32u operator<< (uint32_t a, SIMD2_32u const & b) { return (SIMD2_32u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD4_32u operator<< (uint32_t a, SIMD4_32u const & b) { return (SIMD4_32u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD8_32u operator<< (uint32_t a, SIMD8_32u const & b) { return (SIMD8_32u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD16_32u operator<< (uint32_t a, SIMD16_32u const & b) { return (SIMD16_32u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD32_32u operator<< (uint32_t a, SIMD32_32u const & b) { return (SIMD32_32u(a)).lsh(b); }

        UME_FORCE_INLINE SIMD1_64u operator<< (uint64_t a, SIMD1_64u const & b) { return (SIMD1_64u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD2_64u operator<< (uint64_t a, SIMD2_64u const & b) { return (SIMD2_64u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD4_64u operator<< (uint64_t a, SIMD4_64u const & b) { return (SIMD4_64u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD8_64u operator<< (uint64_t a, SIMD8_64u const & b) { return (SIMD8_64u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD16_64u operator<< (uint64_t a, SIMD16_64u const & b) { return (SIMD16_64u(a)).lsh(b); }

        UME_FORCE_INLINE SIMD1_8i operator<< (int8_t a, SIMD1_8u const & b) { return (SIMD1_8u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD2_8i operator<< (int8_t a, SIMD2_8u const & b) { return (SIMD2_8u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD4_8i operator<< (int8_t a, SIMD4_8u const & b) { return (SIMD4_8u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD8_8i operator<< (int8_t a, SIMD8_8u const & b) { return (SIMD8_8u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD16_8i operator<< (int8_t a, SIMD16_8u const & b) { return (SIMD16_8u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD32_8i operator<< (int8_t a, SIMD32_8u const & b) { return (SIMD32_8u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD64_8i operator<< (int8_t a, SIMD64_8u const & b) { return (SIMD64_8u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD128_8i operator<< (int8_t a, SIMD128_8u const & b) { return (SIMD128_8u(a)).lsh(b); }

        UME_FORCE_INLINE SIMD1_16i operator<< (int16_t a, SIMD1_16u const & b) { return (SIMD1_16u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD2_16i operator<< (int16_t a, SIMD2_16u const & b) { return (SIMD2_16u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD4_16i operator<< (int16_t a, SIMD4_16u const & b) { return (SIMD4_16u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD8_16i operator<< (int16_t a, SIMD8_16u const & b) { return (SIMD8_16u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD16_16i operator<< (int16_t a, SIMD16_16u const & b) { return (SIMD16_16u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD32_16i operator<< (int16_t a, SIMD32_16u const & b) { return (SIMD32_16u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD64_16i operator<< (int16_t a, SIMD64_16u const & b) { return (SIMD64_16u(a)).lsh(b); }

        UME_FORCE_INLINE SIMD1_32i operator<< (int32_t a, SIMD1_32u const & b) { return (SIMD1_32u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD2_32i operator<< (int32_t a, SIMD2_32u const & b) { return (SIMD2_32u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD4_32i operator<< (int32_t a, SIMD4_32u const & b) { return (SIMD4_32u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD8_32i operator<< (int32_t a, SIMD8_32u const & b) { return (SIMD8_32u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD16_32i operator<< (int32_t a, SIMD16_32u const & b) { return (SIMD16_32u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD32_32i operator<< (int32_t a, SIMD32_32u const & b) { return (SIMD32_32u(a)).lsh(b); }

        UME_FORCE_INLINE SIMD1_64i operator<< (int64_t a, SIMD1_64u const & b) { return (SIMD1_64u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD2_64i operator<< (int64_t a, SIMD2_64u const & b) { return (SIMD2_64u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD4_64i operator<< (int64_t a, SIMD4_64u const & b) { return (SIMD4_64u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD8_64i operator<< (int64_t a, SIMD8_64u const & b) { return (SIMD8_64u(a)).lsh(b); }
        UME_FORCE_INLINE SIMD16_64i operator<< (int64_t a, SIMD16_64u const & b) { return (SIMD16_64u(a)).lsh(b); }

        // RSHS
        // This can only be defined for RHS unsingned integer vectors The result type will depend on
        // the LHS scalar type then.
        UME_FORCE_INLINE SIMD1_8u operator>>(uint8_t a, SIMD1_8u const & b) { return (SIMD1_8u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD2_8u operator>>(uint8_t a, SIMD2_8u const & b) { return (SIMD2_8u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD4_8u operator>>(uint8_t a, SIMD4_8u const & b) { return (SIMD4_8u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD8_8u operator>>(uint8_t a, SIMD8_8u const & b) { return (SIMD8_8u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD16_8u operator>>(uint8_t a, SIMD16_8u const & b) { return (SIMD16_8u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD32_8u operator>>(uint8_t a, SIMD32_8u const & b) { return (SIMD32_8u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD64_8u operator>>(uint8_t a, SIMD64_8u const & b) { return (SIMD64_8u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD128_8u operator>>(uint8_t a, SIMD128_8u const & b) { return (SIMD128_8u(a)).rsh(b); }

        UME_FORCE_INLINE SIMD1_16u operator>>(uint16_t a, SIMD1_16u const & b) { return (SIMD1_16u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD2_16u operator>>(uint16_t a, SIMD2_16u const & b) { return (SIMD2_16u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD4_16u operator>>(uint16_t a, SIMD4_16u const & b) { return (SIMD4_16u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD8_16u operator>>(uint16_t a, SIMD8_16u const & b) { return (SIMD8_16u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD16_16u operator>>(uint16_t a, SIMD16_16u const & b) { return (SIMD16_16u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD32_16u operator>>(uint16_t a, SIMD32_16u const & b) { return (SIMD32_16u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD64_16u operator>>(uint16_t a, SIMD64_16u const & b) { return (SIMD64_16u(a)).rsh(b); }

        UME_FORCE_INLINE SIMD1_32u operator>>(uint32_t a, SIMD1_32u const & b) { return (SIMD1_32u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD2_32u operator>>(uint32_t a, SIMD2_32u const & b) { return (SIMD2_32u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD4_32u operator>>(uint32_t a, SIMD4_32u const & b) { return (SIMD4_32u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD8_32u operator>>(uint32_t a, SIMD8_32u const & b) { return (SIMD8_32u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD16_32u operator>>(uint32_t a, SIMD16_32u const & b) { return (SIMD16_32u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD32_32u operator>>(uint32_t a, SIMD32_32u const & b) { return (SIMD32_32u(a)).rsh(b); }

        UME_FORCE_INLINE SIMD1_64u operator>>(uint64_t a, SIMD1_64u const & b) { return (SIMD1_64u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD2_64u operator>>(uint64_t a, SIMD2_64u const & b) { return (SIMD2_64u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD4_64u operator>>(uint64_t a, SIMD4_64u const & b) { return (SIMD4_64u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD8_64u operator>>(uint64_t a, SIMD8_64u const & b) { return (SIMD8_64u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD16_64u operator>>(uint64_t a, SIMD16_64u const & b) { return (SIMD16_64u(a)).rsh(b); }

        UME_FORCE_INLINE SIMD1_8i operator>>(int8_t a, SIMD1_8u const & b) { return (SIMD1_8u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD2_8i operator>>(int8_t a, SIMD2_8u const & b) { return (SIMD2_8u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD4_8i operator>>(int8_t a, SIMD4_8u const & b) { return (SIMD4_8u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD8_8i operator>>(int8_t a, SIMD8_8u const & b) { return (SIMD8_8u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD16_8i operator>>(int8_t a, SIMD16_8u const & b) { return (SIMD16_8u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD32_8i operator>>(int8_t a, SIMD32_8u const & b) { return (SIMD32_8u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD64_8i operator>>(int8_t a, SIMD64_8u const & b) { return (SIMD64_8u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD128_8i operator>>(int8_t a, SIMD128_8u const & b) { return (SIMD128_8u(a)).rsh(b); }

        UME_FORCE_INLINE SIMD1_16i operator>>(int16_t a, SIMD1_16u const & b) { return (SIMD1_16u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD2_16i operator>>(int16_t a, SIMD2_16u const & b) { return (SIMD2_16u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD4_16i operator>>(int16_t a, SIMD4_16u const & b) { return (SIMD4_16u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD8_16i operator>>(int16_t a, SIMD8_16u const & b) { return (SIMD8_16u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD16_16i operator>>(int16_t a, SIMD16_16u const & b) { return (SIMD16_16u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD32_16i operator>>(int16_t a, SIMD32_16u const & b) { return (SIMD32_16u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD64_16i operator>>(int16_t a, SIMD64_16u const & b) { return (SIMD64_16u(a)).rsh(b); }

        UME_FORCE_INLINE SIMD1_32i operator>>(int32_t a, SIMD1_32u const & b) { return (SIMD1_32u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD2_32i operator>>(int32_t a, SIMD2_32u const & b) { return (SIMD2_32u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD4_32i operator>>(int32_t a, SIMD4_32u const & b) { return (SIMD4_32u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD8_32i operator>>(int32_t a, SIMD8_32u const & b) { return (SIMD8_32u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD16_32i operator>>(int32_t a, SIMD16_32u const & b) { return (SIMD16_32u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD32_32i operator>>(int32_t a, SIMD32_32u const & b) { return (SIMD32_32u(a)).rsh(b); }

        UME_FORCE_INLINE SIMD1_64i operator>>(int64_t a, SIMD1_64u const & b) { return (SIMD1_64u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD2_64i operator>>(int64_t a, SIMD2_64u const & b) { return (SIMD2_64u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD4_64i operator>>(int64_t a, SIMD4_64u const & b) { return (SIMD4_64u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD8_64i operator>>(int64_t a, SIMD8_64u const & b) { return (SIMD8_64u(a)).rsh(b); }
        UME_FORCE_INLINE SIMD16_64i operator>>(int64_t a, SIMD16_64u const & b) { return (SIMD16_64u(a)).rsh(b); }

        // CMPEQS
        UME_FORCE_INLINE SIMDMask1 operator== (uint8_t a, SIMD1_8u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask2 operator== (uint8_t a, SIMD2_8u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask4 operator== (uint8_t a, SIMD4_8u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask8 operator== (uint8_t a, SIMD8_8u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask16 operator== (uint8_t a, SIMD16_8u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask32 operator== (uint8_t a, SIMD32_8u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask64 operator== (uint8_t a, SIMD64_8u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask128 operator== (uint8_t a, SIMD128_8u const & b) { return b.cmpeq(a); }

        UME_FORCE_INLINE SIMDMask1 operator== (uint16_t a, SIMD1_16u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask2 operator== (uint16_t a, SIMD2_16u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask4 operator== (uint16_t a, SIMD4_16u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask8 operator== (uint16_t a, SIMD8_16u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask16 operator== (uint16_t a, SIMD16_16u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask32 operator== (uint16_t a, SIMD32_16u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask64 operator== (uint16_t a, SIMD64_16u const & b) { return b.cmpeq(a); }

        UME_FORCE_INLINE SIMDMask1 operator== (uint32_t a, SIMD1_32u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask2 operator== (uint32_t a, SIMD2_32u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask4 operator== (uint32_t a, SIMD4_32u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask8 operator== (uint32_t a, SIMD8_32u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask16 operator== (uint32_t a, SIMD16_32u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask32 operator== (uint32_t a, SIMD32_32u const & b) { return b.cmpeq(a); }

        UME_FORCE_INLINE SIMDMask1 operator== (uint64_t a, SIMD1_64u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask2 operator== (uint64_t a, SIMD2_64u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask4 operator== (uint64_t a, SIMD4_64u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask8 operator== (uint64_t a, SIMD8_64u const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask16 operator== (uint64_t a, SIMD16_64u const & b) { return b.cmpeq(a); }

        UME_FORCE_INLINE SIMDMask1 operator== (int8_t a, SIMD1_8i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask2 operator== (int8_t a, SIMD2_8i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask4 operator== (int8_t a, SIMD4_8i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask8 operator== (int8_t a, SIMD8_8i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask16 operator== (int8_t a, SIMD16_8i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask32 operator== (int8_t a, SIMD32_8i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask64 operator== (int8_t a, SIMD64_8i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask128 operator== (int8_t a, SIMD128_8i const & b) { return b.cmpeq(a); }

        UME_FORCE_INLINE SIMDMask1 operator== (int16_t a, SIMD1_16i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask2 operator== (int16_t a, SIMD2_16i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask4 operator== (int16_t a, SIMD4_16i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask8 operator== (int16_t a, SIMD8_16i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask16 operator== (int16_t a, SIMD16_16i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask32 operator== (int16_t a, SIMD32_16i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask64 operator== (int16_t a, SIMD64_16i const & b) { return b.cmpeq(a); }

        UME_FORCE_INLINE SIMDMask1 operator== (int32_t a, SIMD1_32i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask2 operator== (int32_t a, SIMD2_32i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask4 operator== (int32_t a, SIMD4_32i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask8 operator== (int32_t a, SIMD8_32i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask16 operator== (int32_t a, SIMD16_32i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask32 operator== (int32_t a, SIMD32_32i const & b) { return b.cmpeq(a); }

        UME_FORCE_INLINE SIMDMask1 operator== (int64_t a, SIMD1_64i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask2 operator== (int64_t a, SIMD2_64i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask4 operator== (int64_t a, SIMD4_64i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask8 operator== (int64_t a, SIMD8_64i const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask16 operator== (int64_t a, SIMD16_64i const & b) { return b.cmpeq(a); }

        UME_FORCE_INLINE SIMDMask1 operator== (float a, SIMD1_32f const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask2 operator== (float a, SIMD2_32f const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask4 operator== (float a, SIMD4_32f const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask8 operator== (float a, SIMD8_32f const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask16 operator== (float a, SIMD16_32f const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask32 operator== (float a, SIMD32_32f const & b) { return b.cmpeq(a); }

        UME_FORCE_INLINE SIMDMask1 operator== (double a, SIMD1_64f const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask2 operator== (double a, SIMD2_64f const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask4 operator== (double a, SIMD4_64f const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask8 operator== (double a, SIMD8_64f const & b) { return b.cmpeq(a); }
        UME_FORCE_INLINE SIMDMask16 operator== (double a, SIMD16_64f const & b) { return b.cmpeq(a); }

        //CMPNEQ
        UME_FORCE_INLINE SIMDMask1 operator!= (uint8_t a, SIMD1_8u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask2 operator!= (uint8_t a, SIMD2_8u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask4 operator!= (uint8_t a, SIMD4_8u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask8 operator!= (uint8_t a, SIMD8_8u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask16 operator!= (uint8_t a, SIMD16_8u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask32 operator!= (uint8_t a, SIMD32_8u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask64 operator!= (uint8_t a, SIMD64_8u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask128 operator!= (uint8_t a, SIMD128_8u const & b) { return b.cmpne(a); }

        UME_FORCE_INLINE SIMDMask1 operator!= (uint16_t a, SIMD1_16u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask2 operator!= (uint16_t a, SIMD2_16u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask4 operator!= (uint16_t a, SIMD4_16u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask8 operator!= (uint16_t a, SIMD8_16u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask16 operator!= (uint16_t a, SIMD16_16u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask32 operator!= (uint16_t a, SIMD32_16u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask64 operator!= (uint16_t a, SIMD64_16u const & b) { return b.cmpne(a); }

        UME_FORCE_INLINE SIMDMask1 operator!= (uint32_t a, SIMD1_32u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask2 operator!= (uint32_t a, SIMD2_32u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask4 operator!= (uint32_t a, SIMD4_32u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask8 operator!= (uint32_t a, SIMD8_32u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask16 operator!= (uint32_t a, SIMD16_32u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask32 operator!= (uint32_t a, SIMD32_32u const & b) { return b.cmpne(a); }

        UME_FORCE_INLINE SIMDMask1 operator!= (uint64_t a, SIMD1_64u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask2 operator!= (uint64_t a, SIMD2_64u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask4 operator!= (uint64_t a, SIMD4_64u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask8 operator!= (uint64_t a, SIMD8_64u const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask16 operator!= (uint64_t a, SIMD16_64u const & b) { return b.cmpne(a); }

        UME_FORCE_INLINE SIMDMask1 operator!= (int8_t a, SIMD1_8i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask2 operator!= (int8_t a, SIMD2_8i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask4 operator!= (int8_t a, SIMD4_8i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask8 operator!= (int8_t a, SIMD8_8i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask16 operator!= (int8_t a, SIMD16_8i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask32 operator!= (int8_t a, SIMD32_8i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask64 operator!= (int8_t a, SIMD64_8i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask128 operator!= (int8_t a, SIMD128_8i const & b) { return b.cmpne(a); }

        UME_FORCE_INLINE SIMDMask1 operator!= (int16_t a, SIMD1_16i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask2 operator!= (int16_t a, SIMD2_16i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask4 operator!= (int16_t a, SIMD4_16i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask8 operator!= (int16_t a, SIMD8_16i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask16 operator!= (int16_t a, SIMD16_16i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask32 operator!= (int16_t a, SIMD32_16i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask64 operator!= (int16_t a, SIMD64_16i const & b) { return b.cmpne(a); }

        UME_FORCE_INLINE SIMDMask1 operator!= (int32_t a, SIMD1_32i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask2 operator!= (int32_t a, SIMD2_32i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask4 operator!= (int32_t a, SIMD4_32i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask8 operator!= (int32_t a, SIMD8_32i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask16 operator!= (int32_t a, SIMD16_32i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask32 operator!= (int32_t a, SIMD32_32i const & b) { return b.cmpne(a); }

        UME_FORCE_INLINE SIMDMask1 operator!= (int64_t a, SIMD1_64i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask2 operator!= (int64_t a, SIMD2_64i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask4 operator!= (int64_t a, SIMD4_64i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask8 operator!= (int64_t a, SIMD8_64i const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask16 operator!= (int64_t a, SIMD16_64i const & b) { return b.cmpne(a); }

        UME_FORCE_INLINE SIMDMask1 operator!= (float a, SIMD1_32f const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask2 operator!= (float a, SIMD2_32f const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask4 operator!= (float a, SIMD4_32f const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask8 operator!= (float a, SIMD8_32f const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask16 operator!= (float a, SIMD16_32f const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask32 operator!= (float a, SIMD32_32f const & b) { return b.cmpne(a); }

        UME_FORCE_INLINE SIMDMask1 operator!= (double a, SIMD1_64f const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask2 operator!= (double a, SIMD2_64f const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask4 operator!= (double a, SIMD4_64f const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask8 operator!= (double a, SIMD8_64f const & b) { return b.cmpne(a); }
        UME_FORCE_INLINE SIMDMask16 operator!= (double a, SIMD16_64f const & b) { return b.cmpne(a); }

        //CMPGTS
        UME_FORCE_INLINE SIMDMask1 operator> (uint8_t a, SIMD1_8u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask2 operator> (uint8_t a, SIMD2_8u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask4 operator> (uint8_t a, SIMD4_8u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask8 operator> (uint8_t a, SIMD8_8u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask16 operator> (uint8_t a, SIMD16_8u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask32 operator> (uint8_t a, SIMD32_8u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask64 operator> (uint8_t a, SIMD64_8u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask128 operator> (uint8_t a, SIMD128_8u const & b) { return b.cmplt(a); }

        UME_FORCE_INLINE SIMDMask1 operator> (uint16_t a, SIMD1_16u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask2 operator> (uint16_t a, SIMD2_16u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask4 operator> (uint16_t a, SIMD4_16u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask8 operator> (uint16_t a, SIMD8_16u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask16 operator> (uint16_t a, SIMD16_16u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask32 operator> (uint16_t a, SIMD32_16u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask64 operator> (uint16_t a, SIMD64_16u const & b) { return b.cmplt(a); }

        UME_FORCE_INLINE SIMDMask1 operator> (uint32_t a, SIMD1_32u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask2 operator> (uint32_t a, SIMD2_32u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask4 operator> (uint32_t a, SIMD4_32u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask8 operator> (uint32_t a, SIMD8_32u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask16 operator> (uint32_t a, SIMD16_32u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask32 operator> (uint32_t a, SIMD32_32u const & b) { return b.cmplt(a); }

        UME_FORCE_INLINE SIMDMask1 operator> (uint64_t a, SIMD1_64u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask2 operator> (uint64_t a, SIMD2_64u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask4 operator> (uint64_t a, SIMD4_64u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask8 operator> (uint64_t a, SIMD8_64u const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask16 operator> (uint64_t a, SIMD16_64u const & b) { return b.cmplt(a); }

        UME_FORCE_INLINE SIMDMask1 operator> (int8_t a, SIMD1_8i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask2 operator> (int8_t a, SIMD2_8i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask4 operator> (int8_t a, SIMD4_8i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask8 operator> (int8_t a, SIMD8_8i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask16 operator> (int8_t a, SIMD16_8i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask32 operator> (int8_t a, SIMD32_8i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask64 operator> (int8_t a, SIMD64_8i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask128 operator> (int8_t a, SIMD128_8i const & b) { return b.cmplt(a); }

        UME_FORCE_INLINE SIMDMask1 operator> (int16_t a, SIMD1_16i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask2 operator> (int16_t a, SIMD2_16i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask4 operator> (int16_t a, SIMD4_16i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask8 operator> (int16_t a, SIMD8_16i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask16 operator> (int16_t a, SIMD16_16i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask32 operator> (int16_t a, SIMD32_16i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask64 operator> (int16_t a, SIMD64_16i const & b) { return b.cmplt(a); }

        UME_FORCE_INLINE SIMDMask1 operator> (int32_t a, SIMD1_32i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask2 operator> (int32_t a, SIMD2_32i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask4 operator> (int32_t a, SIMD4_32i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask8 operator> (int32_t a, SIMD8_32i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask16 operator> (int32_t a, SIMD16_32i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask32 operator> (int32_t a, SIMD32_32i const & b) { return b.cmplt(a); }

        UME_FORCE_INLINE SIMDMask1 operator> (int64_t a, SIMD1_64i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask2 operator> (int64_t a, SIMD2_64i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask4 operator> (int64_t a, SIMD4_64i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask8 operator> (int64_t a, SIMD8_64i const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask16 operator> (int64_t a, SIMD16_64i const & b) { return b.cmplt(a); }

        UME_FORCE_INLINE SIMDMask1 operator> (float a, SIMD1_32f const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask2 operator> (float a, SIMD2_32f const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask4 operator> (float a, SIMD4_32f const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask8 operator> (float a, SIMD8_32f const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask16 operator> (float a, SIMD16_32f const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask32 operator> (float a, SIMD32_32f const & b) { return b.cmplt(a); }

        UME_FORCE_INLINE SIMDMask1 operator> (double a, SIMD1_64f const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask2 operator> (double a, SIMD2_64f const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask4 operator> (double a, SIMD4_64f const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask8 operator> (double a, SIMD8_64f const & b) { return b.cmplt(a); }
        UME_FORCE_INLINE SIMDMask16 operator> (double a, SIMD16_64f const & b) { return b.cmplt(a); }

        //CMPLT
        UME_FORCE_INLINE SIMDMask1 operator< (uint8_t a, SIMD1_8u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask2 operator< (uint8_t a, SIMD2_8u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask4 operator< (uint8_t a, SIMD4_8u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask8 operator< (uint8_t a, SIMD8_8u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask16 operator< (uint8_t a, SIMD16_8u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask32 operator< (uint8_t a, SIMD32_8u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask64 operator< (uint8_t a, SIMD64_8u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask128 operator< (uint8_t a, SIMD128_8u const & b) { return b.cmpgt(a); }

        UME_FORCE_INLINE SIMDMask1 operator< (uint16_t a, SIMD1_16u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask2 operator< (uint16_t a, SIMD2_16u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask4 operator< (uint16_t a, SIMD4_16u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask8 operator< (uint16_t a, SIMD8_16u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask16 operator< (uint16_t a, SIMD16_16u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask32 operator< (uint16_t a, SIMD32_16u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask64 operator< (uint16_t a, SIMD64_16u const & b) { return b.cmpgt(a); }

        UME_FORCE_INLINE SIMDMask1 operator< (uint32_t a, SIMD1_32u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask2 operator< (uint32_t a, SIMD2_32u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask4 operator< (uint32_t a, SIMD4_32u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask8 operator< (uint32_t a, SIMD8_32u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask16 operator< (uint32_t a, SIMD16_32u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask32 operator< (uint32_t a, SIMD32_32u const & b) { return b.cmpgt(a); }

        UME_FORCE_INLINE SIMDMask1 operator< (uint64_t a, SIMD1_64u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask2 operator< (uint64_t a, SIMD2_64u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask4 operator< (uint64_t a, SIMD4_64u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask8 operator< (uint64_t a, SIMD8_64u const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask16 operator< (uint64_t a, SIMD16_64u const & b) { return b.cmpgt(a); }

        UME_FORCE_INLINE SIMDMask1 operator< (int8_t a, SIMD1_8i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask2 operator< (int8_t a, SIMD2_8i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask4 operator< (int8_t a, SIMD4_8i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask8 operator< (int8_t a, SIMD8_8i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask16 operator< (int8_t a, SIMD16_8i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask32 operator< (int8_t a, SIMD32_8i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask64 operator< (int8_t a, SIMD64_8i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask128 operator< (int8_t a, SIMD128_8i const & b) { return b.cmpgt(a); }

        UME_FORCE_INLINE SIMDMask1 operator< (int16_t a, SIMD1_16i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask2 operator< (int16_t a, SIMD2_16i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask4 operator< (int16_t a, SIMD4_16i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask8 operator< (int16_t a, SIMD8_16i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask16 operator< (int16_t a, SIMD16_16i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask32 operator< (int16_t a, SIMD32_16i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask64 operator< (int16_t a, SIMD64_16i const & b) { return b.cmpgt(a); }

        UME_FORCE_INLINE SIMDMask1 operator< (int32_t a, SIMD1_32i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask2 operator< (int32_t a, SIMD2_32i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask4 operator< (int32_t a, SIMD4_32i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask8 operator< (int32_t a, SIMD8_32i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask16 operator< (int32_t a, SIMD16_32i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask32 operator< (int32_t a, SIMD32_32i const & b) { return b.cmpgt(a); }

        UME_FORCE_INLINE SIMDMask1 operator< (int64_t a, SIMD1_64i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask2 operator< (int64_t a, SIMD2_64i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask4 operator< (int64_t a, SIMD4_64i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask8 operator< (int64_t a, SIMD8_64i const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask16 operator< (int64_t a, SIMD16_64i const & b) { return b.cmpgt(a); }

        UME_FORCE_INLINE SIMDMask1 operator< (float a, SIMD1_32f const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask2 operator< (float a, SIMD2_32f const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask4 operator< (float a, SIMD4_32f const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask8 operator< (float a, SIMD8_32f const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask16 operator< (float a, SIMD16_32f const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask32 operator< (float a, SIMD32_32f const & b) { return b.cmpgt(a); }

        UME_FORCE_INLINE SIMDMask1 operator< (double a, SIMD1_64f const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask2 operator< (double a, SIMD2_64f const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask4 operator< (double a, SIMD4_64f const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask8 operator< (double a, SIMD8_64f const & b) { return b.cmpgt(a); }
        UME_FORCE_INLINE SIMDMask16 operator< (double a, SIMD16_64f const & b) { return b.cmpgt(a); }

        //CMPGES
        UME_FORCE_INLINE SIMDMask1 operator>= (uint8_t a, SIMD1_8u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask2 operator>= (uint8_t a, SIMD2_8u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask4 operator>= (uint8_t a, SIMD4_8u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask8 operator>= (uint8_t a, SIMD8_8u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask16 operator>= (uint8_t a, SIMD16_8u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask32 operator>= (uint8_t a, SIMD32_8u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask64 operator>= (uint8_t a, SIMD64_8u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask128 operator>= (uint8_t a, SIMD128_8u const & b) { return b.cmple(a); }

        UME_FORCE_INLINE SIMDMask1 operator>= (uint16_t a, SIMD1_16u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask2 operator>= (uint16_t a, SIMD2_16u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask4 operator>= (uint16_t a, SIMD4_16u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask8 operator>= (uint16_t a, SIMD8_16u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask16 operator>= (uint16_t a, SIMD16_16u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask32 operator>= (uint16_t a, SIMD32_16u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask64 operator>= (uint16_t a, SIMD64_16u const & b) { return b.cmple(a); }

        UME_FORCE_INLINE SIMDMask1 operator>= (uint32_t a, SIMD1_32u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask2 operator>= (uint32_t a, SIMD2_32u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask4 operator>= (uint32_t a, SIMD4_32u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask8 operator>= (uint32_t a, SIMD8_32u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask16 operator>= (uint32_t a, SIMD16_32u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask32 operator>= (uint32_t a, SIMD32_32u const & b) { return b.cmple(a); }

        UME_FORCE_INLINE SIMDMask1 operator>= (uint64_t a, SIMD1_64u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask2 operator>= (uint64_t a, SIMD2_64u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask4 operator>= (uint64_t a, SIMD4_64u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask8 operator>= (uint64_t a, SIMD8_64u const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask16 operator>= (uint64_t a, SIMD16_64u const & b) { return b.cmple(a); }

        UME_FORCE_INLINE SIMDMask1 operator>= (int8_t a, SIMD1_8i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask2 operator>= (int8_t a, SIMD2_8i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask4 operator>= (int8_t a, SIMD4_8i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask8 operator>= (int8_t a, SIMD8_8i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask16 operator>= (int8_t a, SIMD16_8i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask32 operator>= (int8_t a, SIMD32_8i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask64 operator>= (int8_t a, SIMD64_8i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask128 operator>= (int8_t a, SIMD128_8i const & b) { return b.cmple(a); }

        UME_FORCE_INLINE SIMDMask1 operator>= (int16_t a, SIMD1_16i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask2 operator>= (int16_t a, SIMD2_16i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask4 operator>= (int16_t a, SIMD4_16i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask8 operator>= (int16_t a, SIMD8_16i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask16 operator>= (int16_t a, SIMD16_16i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask32 operator>= (int16_t a, SIMD32_16i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask64 operator>= (int16_t a, SIMD64_16i const & b) { return b.cmple(a); }

        UME_FORCE_INLINE SIMDMask1 operator>= (int32_t a, SIMD1_32i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask2 operator>= (int32_t a, SIMD2_32i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask4 operator>= (int32_t a, SIMD4_32i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask8 operator>= (int32_t a, SIMD8_32i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask16 operator>= (int32_t a, SIMD16_32i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask32 operator>= (int32_t a, SIMD32_32i const & b) { return b.cmple(a); }

        UME_FORCE_INLINE SIMDMask1 operator>= (int64_t a, SIMD1_64i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask2 operator>= (int64_t a, SIMD2_64i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask4 operator>= (int64_t a, SIMD4_64i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask8 operator>= (int64_t a, SIMD8_64i const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask16 operator>= (int64_t a, SIMD16_64i const & b) { return b.cmple(a); }

        UME_FORCE_INLINE SIMDMask1 operator>= (float a, SIMD1_32f const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask2 operator>= (float a, SIMD2_32f const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask4 operator>= (float a, SIMD4_32f const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask8 operator>= (float a, SIMD8_32f const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask16 operator>= (float a, SIMD16_32f const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask32 operator>= (float a, SIMD32_32f const & b) { return b.cmple(a); }

        UME_FORCE_INLINE SIMDMask1 operator>= (double a, SIMD1_64f const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask2 operator>= (double a, SIMD2_64f const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask4 operator>= (double a, SIMD4_64f const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask8 operator>= (double a, SIMD8_64f const & b) { return b.cmple(a); }
        UME_FORCE_INLINE SIMDMask16 operator>= (double a, SIMD16_64f const & b) { return b.cmple(a); }

        //CMPLES
        UME_FORCE_INLINE SIMDMask1 operator<= (uint8_t a, SIMD1_8u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask2 operator<= (uint8_t a, SIMD2_8u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask4 operator<= (uint8_t a, SIMD4_8u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask8 operator<= (uint8_t a, SIMD8_8u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask16 operator<= (uint8_t a, SIMD16_8u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask32 operator<= (uint8_t a, SIMD32_8u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask64 operator<= (uint8_t a, SIMD64_8u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask128 operator<= (uint8_t a, SIMD128_8u const & b) { return b.cmpge(a); }

        UME_FORCE_INLINE SIMDMask1 operator<= (uint16_t a, SIMD1_16u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask2 operator<= (uint16_t a, SIMD2_16u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask4 operator<= (uint16_t a, SIMD4_16u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask8 operator<= (uint16_t a, SIMD8_16u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask16 operator<= (uint16_t a, SIMD16_16u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask32 operator<= (uint16_t a, SIMD32_16u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask64 operator<= (uint16_t a, SIMD64_16u const & b) { return b.cmpge(a); }

        UME_FORCE_INLINE SIMDMask1 operator<= (uint32_t a, SIMD1_32u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask2 operator<= (uint32_t a, SIMD2_32u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask4 operator<= (uint32_t a, SIMD4_32u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask8 operator<= (uint32_t a, SIMD8_32u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask16 operator<= (uint32_t a, SIMD16_32u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask32 operator<= (uint32_t a, SIMD32_32u const & b) { return b.cmpge(a); }

        UME_FORCE_INLINE SIMDMask1 operator<= (uint64_t a, SIMD1_64u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask2 operator<= (uint64_t a, SIMD2_64u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask4 operator<= (uint64_t a, SIMD4_64u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask8 operator<= (uint64_t a, SIMD8_64u const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask16 operator<= (uint64_t a, SIMD16_64u const & b) { return b.cmpge(a); }

        UME_FORCE_INLINE SIMDMask1 operator<= (int8_t a, SIMD1_8i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask2 operator<= (int8_t a, SIMD2_8i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask4 operator<= (int8_t a, SIMD4_8i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask8 operator<= (int8_t a, SIMD8_8i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask16 operator<= (int8_t a, SIMD16_8i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask32 operator<= (int8_t a, SIMD32_8i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask64 operator<= (int8_t a, SIMD64_8i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask128 operator<= (int8_t a, SIMD128_8i const & b) { return b.cmpge(a); }

        UME_FORCE_INLINE SIMDMask1 operator<= (int16_t a, SIMD1_16i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask2 operator<= (int16_t a, SIMD2_16i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask4 operator<= (int16_t a, SIMD4_16i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask8 operator<= (int16_t a, SIMD8_16i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask16 operator<= (int16_t a, SIMD16_16i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask32 operator<= (int16_t a, SIMD32_16i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask64 operator<= (int16_t a, SIMD64_16i const & b) { return b.cmpge(a); }

        UME_FORCE_INLINE SIMDMask1 operator<= (int32_t a, SIMD1_32i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask2 operator<= (int32_t a, SIMD2_32i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask4 operator<= (int32_t a, SIMD4_32i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask8 operator<= (int32_t a, SIMD8_32i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask16 operator<= (int32_t a, SIMD16_32i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask32 operator<= (int32_t a, SIMD32_32i const & b) { return b.cmpge(a); }

        UME_FORCE_INLINE SIMDMask1 operator<= (int64_t a, SIMD1_64i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask2 operator<= (int64_t a, SIMD2_64i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask4 operator<= (int64_t a, SIMD4_64i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask8 operator<= (int64_t a, SIMD8_64i const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask16 operator<= (int64_t a, SIMD16_64i const & b) { return b.cmpge(a); }

        UME_FORCE_INLINE SIMDMask1 operator<= (float a, SIMD1_32f const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask2 operator<= (float a, SIMD2_32f const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask4 operator<= (float a, SIMD4_32f const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask8 operator<= (float a, SIMD8_32f const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask16 operator<= (float a, SIMD16_32f const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask32 operator<= (float a, SIMD32_32f const & b) { return b.cmpge(a); }

        UME_FORCE_INLINE SIMDMask1 operator<= (double a, SIMD1_64f const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask2 operator<= (double a, SIMD2_64f const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask4 operator<= (double a, SIMD4_64f const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask8 operator<= (double a, SIMD8_64f const & b) { return b.cmpge(a); }
        UME_FORCE_INLINE SIMDMask16 operator<= (double a, SIMD16_64f const & b) { return b.cmpge(a); }
    }
}
#endif
