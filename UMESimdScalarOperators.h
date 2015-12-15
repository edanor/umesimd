#ifndef UME_SIMD_SCALAR_OPERATORS_H_
#define UME_SIMD_SCALAR_OPERATORS_H_

// Operators that take scalar left LHS operand have to be defined outside interface. The scalar type cannot be
// made a template parameter because it will cause problems with operators having scalar RHS operand. This
// requires explicit declaration of operators for every scalar type.
namespace UME {
    namespace SIMD {

        // LANDS
        inline SIMDMask1 operator& (bool a, SIMDMask1 const &b) { return b.land(a); }
        inline SIMDMask2 operator& (bool a, SIMDMask2 const &b) { return b.land(a); }
        inline SIMDMask4 operator& (bool a, SIMDMask4 const &b) { return b.land(a); }
        inline SIMDMask8 operator& (bool a, SIMDMask8 const &b) { return b.land(a); }
        inline SIMDMask16 operator& (bool a, SIMDMask16 const &b) { return b.land(a); }
        inline SIMDMask32 operator& (bool a, SIMDMask32 const &b) { return b.land(a); }
        inline SIMDMask64 operator& (bool a, SIMDMask64 const &b) { return b.land(a); }
        inline SIMDMask128 operator& (bool a, SIMDMask128 const &b) { return b.land(a); }

        inline SIMDMask1 operator&& (bool a, SIMDMask1 const &b) { return b.land(a); }
        inline SIMDMask2 operator&& (bool a, SIMDMask2 const &b) { return b.land(a); }
        inline SIMDMask4 operator&& (bool a, SIMDMask4 const &b) { return b.land(a); }
        inline SIMDMask8 operator&& (bool a, SIMDMask8 const &b) { return b.land(a); }
        inline SIMDMask16 operator&& (bool a, SIMDMask16 const &b) { return b.land(a); }
        inline SIMDMask32 operator&& (bool a, SIMDMask32 const &b) { return b.land(a); }
        inline SIMDMask64 operator&& (bool a, SIMDMask64 const &b) { return b.land(a); }
        inline SIMDMask128 operator&& (bool a, SIMDMask128 const &b) { return b.land(a); }

        // LORS
        inline SIMDMask1 operator| (bool a, SIMDMask1 const &b) { return b.lor(a); }
        inline SIMDMask2 operator| (bool a, SIMDMask2 const &b) { return b.lor(a); }
        inline SIMDMask4 operator| (bool a, SIMDMask4 const &b) { return b.lor(a); }
        inline SIMDMask8 operator| (bool a, SIMDMask8 const &b) { return b.lor(a); }
        inline SIMDMask16 operator| (bool a, SIMDMask16 const &b) { return b.lor(a); }
        inline SIMDMask32 operator| (bool a, SIMDMask32 const &b) { return b.lor(a); }
        inline SIMDMask64 operator| (bool a, SIMDMask64 const &b) { return b.lor(a); }
        inline SIMDMask128 operator| (bool a, SIMDMask128 const &b) { return b.lor(a); }

        inline SIMDMask1 operator|| (bool a, SIMDMask1 const &b) { return b.lor(a); }
        inline SIMDMask2 operator|| (bool a, SIMDMask2 const &b) { return b.lor(a); }
        inline SIMDMask4 operator|| (bool a, SIMDMask4 const &b) { return b.lor(a); }
        inline SIMDMask8 operator|| (bool a, SIMDMask8 const &b) { return b.lor(a); }
        inline SIMDMask16 operator|| (bool a, SIMDMask16 const &b) { return b.lor(a); }
        inline SIMDMask32 operator|| (bool a, SIMDMask32 const &b) { return b.lor(a); }
        inline SIMDMask64 operator|| (bool a, SIMDMask64 const &b) { return b.lor(a); }
        inline SIMDMask128 operator|| (bool a, SIMDMask128 const &b) { return b.lor(a); }

        // LXORS
        inline SIMDMask1 operator^ (bool a, SIMDMask1 const &b) { return b.lxor(a); }
        inline SIMDMask2 operator^ (bool a, SIMDMask2 const &b) { return b.lxor(a); }
        inline SIMDMask4 operator^ (bool a, SIMDMask4 const &b) { return b.lxor(a); }
        inline SIMDMask8 operator^ (bool a, SIMDMask8 const &b) { return b.lxor(a); }
        inline SIMDMask16 operator^ (bool a, SIMDMask16 const &b) { return b.lxor(a); }
        inline SIMDMask32 operator^ (bool a, SIMDMask32 const &b) { return b.lxor(a); }
        inline SIMDMask64 operator^ (bool a, SIMDMask64 const &b) { return b.lxor(a); }
        inline SIMDMask128 operator^ (bool a, SIMDMask128 const &b) { return b.lxor(a); }

        // ADDS
        inline SIMD1_8u operator+ (uint8_t a, SIMD1_8u const & b) { return b.add(a); }
        inline SIMD2_8u operator+ (uint8_t a, SIMD2_8u const & b) { return b.add(a); }
        inline SIMD4_8u operator+ (uint8_t a, SIMD4_8u const & b) { return b.add(a); }
        inline SIMD8_8u operator+ (uint8_t a, SIMD8_8u const & b) { return b.add(a); }
        inline SIMD16_8u operator+ (uint8_t a, SIMD16_8u const & b) { return b.add(a); }
        inline SIMD32_8u operator+ (uint8_t a, SIMD32_8u const & b) { return b.add(a); }
        inline SIMD64_8u operator+ (uint8_t a, SIMD64_8u const & b) { return b.add(a); }
        inline SIMD128_8u operator+ (uint8_t a, SIMD128_8u const & b) { return b.add(a); }

        inline SIMD1_16u operator+ (uint16_t a, SIMD1_16u const & b) { return b.add(a); }
        inline SIMD2_16u operator+ (uint16_t a, SIMD2_16u const & b) { return b.add(a); }
        inline SIMD4_16u operator+ (uint16_t a, SIMD4_16u const & b) { return b.add(a); }
        inline SIMD8_16u operator+ (uint16_t a, SIMD8_16u const & b) { return b.add(a); }
        inline SIMD16_16u operator+ (uint16_t a, SIMD16_16u const & b) { return b.add(a); }
        inline SIMD32_16u operator+ (uint16_t a, SIMD32_16u const & b) { return b.add(a); }
        inline SIMD64_16u operator+ (uint16_t a, SIMD64_16u const & b) { return b.add(a); }

        inline SIMD1_32u operator+ (uint32_t a, SIMD1_32u const & b) { return b.add(a); }
        inline SIMD2_32u operator+ (uint32_t a, SIMD2_32u const & b) { return b.add(a); }
        inline SIMD4_32u operator+ (uint32_t a, SIMD4_32u const & b) { return b.add(a); }
        inline SIMD8_32u operator+ (uint32_t a, SIMD8_32u const & b) { return b.add(a); }
        inline SIMD16_32u operator+ (uint32_t a, SIMD16_32u const & b) { return b.add(a); }
        inline SIMD32_32u operator+ (uint32_t a, SIMD32_32u const & b) { return b.add(a); }

        inline SIMD1_64u operator+ (uint64_t a, SIMD1_64u const & b) { return b.add(a); }
        inline SIMD2_64u operator+ (uint64_t a, SIMD2_64u const & b) { return b.add(a); }
        inline SIMD4_64u operator+ (uint64_t a, SIMD4_64u const & b) { return b.add(a); }
        inline SIMD8_64u operator+ (uint64_t a, SIMD8_64u const & b) { return b.add(a); }
        inline SIMD16_64u operator+ (uint64_t a, SIMD16_64u const & b) { return b.add(a); }

        inline SIMD1_8i operator+ (uint8_t a, SIMD1_8i const & b) { return b.add(a); }
        inline SIMD2_8i operator+ (uint8_t a, SIMD2_8i const & b) { return b.add(a); }
        inline SIMD4_8i operator+ (uint8_t a, SIMD4_8i const & b) { return b.add(a); }
        inline SIMD8_8i operator+ (uint8_t a, SIMD8_8i const & b) { return b.add(a); }
        inline SIMD16_8i operator+ (uint8_t a, SIMD16_8i const & b) { return b.add(a); }
        inline SIMD32_8i operator+ (uint8_t a, SIMD32_8i const & b) { return b.add(a); }
        inline SIMD64_8i operator+ (uint8_t a, SIMD64_8i const & b) { return b.add(a); }
        inline SIMD128_8i operator+ (uint8_t a, SIMD128_8i const & b) { return b.add(a); }

        inline SIMD1_16i operator+ (uint16_t a, SIMD1_16i const & b) { return b.add(a); }
        inline SIMD2_16i operator+ (uint16_t a, SIMD2_16i const & b) { return b.add(a); }
        inline SIMD4_16i operator+ (uint16_t a, SIMD4_16i const & b) { return b.add(a); }
        inline SIMD8_16i operator+ (uint16_t a, SIMD8_16i const & b) { return b.add(a); }
        inline SIMD16_16i operator+ (uint16_t a, SIMD16_16i const & b) { return b.add(a); }
        inline SIMD32_16i operator+ (uint16_t a, SIMD32_16i const & b) { return b.add(a); }
        inline SIMD64_16i operator+ (uint16_t a, SIMD64_16i const & b) { return b.add(a); }

        inline SIMD1_32i operator+ (uint32_t a, SIMD1_32i const & b) { return b.add(a); }
        inline SIMD2_32i operator+ (uint32_t a, SIMD2_32i const & b) { return b.add(a); }
        inline SIMD4_32i operator+ (uint32_t a, SIMD4_32i const & b) { return b.add(a); }
        inline SIMD8_32i operator+ (uint32_t a, SIMD8_32i const & b) { return b.add(a); }
        inline SIMD16_32i operator+ (uint32_t a, SIMD16_32i const & b) { return b.add(a); }
        inline SIMD32_32i operator+ (uint32_t a, SIMD32_32i const & b) { return b.add(a); }

        inline SIMD1_64i operator+ (uint64_t a, SIMD1_64i const & b) { return b.add(a); }
        inline SIMD2_64i operator+ (uint64_t a, SIMD2_64i const & b) { return b.add(a); }
        inline SIMD4_64i operator+ (uint64_t a, SIMD4_64i const & b) { return b.add(a); }
        inline SIMD8_64i operator+ (uint64_t a, SIMD8_64i const & b) { return b.add(a); }
        inline SIMD16_64i operator+ (uint64_t a, SIMD16_64i const & b) { return b.add(a); }

        inline SIMD1_32f operator+ (float a, SIMD1_32f const & b) { return b.add(a); }
        inline SIMD2_32f operator+ (float a, SIMD2_32f const & b) { return b.add(a); }
        inline SIMD4_32f operator+ (float a, SIMD4_32f const & b) { return b.add(a); }
        inline SIMD8_32f operator+ (float a, SIMD8_32f const & b) { return b.add(a); }
        inline SIMD16_32f operator+ (float a, SIMD16_32f const & b) { return b.add(a); }
        inline SIMD32_32f operator+ (float a, SIMD32_32f const & b) { return b.add(a); }

        inline SIMD1_64f operator+ (double a, SIMD1_64f const & b) { return b.add(a); }
        inline SIMD2_64f operator+ (double a, SIMD2_64f const & b) { return b.add(a); }
        inline SIMD4_64f operator+ (double a, SIMD4_64f const & b) { return b.add(a); }
        inline SIMD8_64f operator+ (double a, SIMD8_64f const & b) { return b.add(a); }
        inline SIMD16_64f operator+ (double a, SIMD16_64f const & b) { return b.add(a); }

        // SUBFROMS
        inline SIMD1_8u operator- (uint8_t a, SIMD1_8u const & b) { return b.subfrom(a); }
        inline SIMD2_8u operator- (uint8_t a, SIMD2_8u const & b) { return b.subfrom(a); }
        inline SIMD4_8u operator- (uint8_t a, SIMD4_8u const & b) { return b.subfrom(a); }
        inline SIMD8_8u operator- (uint8_t a, SIMD8_8u const & b) { return b.subfrom(a); }
        inline SIMD16_8u operator- (uint8_t a, SIMD16_8u const & b) { return b.subfrom(a); }
        inline SIMD32_8u operator- (uint8_t a, SIMD32_8u const & b) { return b.subfrom(a); }
        inline SIMD64_8u operator- (uint8_t a, SIMD64_8u const & b) { return b.subfrom(a); }
        inline SIMD128_8u operator- (uint8_t a, SIMD128_8u const & b) { return b.subfrom(a); }

        inline SIMD1_16u operator- (uint16_t a, SIMD1_16u const & b) { return b.subfrom(a); }
        inline SIMD2_16u operator- (uint16_t a, SIMD2_16u const & b) { return b.subfrom(a); }
        inline SIMD4_16u operator- (uint16_t a, SIMD4_16u const & b) { return b.subfrom(a); }
        inline SIMD8_16u operator- (uint16_t a, SIMD8_16u const & b) { return b.subfrom(a); }
        inline SIMD16_16u operator- (uint16_t a, SIMD16_16u const & b) { return b.subfrom(a); }
        inline SIMD32_16u operator- (uint16_t a, SIMD32_16u const & b) { return b.subfrom(a); }
        inline SIMD64_16u operator- (uint16_t a, SIMD64_16u const & b) { return b.subfrom(a); }

        inline SIMD1_32u operator- (uint32_t a, SIMD1_32u const & b) { return b.subfrom(a); }
        inline SIMD2_32u operator- (uint32_t a, SIMD2_32u const & b) { return b.subfrom(a); }
        inline SIMD4_32u operator- (uint32_t a, SIMD4_32u const & b) { return b.subfrom(a); }
        inline SIMD8_32u operator- (uint32_t a, SIMD8_32u const & b) { return b.subfrom(a); }
        inline SIMD16_32u operator- (uint32_t a, SIMD16_32u const & b) { return b.subfrom(a); }
        inline SIMD32_32u operator- (uint32_t a, SIMD32_32u const & b) { return b.subfrom(a); }

        inline SIMD1_64u operator- (uint64_t a, SIMD1_64u const & b) { return b.subfrom(a); }
        inline SIMD2_64u operator- (uint64_t a, SIMD2_64u const & b) { return b.subfrom(a); }
        inline SIMD4_64u operator- (uint64_t a, SIMD4_64u const & b) { return b.subfrom(a); }
        inline SIMD8_64u operator- (uint64_t a, SIMD8_64u const & b) { return b.subfrom(a); }
        inline SIMD16_64u operator- (uint64_t a, SIMD16_64u const & b) { return b.subfrom(a); }

        inline SIMD1_8i operator- (uint8_t a, SIMD1_8i const & b) { return b.subfrom(a); }
        inline SIMD2_8i operator- (uint8_t a, SIMD2_8i const & b) { return b.subfrom(a); }
        inline SIMD4_8i operator- (uint8_t a, SIMD4_8i const & b) { return b.subfrom(a); }
        inline SIMD8_8i operator- (uint8_t a, SIMD8_8i const & b) { return b.subfrom(a); }
        inline SIMD16_8i operator- (uint8_t a, SIMD16_8i const & b) { return b.subfrom(a); }
        inline SIMD32_8i operator- (uint8_t a, SIMD32_8i const & b) { return b.subfrom(a); }
        inline SIMD64_8i operator- (uint8_t a, SIMD64_8i const & b) { return b.subfrom(a); }
        inline SIMD128_8i operator- (uint8_t a, SIMD128_8i const & b) { return b.subfrom(a); }

        inline SIMD1_16i operator- (uint16_t a, SIMD1_16i const & b) { return b.subfrom(a); }
        inline SIMD2_16i operator- (uint16_t a, SIMD2_16i const & b) { return b.subfrom(a); }
        inline SIMD4_16i operator- (uint16_t a, SIMD4_16i const & b) { return b.subfrom(a); }
        inline SIMD8_16i operator- (uint16_t a, SIMD8_16i const & b) { return b.subfrom(a); }
        inline SIMD16_16i operator- (uint16_t a, SIMD16_16i const & b) { return b.subfrom(a); }
        inline SIMD32_16i operator- (uint16_t a, SIMD32_16i const & b) { return b.subfrom(a); }
        inline SIMD64_16i operator- (uint16_t a, SIMD64_16i const & b) { return b.subfrom(a); }

        inline SIMD1_32i operator- (uint32_t a, SIMD1_32i const & b) { return b.subfrom(a); }
        inline SIMD2_32i operator- (uint32_t a, SIMD2_32i const & b) { return b.subfrom(a); }
        inline SIMD4_32i operator- (uint32_t a, SIMD4_32i const & b) { return b.subfrom(a); }
        inline SIMD8_32i operator- (uint32_t a, SIMD8_32i const & b) { return b.subfrom(a); }
        inline SIMD16_32i operator- (uint32_t a, SIMD16_32i const & b) { return b.subfrom(a); }
        inline SIMD32_32i operator- (uint32_t a, SIMD32_32i const & b) { return b.subfrom(a); }

        inline SIMD1_64i operator- (uint64_t a, SIMD1_64i const & b) { return b.subfrom(a); }
        inline SIMD2_64i operator- (uint64_t a, SIMD2_64i const & b) { return b.subfrom(a); }
        inline SIMD4_64i operator- (uint64_t a, SIMD4_64i const & b) { return b.subfrom(a); }
        inline SIMD8_64i operator- (uint64_t a, SIMD8_64i const & b) { return b.subfrom(a); }
        inline SIMD16_64i operator- (uint64_t a, SIMD16_64i const & b) { return b.subfrom(a); }

        inline SIMD1_32f operator- (float a, SIMD1_32f const & b) { return b.subfrom(a); }
        inline SIMD2_32f operator- (float a, SIMD2_32f const & b) { return b.subfrom(a); }
        inline SIMD4_32f operator- (float a, SIMD4_32f const & b) { return b.subfrom(a); }
        inline SIMD8_32f operator- (float a, SIMD8_32f const & b) { return b.subfrom(a); }
        inline SIMD16_32f operator- (float a, SIMD16_32f const & b) { return b.subfrom(a); }
        inline SIMD32_32f operator- (float a, SIMD32_32f const & b) { return b.subfrom(a); }

        inline SIMD1_64f operator- (double a, SIMD1_64f const & b) { return b.subfrom(a); }
        inline SIMD2_64f operator- (double a, SIMD2_64f const & b) { return b.subfrom(a); }
        inline SIMD4_64f operator- (double a, SIMD4_64f const & b) { return b.subfrom(a); }
        inline SIMD8_64f operator- (double a, SIMD8_64f const & b) { return b.subfrom(a); }
        inline SIMD16_64f operator- (double a, SIMD16_64f const & b) { return b.subfrom(a); }

        // MULS
        inline SIMD1_8u operator* (uint8_t a, SIMD1_8u const & b) { return b.mul(a); }
        inline SIMD2_8u operator* (uint8_t a, SIMD2_8u const & b) { return b.mul(a); }
        inline SIMD4_8u operator* (uint8_t a, SIMD4_8u const & b) { return b.mul(a); }
        inline SIMD8_8u operator* (uint8_t a, SIMD8_8u const & b) { return b.mul(a); }
        inline SIMD16_8u operator* (uint8_t a, SIMD16_8u const & b) { return b.mul(a); }
        inline SIMD32_8u operator* (uint8_t a, SIMD32_8u const & b) { return b.mul(a); }
        inline SIMD64_8u operator* (uint8_t a, SIMD64_8u const & b) { return b.mul(a); }
        inline SIMD128_8u operator* (uint8_t a, SIMD128_8u const & b) { return b.mul(a); }

        inline SIMD1_16u operator* (uint16_t a, SIMD1_16u const & b) { return b.mul(a); }
        inline SIMD2_16u operator* (uint16_t a, SIMD2_16u const & b) { return b.mul(a); }
        inline SIMD4_16u operator* (uint16_t a, SIMD4_16u const & b) { return b.mul(a); }
        inline SIMD8_16u operator* (uint16_t a, SIMD8_16u const & b) { return b.mul(a); }
        inline SIMD16_16u operator* (uint16_t a, SIMD16_16u const & b) { return b.mul(a); }
        inline SIMD32_16u operator* (uint16_t a, SIMD32_16u const & b) { return b.mul(a); }
        inline SIMD64_16u operator* (uint16_t a, SIMD64_16u const & b) { return b.mul(a); }

        inline SIMD1_32u operator* (uint32_t a, SIMD1_32u const & b) { return b.mul(a); }
        inline SIMD2_32u operator* (uint32_t a, SIMD2_32u const & b) { return b.mul(a); }
        inline SIMD4_32u operator* (uint32_t a, SIMD4_32u const & b) { return b.mul(a); }
        inline SIMD8_32u operator* (uint32_t a, SIMD8_32u const & b) { return b.mul(a); }
        inline SIMD16_32u operator* (uint32_t a, SIMD16_32u const & b) { return b.mul(a); }
        inline SIMD32_32u operator* (uint32_t a, SIMD32_32u const & b) { return b.mul(a); }

        inline SIMD1_64u operator* (uint64_t a, SIMD1_64u const & b) { return b.mul(a); }
        inline SIMD2_64u operator* (uint64_t a, SIMD2_64u const & b) { return b.mul(a); }
        inline SIMD4_64u operator* (uint64_t a, SIMD4_64u const & b) { return b.mul(a); }
        inline SIMD8_64u operator* (uint64_t a, SIMD8_64u const & b) { return b.mul(a); }
        inline SIMD16_64u operator* (uint64_t a, SIMD16_64u const & b) { return b.mul(a); }

        inline SIMD1_8i operator* (uint8_t a, SIMD1_8i const & b) { return b.mul(a); }
        inline SIMD2_8i operator* (uint8_t a, SIMD2_8i const & b) { return b.mul(a); }
        inline SIMD4_8i operator* (uint8_t a, SIMD4_8i const & b) { return b.mul(a); }
        inline SIMD8_8i operator* (uint8_t a, SIMD8_8i const & b) { return b.mul(a); }
        inline SIMD16_8i operator* (uint8_t a, SIMD16_8i const & b) { return b.mul(a); }
        inline SIMD32_8i operator* (uint8_t a, SIMD32_8i const & b) { return b.mul(a); }
        inline SIMD64_8i operator* (uint8_t a, SIMD64_8i const & b) { return b.mul(a); }
        inline SIMD128_8i operator* (uint8_t a, SIMD128_8i const & b) { return b.mul(a); }

        inline SIMD1_16i operator* (uint16_t a, SIMD1_16i const & b) { return b.mul(a); }
        inline SIMD2_16i operator* (uint16_t a, SIMD2_16i const & b) { return b.mul(a); }
        inline SIMD4_16i operator* (uint16_t a, SIMD4_16i const & b) { return b.mul(a); }
        inline SIMD8_16i operator* (uint16_t a, SIMD8_16i const & b) { return b.mul(a); }
        inline SIMD16_16i operator* (uint16_t a, SIMD16_16i const & b) { return b.mul(a); }
        inline SIMD32_16i operator* (uint16_t a, SIMD32_16i const & b) { return b.mul(a); }
        inline SIMD64_16i operator* (uint16_t a, SIMD64_16i const & b) { return b.mul(a); }

        inline SIMD1_32i operator* (uint32_t a, SIMD1_32i const & b) { return b.mul(a); }
        inline SIMD2_32i operator* (uint32_t a, SIMD2_32i const & b) { return b.mul(a); }
        inline SIMD4_32i operator* (uint32_t a, SIMD4_32i const & b) { return b.mul(a); }
        inline SIMD8_32i operator* (uint32_t a, SIMD8_32i const & b) { return b.mul(a); }
        inline SIMD16_32i operator* (uint32_t a, SIMD16_32i const & b) { return b.mul(a); }
        inline SIMD32_32i operator* (uint32_t a, SIMD32_32i const & b) { return b.mul(a); }

        inline SIMD1_64i operator* (uint64_t a, SIMD1_64i const & b) { return b.mul(a); }
        inline SIMD2_64i operator* (uint64_t a, SIMD2_64i const & b) { return b.mul(a); }
        inline SIMD4_64i operator* (uint64_t a, SIMD4_64i const & b) { return b.mul(a); }
        inline SIMD8_64i operator* (uint64_t a, SIMD8_64i const & b) { return b.mul(a); }
        inline SIMD16_64i operator* (uint64_t a, SIMD16_64i const & b) { return b.mul(a); }

        inline SIMD1_32f operator* (float a, SIMD1_32f const & b) { return b.mul(a); }
        inline SIMD2_32f operator* (float a, SIMD2_32f const & b) { return b.mul(a); }
        inline SIMD4_32f operator* (float a, SIMD4_32f const & b) { return b.mul(a); }
        inline SIMD8_32f operator* (float a, SIMD8_32f const & b) { return b.mul(a); }
        inline SIMD16_32f operator* (float a, SIMD16_32f const & b) { return b.mul(a); }
        inline SIMD32_32f operator* (float a, SIMD32_32f const & b) { return b.mul(a); }

        inline SIMD1_64f operator* (double a, SIMD1_64f const & b) { return b.mul(a); }
        inline SIMD2_64f operator* (double a, SIMD2_64f const & b) { return b.mul(a); }
        inline SIMD4_64f operator* (double a, SIMD4_64f const & b) { return b.mul(a); }
        inline SIMD8_64f operator* (double a, SIMD8_64f const & b) { return b.mul(a); }
        inline SIMD16_64f operator* (double a, SIMD16_64f const & b) { return b.mul(a); }

        // RCPS
        inline SIMD1_8u operator/ (uint8_t a, SIMD1_8u const & b) { return b.rcp(a); }
        inline SIMD2_8u operator/ (uint8_t a, SIMD2_8u const & b) { return b.rcp(a); }
        inline SIMD4_8u operator/ (uint8_t a, SIMD4_8u const & b) { return b.rcp(a); }
        inline SIMD8_8u operator/ (uint8_t a, SIMD8_8u const & b) { return b.rcp(a); }
        inline SIMD16_8u operator/ (uint8_t a, SIMD16_8u const & b) { return b.rcp(a); }
        inline SIMD32_8u operator/ (uint8_t a, SIMD32_8u const & b) { return b.rcp(a); }
        inline SIMD64_8u operator/ (uint8_t a, SIMD64_8u const & b) { return b.rcp(a); }
        inline SIMD128_8u operator/ (uint8_t a, SIMD128_8u const & b) { return b.rcp(a); }

        inline SIMD1_16u operator/ (uint16_t a, SIMD1_16u const & b) { return b.rcp(a); }
        inline SIMD2_16u operator/ (uint16_t a, SIMD2_16u const & b) { return b.rcp(a); }
        inline SIMD4_16u operator/ (uint16_t a, SIMD4_16u const & b) { return b.rcp(a); }
        inline SIMD8_16u operator/ (uint16_t a, SIMD8_16u const & b) { return b.rcp(a); }
        inline SIMD16_16u operator/ (uint16_t a, SIMD16_16u const & b) { return b.rcp(a); }
        inline SIMD32_16u operator/ (uint16_t a, SIMD32_16u const & b) { return b.rcp(a); }
        inline SIMD64_16u operator/ (uint16_t a, SIMD64_16u const & b) { return b.rcp(a); }

        inline SIMD1_32u operator/ (uint32_t a, SIMD1_32u const & b) { return b.rcp(a); }
        inline SIMD2_32u operator/ (uint32_t a, SIMD2_32u const & b) { return b.rcp(a); }
        inline SIMD4_32u operator/ (uint32_t a, SIMD4_32u const & b) { return b.rcp(a); }
        inline SIMD8_32u operator/ (uint32_t a, SIMD8_32u const & b) { return b.rcp(a); }
        inline SIMD16_32u operator/ (uint32_t a, SIMD16_32u const & b) { return b.rcp(a); }
        inline SIMD32_32u operator/ (uint32_t a, SIMD32_32u const & b) { return b.rcp(a); }

        inline SIMD1_64u operator/ (uint64_t a, SIMD1_64u const & b) { return b.rcp(a); }
        inline SIMD2_64u operator/ (uint64_t a, SIMD2_64u const & b) { return b.rcp(a); }
        inline SIMD4_64u operator/ (uint64_t a, SIMD4_64u const & b) { return b.rcp(a); }
        inline SIMD8_64u operator/ (uint64_t a, SIMD8_64u const & b) { return b.rcp(a); }
        inline SIMD16_64u operator/ (uint64_t a, SIMD16_64u const & b) { return b.rcp(a); }

        inline SIMD1_8i operator/ (uint8_t a, SIMD1_8i const & b) { return b.rcp(a); }
        inline SIMD2_8i operator/ (uint8_t a, SIMD2_8i const & b) { return b.rcp(a); }
        inline SIMD4_8i operator/ (uint8_t a, SIMD4_8i const & b) { return b.rcp(a); }
        inline SIMD8_8i operator/ (uint8_t a, SIMD8_8i const & b) { return b.rcp(a); }
        inline SIMD16_8i operator/ (uint8_t a, SIMD16_8i const & b) { return b.rcp(a); }
        inline SIMD32_8i operator/ (uint8_t a, SIMD32_8i const & b) { return b.rcp(a); }
        inline SIMD64_8i operator/ (uint8_t a, SIMD64_8i const & b) { return b.rcp(a); }
        inline SIMD128_8i operator/ (uint8_t a, SIMD128_8i const & b) { return b.rcp(a); }

        inline SIMD1_16i operator/ (uint16_t a, SIMD1_16i const & b) { return b.rcp(a); }
        inline SIMD2_16i operator/ (uint16_t a, SIMD2_16i const & b) { return b.rcp(a); }
        inline SIMD4_16i operator/ (uint16_t a, SIMD4_16i const & b) { return b.rcp(a); }
        inline SIMD8_16i operator/ (uint16_t a, SIMD8_16i const & b) { return b.rcp(a); }
        inline SIMD16_16i operator/ (uint16_t a, SIMD16_16i const & b) { return b.rcp(a); }
        inline SIMD32_16i operator/ (uint16_t a, SIMD32_16i const & b) { return b.rcp(a); }
        inline SIMD64_16i operator/ (uint16_t a, SIMD64_16i const & b) { return b.rcp(a); }

        inline SIMD1_32i operator/ (uint32_t a, SIMD1_32i const & b) { return b.rcp(a); }
        inline SIMD2_32i operator/ (uint32_t a, SIMD2_32i const & b) { return b.rcp(a); }
        inline SIMD4_32i operator/ (uint32_t a, SIMD4_32i const & b) { return b.rcp(a); }
        inline SIMD8_32i operator/ (uint32_t a, SIMD8_32i const & b) { return b.rcp(a); }
        inline SIMD16_32i operator/ (uint32_t a, SIMD16_32i const & b) { return b.rcp(a); }
        inline SIMD32_32i operator/ (uint32_t a, SIMD32_32i const & b) { return b.rcp(a); }

        inline SIMD1_64i operator/ (uint64_t a, SIMD1_64i const & b) { return b.rcp(a); }
        inline SIMD2_64i operator/ (uint64_t a, SIMD2_64i const & b) { return b.rcp(a); }
        inline SIMD4_64i operator/ (uint64_t a, SIMD4_64i const & b) { return b.rcp(a); }
        inline SIMD8_64i operator/ (uint64_t a, SIMD8_64i const & b) { return b.rcp(a); }
        inline SIMD16_64i operator/ (uint64_t a, SIMD16_64i const & b) { return b.rcp(a); }

        inline SIMD1_32f operator/ (float a, SIMD1_32f const & b) { return b.rcp(a); }
        inline SIMD2_32f operator/ (float a, SIMD2_32f const & b) { return b.rcp(a); }
        inline SIMD4_32f operator/ (float a, SIMD4_32f const & b) { return b.rcp(a); }
        inline SIMD8_32f operator/ (float a, SIMD8_32f const & b) { return b.rcp(a); }
        inline SIMD16_32f operator/ (float a, SIMD16_32f const & b) { return b.rcp(a); }
        inline SIMD32_32f operator/ (float a, SIMD32_32f const & b) { return b.rcp(a); }

        inline SIMD1_64f operator/ (double a, SIMD1_64f const & b) { return b.rcp(a); }
        inline SIMD2_64f operator/ (double a, SIMD2_64f const & b) { return b.rcp(a); }
        inline SIMD4_64f operator/ (double a, SIMD4_64f const & b) { return b.rcp(a); }
        inline SIMD8_64f operator/ (double a, SIMD8_64f const & b) { return b.rcp(a); }
        inline SIMD16_64f operator/ (double a, SIMD16_64f const & b) { return b.rcp(a); }

        // BANDS
        inline SIMD1_8u operator& (uint8_t a, SIMD1_8u const & b) { return b.band(a); }
        inline SIMD2_8u operator& (uint8_t a, SIMD2_8u const & b) { return b.band(a); }
        inline SIMD4_8u operator& (uint8_t a, SIMD4_8u const & b) { return b.band(a); }
        inline SIMD8_8u operator& (uint8_t a, SIMD8_8u const & b) { return b.band(a); }
        inline SIMD16_8u operator& (uint8_t a, SIMD16_8u const & b) { return b.band(a); }
        inline SIMD32_8u operator& (uint8_t a, SIMD32_8u const & b) { return b.band(a); }
        inline SIMD64_8u operator& (uint8_t a, SIMD64_8u const & b) { return b.band(a); }
        inline SIMD128_8u operator& (uint8_t a, SIMD128_8u const & b) { return b.band(a); }

        inline SIMD1_16u operator& (uint16_t a, SIMD1_16u const & b) { return b.band(a); }
        inline SIMD2_16u operator& (uint16_t a, SIMD2_16u const & b) { return b.band(a); }
        inline SIMD4_16u operator& (uint16_t a, SIMD4_16u const & b) { return b.band(a); }
        inline SIMD8_16u operator& (uint16_t a, SIMD8_16u const & b) { return b.band(a); }
        inline SIMD16_16u operator& (uint16_t a, SIMD16_16u const & b) { return b.band(a); }
        inline SIMD32_16u operator& (uint16_t a, SIMD32_16u const & b) { return b.band(a); }
        inline SIMD64_16u operator& (uint16_t a, SIMD64_16u const & b) { return b.band(a); }

        inline SIMD1_32u operator& (uint32_t a, SIMD1_32u const & b) { return b.band(a); }
        inline SIMD2_32u operator& (uint32_t a, SIMD2_32u const & b) { return b.band(a); }
        inline SIMD4_32u operator& (uint32_t a, SIMD4_32u const & b) { return b.band(a); }
        inline SIMD8_32u operator& (uint32_t a, SIMD8_32u const & b) { return b.band(a); }
        inline SIMD16_32u operator& (uint32_t a, SIMD16_32u const & b) { return b.band(a); }
        inline SIMD32_32u operator& (uint32_t a, SIMD32_32u const & b) { return b.band(a); }

        inline SIMD1_64u operator& (uint64_t a, SIMD1_64u const & b) { return b.band(a); }
        inline SIMD2_64u operator& (uint64_t a, SIMD2_64u const & b) { return b.band(a); }
        inline SIMD4_64u operator& (uint64_t a, SIMD4_64u const & b) { return b.band(a); }
        inline SIMD8_64u operator& (uint64_t a, SIMD8_64u const & b) { return b.band(a); }
        inline SIMD16_64u operator& (uint64_t a, SIMD16_64u const & b) { return b.band(a); }

        inline SIMD1_8i operator& (uint8_t a, SIMD1_8i const & b) { return b.band(a); }
        inline SIMD2_8i operator& (uint8_t a, SIMD2_8i const & b) { return b.band(a); }
        inline SIMD4_8i operator& (uint8_t a, SIMD4_8i const & b) { return b.band(a); }
        inline SIMD8_8i operator& (uint8_t a, SIMD8_8i const & b) { return b.band(a); }
        inline SIMD16_8i operator& (uint8_t a, SIMD16_8i const & b) { return b.band(a); }
        inline SIMD32_8i operator& (uint8_t a, SIMD32_8i const & b) { return b.band(a); }
        inline SIMD64_8i operator& (uint8_t a, SIMD64_8i const & b) { return b.band(a); }
        inline SIMD128_8i operator& (uint8_t a, SIMD128_8i const & b) { return b.band(a); }

        inline SIMD1_16i operator& (uint16_t a, SIMD1_16i const & b) { return b.band(a); }
        inline SIMD2_16i operator& (uint16_t a, SIMD2_16i const & b) { return b.band(a); }
        inline SIMD4_16i operator& (uint16_t a, SIMD4_16i const & b) { return b.band(a); }
        inline SIMD8_16i operator& (uint16_t a, SIMD8_16i const & b) { return b.band(a); }
        inline SIMD16_16i operator& (uint16_t a, SIMD16_16i const & b) { return b.band(a); }
        inline SIMD32_16i operator& (uint16_t a, SIMD32_16i const & b) { return b.band(a); }
        inline SIMD64_16i operator& (uint16_t a, SIMD64_16i const & b) { return b.band(a); }

        inline SIMD1_32i operator& (uint32_t a, SIMD1_32i const & b) { return b.band(a); }
        inline SIMD2_32i operator& (uint32_t a, SIMD2_32i const & b) { return b.band(a); }
        inline SIMD4_32i operator& (uint32_t a, SIMD4_32i const & b) { return b.band(a); }
        inline SIMD8_32i operator& (uint32_t a, SIMD8_32i const & b) { return b.band(a); }
        inline SIMD16_32i operator& (uint32_t a, SIMD16_32i const & b) { return b.band(a); }
        inline SIMD32_32i operator& (uint32_t a, SIMD32_32i const & b) { return b.band(a); }

        inline SIMD1_64i operator& (uint64_t a, SIMD1_64i const & b) { return b.band(a); }
        inline SIMD2_64i operator& (uint64_t a, SIMD2_64i const & b) { return b.band(a); }
        inline SIMD4_64i operator& (uint64_t a, SIMD4_64i const & b) { return b.band(a); }
        inline SIMD8_64i operator& (uint64_t a, SIMD8_64i const & b) { return b.band(a); }
        inline SIMD16_64i operator& (uint64_t a, SIMD16_64i const & b) { return b.band(a); }

        // BORS
        inline SIMD1_8u operator| (uint8_t a, SIMD1_8u const & b) { return b.bor(a); }
        inline SIMD2_8u operator| (uint8_t a, SIMD2_8u const & b) { return b.bor(a); }
        inline SIMD4_8u operator| (uint8_t a, SIMD4_8u const & b) { return b.bor(a); }
        inline SIMD8_8u operator| (uint8_t a, SIMD8_8u const & b) { return b.bor(a); }
        inline SIMD16_8u operator| (uint8_t a, SIMD16_8u const & b) { return b.bor(a); }
        inline SIMD32_8u operator| (uint8_t a, SIMD32_8u const & b) { return b.bor(a); }
        inline SIMD64_8u operator| (uint8_t a, SIMD64_8u const & b) { return b.bor(a); }
        inline SIMD128_8u operator| (uint8_t a, SIMD128_8u const & b) { return b.bor(a); }

        inline SIMD1_16u operator| (uint16_t a, SIMD1_16u const & b) { return b.bor(a); }
        inline SIMD2_16u operator| (uint16_t a, SIMD2_16u const & b) { return b.bor(a); }
        inline SIMD4_16u operator| (uint16_t a, SIMD4_16u const & b) { return b.bor(a); }
        inline SIMD8_16u operator| (uint16_t a, SIMD8_16u const & b) { return b.bor(a); }
        inline SIMD16_16u operator| (uint16_t a, SIMD16_16u const & b) { return b.bor(a); }
        inline SIMD32_16u operator| (uint16_t a, SIMD32_16u const & b) { return b.bor(a); }
        inline SIMD64_16u operator| (uint16_t a, SIMD64_16u const & b) { return b.bor(a); }

        inline SIMD1_32u operator| (uint32_t a, SIMD1_32u const & b) { return b.bor(a); }
        inline SIMD2_32u operator| (uint32_t a, SIMD2_32u const & b) { return b.bor(a); }
        inline SIMD4_32u operator| (uint32_t a, SIMD4_32u const & b) { return b.bor(a); }
        inline SIMD8_32u operator| (uint32_t a, SIMD8_32u const & b) { return b.bor(a); }
        inline SIMD16_32u operator| (uint32_t a, SIMD16_32u const & b) { return b.bor(a); }
        inline SIMD32_32u operator| (uint32_t a, SIMD32_32u const & b) { return b.bor(a); }

        inline SIMD1_64u operator| (uint64_t a, SIMD1_64u const & b) { return b.bor(a); }
        inline SIMD2_64u operator| (uint64_t a, SIMD2_64u const & b) { return b.bor(a); }
        inline SIMD4_64u operator| (uint64_t a, SIMD4_64u const & b) { return b.bor(a); }
        inline SIMD8_64u operator| (uint64_t a, SIMD8_64u const & b) { return b.bor(a); }
        inline SIMD16_64u operator| (uint64_t a, SIMD16_64u const & b) { return b.bor(a); }

        inline SIMD1_8i operator| (uint8_t a, SIMD1_8i const & b) { return b.bor(a); }
        inline SIMD2_8i operator| (uint8_t a, SIMD2_8i const & b) { return b.bor(a); }
        inline SIMD4_8i operator| (uint8_t a, SIMD4_8i const & b) { return b.bor(a); }
        inline SIMD8_8i operator| (uint8_t a, SIMD8_8i const & b) { return b.bor(a); }
        inline SIMD16_8i operator| (uint8_t a, SIMD16_8i const & b) { return b.bor(a); }
        inline SIMD32_8i operator| (uint8_t a, SIMD32_8i const & b) { return b.bor(a); }
        inline SIMD64_8i operator| (uint8_t a, SIMD64_8i const & b) { return b.bor(a); }
        inline SIMD128_8i operator| (uint8_t a, SIMD128_8i const & b) { return b.bor(a); }

        inline SIMD1_16i operator| (uint16_t a, SIMD1_16i const & b) { return b.bor(a); }
        inline SIMD2_16i operator| (uint16_t a, SIMD2_16i const & b) { return b.bor(a); }
        inline SIMD4_16i operator| (uint16_t a, SIMD4_16i const & b) { return b.bor(a); }
        inline SIMD8_16i operator| (uint16_t a, SIMD8_16i const & b) { return b.bor(a); }
        inline SIMD16_16i operator| (uint16_t a, SIMD16_16i const & b) { return b.bor(a); }
        inline SIMD32_16i operator| (uint16_t a, SIMD32_16i const & b) { return b.bor(a); }
        inline SIMD64_16i operator| (uint16_t a, SIMD64_16i const & b) { return b.bor(a); }

        inline SIMD1_32i operator| (uint32_t a, SIMD1_32i const & b) { return b.bor(a); }
        inline SIMD2_32i operator| (uint32_t a, SIMD2_32i const & b) { return b.bor(a); }
        inline SIMD4_32i operator| (uint32_t a, SIMD4_32i const & b) { return b.bor(a); }
        inline SIMD8_32i operator| (uint32_t a, SIMD8_32i const & b) { return b.bor(a); }
        inline SIMD16_32i operator| (uint32_t a, SIMD16_32i const & b) { return b.bor(a); }
        inline SIMD32_32i operator| (uint32_t a, SIMD32_32i const & b) { return b.bor(a); }

        inline SIMD1_64i operator| (uint64_t a, SIMD1_64i const & b) { return b.bor(a); }
        inline SIMD2_64i operator| (uint64_t a, SIMD2_64i const & b) { return b.bor(a); }
        inline SIMD4_64i operator| (uint64_t a, SIMD4_64i const & b) { return b.bor(a); }
        inline SIMD8_64i operator| (uint64_t a, SIMD8_64i const & b) { return b.bor(a); }
        inline SIMD16_64i operator| (uint64_t a, SIMD16_64i const & b) { return b.bor(a); }

        // CMPEQS
        inline SIMDMask1 operator== (uint8_t a, SIMD1_8u const & b) { return b.cmpeq(a); }
        inline SIMDMask2 operator== (uint8_t a, SIMD2_8u const & b) { return b.cmpeq(a); }
        inline SIMDMask4 operator== (uint8_t a, SIMD4_8u const & b) { return b.cmpeq(a); }
        inline SIMDMask8 operator== (uint8_t a, SIMD8_8u const & b) { return b.cmpeq(a); }
        inline SIMDMask16 operator== (uint8_t a, SIMD16_8u const & b) { return b.cmpeq(a); }
        inline SIMDMask32 operator== (uint8_t a, SIMD32_8u const & b) { return b.cmpeq(a); }
        inline SIMDMask64 operator== (uint8_t a, SIMD64_8u const & b) { return b.cmpeq(a); }
        inline SIMDMask128 operator== (uint8_t a, SIMD128_8u const & b) { return b.cmpeq(a); }

        inline SIMDMask1 operator== (uint16_t a, SIMD1_16u const & b) { return b.cmpeq(a); }
        inline SIMDMask2 operator== (uint16_t a, SIMD2_16u const & b) { return b.cmpeq(a); }
        inline SIMDMask4 operator== (uint16_t a, SIMD4_16u const & b) { return b.cmpeq(a); }
        inline SIMDMask8 operator== (uint16_t a, SIMD8_16u const & b) { return b.cmpeq(a); }
        inline SIMDMask16 operator== (uint16_t a, SIMD16_16u const & b) { return b.cmpeq(a); }
        inline SIMDMask32 operator== (uint16_t a, SIMD32_16u const & b) { return b.cmpeq(a); }
        inline SIMDMask64 operator== (uint16_t a, SIMD64_16u const & b) { return b.cmpeq(a); }

        inline SIMDMask1 operator== (uint32_t a, SIMD1_32u const & b) { return b.cmpeq(a); }
        inline SIMDMask2 operator== (uint32_t a, SIMD2_32u const & b) { return b.cmpeq(a); }
        inline SIMDMask4 operator== (uint32_t a, SIMD4_32u const & b) { return b.cmpeq(a); }
        inline SIMDMask8 operator== (uint32_t a, SIMD8_32u const & b) { return b.cmpeq(a); }
        inline SIMDMask16 operator== (uint32_t a, SIMD16_32u const & b) { return b.cmpeq(a); }
        inline SIMDMask32 operator== (uint32_t a, SIMD32_32u const & b) { return b.cmpeq(a); }

        inline SIMDMask1 operator== (uint64_t a, SIMD1_64u const & b) { return b.cmpeq(a); }
        inline SIMDMask2 operator== (uint64_t a, SIMD2_64u const & b) { return b.cmpeq(a); }
        inline SIMDMask4 operator== (uint64_t a, SIMD4_64u const & b) { return b.cmpeq(a); }
        inline SIMDMask8 operator== (uint64_t a, SIMD8_64u const & b) { return b.cmpeq(a); }
        inline SIMDMask16 operator== (uint64_t a, SIMD16_64u const & b) { return b.cmpeq(a); }

        inline SIMDMask1 operator== (uint8_t a, SIMD1_8i const & b) { return b.cmpeq(a); }
        inline SIMDMask2 operator== (uint8_t a, SIMD2_8i const & b) { return b.cmpeq(a); }
        inline SIMDMask4 operator== (uint8_t a, SIMD4_8i const & b) { return b.cmpeq(a); }
        inline SIMDMask8 operator== (uint8_t a, SIMD8_8i const & b) { return b.cmpeq(a); }
        inline SIMDMask16 operator== (uint8_t a, SIMD16_8i const & b) { return b.cmpeq(a); }
        inline SIMDMask32 operator== (uint8_t a, SIMD32_8i const & b) { return b.cmpeq(a); }
        inline SIMDMask64 operator== (uint8_t a, SIMD64_8i const & b) { return b.cmpeq(a); }
        inline SIMDMask128 operator== (uint8_t a, SIMD128_8i const & b) { return b.cmpeq(a); }

        inline SIMDMask1 operator== (uint16_t a, SIMD1_16i const & b) { return b.cmpeq(a); }
        inline SIMDMask2 operator== (uint16_t a, SIMD2_16i const & b) { return b.cmpeq(a); }
        inline SIMDMask4 operator== (uint16_t a, SIMD4_16i const & b) { return b.cmpeq(a); }
        inline SIMDMask8 operator== (uint16_t a, SIMD8_16i const & b) { return b.cmpeq(a); }
        inline SIMDMask16 operator== (uint16_t a, SIMD16_16i const & b) { return b.cmpeq(a); }
        inline SIMDMask32 operator== (uint16_t a, SIMD32_16i const & b) { return b.cmpeq(a); }
        inline SIMDMask64 operator== (uint16_t a, SIMD64_16i const & b) { return b.cmpeq(a); }

        inline SIMDMask1 operator== (uint32_t a, SIMD1_32i const & b) { return b.cmpeq(a); }
        inline SIMDMask2 operator== (uint32_t a, SIMD2_32i const & b) { return b.cmpeq(a); }
        inline SIMDMask4 operator== (uint32_t a, SIMD4_32i const & b) { return b.cmpeq(a); }
        inline SIMDMask8 operator== (uint32_t a, SIMD8_32i const & b) { return b.cmpeq(a); }
        inline SIMDMask16 operator== (uint32_t a, SIMD16_32i const & b) { return b.cmpeq(a); }
        inline SIMDMask32 operator== (uint32_t a, SIMD32_32i const & b) { return b.cmpeq(a); }

        inline SIMDMask1 operator== (uint64_t a, SIMD1_64i const & b) { return b.cmpeq(a); }
        inline SIMDMask2 operator== (uint64_t a, SIMD2_64i const & b) { return b.cmpeq(a); }
        inline SIMDMask4 operator== (uint64_t a, SIMD4_64i const & b) { return b.cmpeq(a); }
        inline SIMDMask8 operator== (uint64_t a, SIMD8_64i const & b) { return b.cmpeq(a); }
        inline SIMDMask16 operator== (uint64_t a, SIMD16_64i const & b) { return b.cmpeq(a); }

        inline SIMDMask1 operator== (float a, SIMD1_32f const & b) { return b.cmpeq(a); }
        inline SIMDMask2 operator== (float a, SIMD2_32f const & b) { return b.cmpeq(a); }
        inline SIMDMask4 operator== (float a, SIMD4_32f const & b) { return b.cmpeq(a); }
        inline SIMDMask8 operator== (float a, SIMD8_32f const & b) { return b.cmpeq(a); }
        inline SIMDMask16 operator== (float a, SIMD16_32f const & b) { return b.cmpeq(a); }
        inline SIMDMask32 operator== (float a, SIMD32_32f const & b) { return b.cmpeq(a); }

        inline SIMDMask1 operator== (double a, SIMD1_64f const & b) { return b.cmpeq(a); }
        inline SIMDMask2 operator== (double a, SIMD2_64f const & b) { return b.cmpeq(a); }
        inline SIMDMask4 operator== (double a, SIMD4_64f const & b) { return b.cmpeq(a); }
        inline SIMDMask8 operator== (double a, SIMD8_64f const & b) { return b.cmpeq(a); }
        inline SIMDMask16 operator== (double a, SIMD16_64f const & b) { return b.cmpeq(a); }

        //CMPNEQ
        inline SIMDMask1 operator!= (uint8_t a, SIMD1_8u const & b) { return b.cmpne(a); }
        inline SIMDMask2 operator!= (uint8_t a, SIMD2_8u const & b) { return b.cmpne(a); }
        inline SIMDMask4 operator!= (uint8_t a, SIMD4_8u const & b) { return b.cmpne(a); }
        inline SIMDMask8 operator!= (uint8_t a, SIMD8_8u const & b) { return b.cmpne(a); }
        inline SIMDMask16 operator!= (uint8_t a, SIMD16_8u const & b) { return b.cmpne(a); }
        inline SIMDMask32 operator!= (uint8_t a, SIMD32_8u const & b) { return b.cmpne(a); }
        inline SIMDMask64 operator!= (uint8_t a, SIMD64_8u const & b) { return b.cmpne(a); }
        inline SIMDMask128 operator!= (uint8_t a, SIMD128_8u const & b) { return b.cmpne(a); }

        inline SIMDMask1 operator!= (uint16_t a, SIMD1_16u const & b) { return b.cmpne(a); }
        inline SIMDMask2 operator!= (uint16_t a, SIMD2_16u const & b) { return b.cmpne(a); }
        inline SIMDMask4 operator!= (uint16_t a, SIMD4_16u const & b) { return b.cmpne(a); }
        inline SIMDMask8 operator!= (uint16_t a, SIMD8_16u const & b) { return b.cmpne(a); }
        inline SIMDMask16 operator!= (uint16_t a, SIMD16_16u const & b) { return b.cmpne(a); }
        inline SIMDMask32 operator!= (uint16_t a, SIMD32_16u const & b) { return b.cmpne(a); }
        inline SIMDMask64 operator!= (uint16_t a, SIMD64_16u const & b) { return b.cmpne(a); }

        inline SIMDMask1 operator!= (uint32_t a, SIMD1_32u const & b) { return b.cmpne(a); }
        inline SIMDMask2 operator!= (uint32_t a, SIMD2_32u const & b) { return b.cmpne(a); }
        inline SIMDMask4 operator!= (uint32_t a, SIMD4_32u const & b) { return b.cmpne(a); }
        inline SIMDMask8 operator!= (uint32_t a, SIMD8_32u const & b) { return b.cmpne(a); }
        inline SIMDMask16 operator!= (uint32_t a, SIMD16_32u const & b) { return b.cmpne(a); }
        inline SIMDMask32 operator!= (uint32_t a, SIMD32_32u const & b) { return b.cmpne(a); }

        inline SIMDMask1 operator!= (uint64_t a, SIMD1_64u const & b) { return b.cmpne(a); }
        inline SIMDMask2 operator!= (uint64_t a, SIMD2_64u const & b) { return b.cmpne(a); }
        inline SIMDMask4 operator!= (uint64_t a, SIMD4_64u const & b) { return b.cmpne(a); }
        inline SIMDMask8 operator!= (uint64_t a, SIMD8_64u const & b) { return b.cmpne(a); }
        inline SIMDMask16 operator!= (uint64_t a, SIMD16_64u const & b) { return b.cmpne(a); }

        inline SIMDMask1 operator!= (uint8_t a, SIMD1_8i const & b) { return b.cmpne(a); }
        inline SIMDMask2 operator!= (uint8_t a, SIMD2_8i const & b) { return b.cmpne(a); }
        inline SIMDMask4 operator!= (uint8_t a, SIMD4_8i const & b) { return b.cmpne(a); }
        inline SIMDMask8 operator!= (uint8_t a, SIMD8_8i const & b) { return b.cmpne(a); }
        inline SIMDMask16 operator!= (uint8_t a, SIMD16_8i const & b) { return b.cmpne(a); }
        inline SIMDMask32 operator!= (uint8_t a, SIMD32_8i const & b) { return b.cmpne(a); }
        inline SIMDMask64 operator!= (uint8_t a, SIMD64_8i const & b) { return b.cmpne(a); }
        inline SIMDMask128 operator!= (uint8_t a, SIMD128_8i const & b) { return b.cmpne(a); }

        inline SIMDMask1 operator!= (uint16_t a, SIMD1_16i const & b) { return b.cmpne(a); }
        inline SIMDMask2 operator!= (uint16_t a, SIMD2_16i const & b) { return b.cmpne(a); }
        inline SIMDMask4 operator!= (uint16_t a, SIMD4_16i const & b) { return b.cmpne(a); }
        inline SIMDMask8 operator!= (uint16_t a, SIMD8_16i const & b) { return b.cmpne(a); }
        inline SIMDMask16 operator!= (uint16_t a, SIMD16_16i const & b) { return b.cmpne(a); }
        inline SIMDMask32 operator!= (uint16_t a, SIMD32_16i const & b) { return b.cmpne(a); }
        inline SIMDMask64 operator!= (uint16_t a, SIMD64_16i const & b) { return b.cmpne(a); }

        inline SIMDMask1 operator!= (uint32_t a, SIMD1_32i const & b) { return b.cmpne(a); }
        inline SIMDMask2 operator!= (uint32_t a, SIMD2_32i const & b) { return b.cmpne(a); }
        inline SIMDMask4 operator!= (uint32_t a, SIMD4_32i const & b) { return b.cmpne(a); }
        inline SIMDMask8 operator!= (uint32_t a, SIMD8_32i const & b) { return b.cmpne(a); }
        inline SIMDMask16 operator!= (uint32_t a, SIMD16_32i const & b) { return b.cmpne(a); }
        inline SIMDMask32 operator!= (uint32_t a, SIMD32_32i const & b) { return b.cmpne(a); }

        inline SIMDMask1 operator!= (uint64_t a, SIMD1_64i const & b) { return b.cmpne(a); }
        inline SIMDMask2 operator!= (uint64_t a, SIMD2_64i const & b) { return b.cmpne(a); }
        inline SIMDMask4 operator!= (uint64_t a, SIMD4_64i const & b) { return b.cmpne(a); }
        inline SIMDMask8 operator!= (uint64_t a, SIMD8_64i const & b) { return b.cmpne(a); }
        inline SIMDMask16 operator!= (uint64_t a, SIMD16_64i const & b) { return b.cmpne(a); }

        inline SIMDMask1 operator!= (float a, SIMD1_32f const & b) { return b.cmpne(a); }
        inline SIMDMask2 operator!= (float a, SIMD2_32f const & b) { return b.cmpne(a); }
        inline SIMDMask4 operator!= (float a, SIMD4_32f const & b) { return b.cmpne(a); }
        inline SIMDMask8 operator!= (float a, SIMD8_32f const & b) { return b.cmpne(a); }
        inline SIMDMask16 operator!= (float a, SIMD16_32f const & b) { return b.cmpne(a); }
        inline SIMDMask32 operator!= (float a, SIMD32_32f const & b) { return b.cmpne(a); }

        inline SIMDMask1 operator!= (double a, SIMD1_64f const & b) { return b.cmpne(a); }
        inline SIMDMask2 operator!= (double a, SIMD2_64f const & b) { return b.cmpne(a); }
        inline SIMDMask4 operator!= (double a, SIMD4_64f const & b) { return b.cmpne(a); }
        inline SIMDMask8 operator!= (double a, SIMD8_64f const & b) { return b.cmpne(a); }
        inline SIMDMask16 operator!= (double a, SIMD16_64f const & b) { return b.cmpne(a); }

        //CMPGTS
        inline SIMDMask1 operator> (uint8_t a, SIMD1_8u const & b) { return b.cmplt(a); }
        inline SIMDMask2 operator> (uint8_t a, SIMD2_8u const & b) { return b.cmplt(a); }
        inline SIMDMask4 operator> (uint8_t a, SIMD4_8u const & b) { return b.cmplt(a); }
        inline SIMDMask8 operator> (uint8_t a, SIMD8_8u const & b) { return b.cmplt(a); }
        inline SIMDMask16 operator> (uint8_t a, SIMD16_8u const & b) { return b.cmplt(a); }
        inline SIMDMask32 operator> (uint8_t a, SIMD32_8u const & b) { return b.cmplt(a); }
        inline SIMDMask64 operator> (uint8_t a, SIMD64_8u const & b) { return b.cmplt(a); }
        inline SIMDMask128 operator> (uint8_t a, SIMD128_8u const & b) { return b.cmplt(a); }

        inline SIMDMask1 operator> (uint16_t a, SIMD1_16u const & b) { return b.cmplt(a); }
        inline SIMDMask2 operator> (uint16_t a, SIMD2_16u const & b) { return b.cmplt(a); }
        inline SIMDMask4 operator> (uint16_t a, SIMD4_16u const & b) { return b.cmplt(a); }
        inline SIMDMask8 operator> (uint16_t a, SIMD8_16u const & b) { return b.cmplt(a); }
        inline SIMDMask16 operator> (uint16_t a, SIMD16_16u const & b) { return b.cmplt(a); }
        inline SIMDMask32 operator> (uint16_t a, SIMD32_16u const & b) { return b.cmplt(a); }
        inline SIMDMask64 operator> (uint16_t a, SIMD64_16u const & b) { return b.cmplt(a); }

        inline SIMDMask1 operator> (uint32_t a, SIMD1_32u const & b) { return b.cmplt(a); }
        inline SIMDMask2 operator> (uint32_t a, SIMD2_32u const & b) { return b.cmplt(a); }
        inline SIMDMask4 operator> (uint32_t a, SIMD4_32u const & b) { return b.cmplt(a); }
        inline SIMDMask8 operator> (uint32_t a, SIMD8_32u const & b) { return b.cmplt(a); }
        inline SIMDMask16 operator> (uint32_t a, SIMD16_32u const & b) { return b.cmplt(a); }
        inline SIMDMask32 operator> (uint32_t a, SIMD32_32u const & b) { return b.cmplt(a); }

        inline SIMDMask1 operator> (uint64_t a, SIMD1_64u const & b) { return b.cmplt(a); }
        inline SIMDMask2 operator> (uint64_t a, SIMD2_64u const & b) { return b.cmplt(a); }
        inline SIMDMask4 operator> (uint64_t a, SIMD4_64u const & b) { return b.cmplt(a); }
        inline SIMDMask8 operator> (uint64_t a, SIMD8_64u const & b) { return b.cmplt(a); }
        inline SIMDMask16 operator> (uint64_t a, SIMD16_64u const & b) { return b.cmplt(a); }

        inline SIMDMask1 operator> (uint8_t a, SIMD1_8i const & b) { return b.cmplt(a); }
        inline SIMDMask2 operator> (uint8_t a, SIMD2_8i const & b) { return b.cmplt(a); }
        inline SIMDMask4 operator> (uint8_t a, SIMD4_8i const & b) { return b.cmplt(a); }
        inline SIMDMask8 operator> (uint8_t a, SIMD8_8i const & b) { return b.cmplt(a); }
        inline SIMDMask16 operator> (uint8_t a, SIMD16_8i const & b) { return b.cmplt(a); }
        inline SIMDMask32 operator> (uint8_t a, SIMD32_8i const & b) { return b.cmplt(a); }
        inline SIMDMask64 operator> (uint8_t a, SIMD64_8i const & b) { return b.cmplt(a); }
        inline SIMDMask128 operator> (uint8_t a, SIMD128_8i const & b) { return b.cmplt(a); }

        inline SIMDMask1 operator> (uint16_t a, SIMD1_16i const & b) { return b.cmplt(a); }
        inline SIMDMask2 operator> (uint16_t a, SIMD2_16i const & b) { return b.cmplt(a); }
        inline SIMDMask4 operator> (uint16_t a, SIMD4_16i const & b) { return b.cmplt(a); }
        inline SIMDMask8 operator> (uint16_t a, SIMD8_16i const & b) { return b.cmplt(a); }
        inline SIMDMask16 operator> (uint16_t a, SIMD16_16i const & b) { return b.cmplt(a); }
        inline SIMDMask32 operator> (uint16_t a, SIMD32_16i const & b) { return b.cmplt(a); }
        inline SIMDMask64 operator> (uint16_t a, SIMD64_16i const & b) { return b.cmplt(a); }

        inline SIMDMask1 operator> (uint32_t a, SIMD1_32i const & b) { return b.cmplt(a); }
        inline SIMDMask2 operator> (uint32_t a, SIMD2_32i const & b) { return b.cmplt(a); }
        inline SIMDMask4 operator> (uint32_t a, SIMD4_32i const & b) { return b.cmplt(a); }
        inline SIMDMask8 operator> (uint32_t a, SIMD8_32i const & b) { return b.cmplt(a); }
        inline SIMDMask16 operator> (uint32_t a, SIMD16_32i const & b) { return b.cmplt(a); }
        inline SIMDMask32 operator> (uint32_t a, SIMD32_32i const & b) { return b.cmplt(a); }

        inline SIMDMask1 operator> (uint64_t a, SIMD1_64i const & b) { return b.cmplt(a); }
        inline SIMDMask2 operator> (uint64_t a, SIMD2_64i const & b) { return b.cmplt(a); }
        inline SIMDMask4 operator> (uint64_t a, SIMD4_64i const & b) { return b.cmplt(a); }
        inline SIMDMask8 operator> (uint64_t a, SIMD8_64i const & b) { return b.cmplt(a); }
        inline SIMDMask16 operator> (uint64_t a, SIMD16_64i const & b) { return b.cmplt(a); }

        inline SIMDMask1 operator> (float a, SIMD1_32f const & b) { return b.cmplt(a); }
        inline SIMDMask2 operator> (float a, SIMD2_32f const & b) { return b.cmplt(a); }
        inline SIMDMask4 operator> (float a, SIMD4_32f const & b) { return b.cmplt(a); }
        inline SIMDMask8 operator> (float a, SIMD8_32f const & b) { return b.cmplt(a); }
        inline SIMDMask16 operator> (float a, SIMD16_32f const & b) { return b.cmplt(a); }
        inline SIMDMask32 operator> (float a, SIMD32_32f const & b) { return b.cmplt(a); }

        inline SIMDMask1 operator> (double a, SIMD1_64f const & b) { return b.cmplt(a); }
        inline SIMDMask2 operator> (double a, SIMD2_64f const & b) { return b.cmplt(a); }
        inline SIMDMask4 operator> (double a, SIMD4_64f const & b) { return b.cmplt(a); }
        inline SIMDMask8 operator> (double a, SIMD8_64f const & b) { return b.cmplt(a); }
        inline SIMDMask16 operator> (double a, SIMD16_64f const & b) { return b.cmplt(a); }

        //CMPLT
        inline SIMDMask1 operator< (uint8_t a, SIMD1_8u const & b) { return b.cmpgt(a); }
        inline SIMDMask2 operator< (uint8_t a, SIMD2_8u const & b) { return b.cmpgt(a); }
        inline SIMDMask4 operator< (uint8_t a, SIMD4_8u const & b) { return b.cmpgt(a); }
        inline SIMDMask8 operator< (uint8_t a, SIMD8_8u const & b) { return b.cmpgt(a); }
        inline SIMDMask16 operator< (uint8_t a, SIMD16_8u const & b) { return b.cmpgt(a); }
        inline SIMDMask32 operator< (uint8_t a, SIMD32_8u const & b) { return b.cmpgt(a); }
        inline SIMDMask64 operator< (uint8_t a, SIMD64_8u const & b) { return b.cmpgt(a); }
        inline SIMDMask128 operator< (uint8_t a, SIMD128_8u const & b) { return b.cmpgt(a); }

        inline SIMDMask1 operator< (uint16_t a, SIMD1_16u const & b) { return b.cmpgt(a); }
        inline SIMDMask2 operator< (uint16_t a, SIMD2_16u const & b) { return b.cmpgt(a); }
        inline SIMDMask4 operator< (uint16_t a, SIMD4_16u const & b) { return b.cmpgt(a); }
        inline SIMDMask8 operator< (uint16_t a, SIMD8_16u const & b) { return b.cmpgt(a); }
        inline SIMDMask16 operator< (uint16_t a, SIMD16_16u const & b) { return b.cmpgt(a); }
        inline SIMDMask32 operator< (uint16_t a, SIMD32_16u const & b) { return b.cmpgt(a); }
        inline SIMDMask64 operator< (uint16_t a, SIMD64_16u const & b) { return b.cmpgt(a); }

        inline SIMDMask1 operator< (uint32_t a, SIMD1_32u const & b) { return b.cmpgt(a); }
        inline SIMDMask2 operator< (uint32_t a, SIMD2_32u const & b) { return b.cmpgt(a); }
        inline SIMDMask4 operator< (uint32_t a, SIMD4_32u const & b) { return b.cmpgt(a); }
        inline SIMDMask8 operator< (uint32_t a, SIMD8_32u const & b) { return b.cmpgt(a); }
        inline SIMDMask16 operator< (uint32_t a, SIMD16_32u const & b) { return b.cmpgt(a); }
        inline SIMDMask32 operator< (uint32_t a, SIMD32_32u const & b) { return b.cmpgt(a); }

        inline SIMDMask1 operator< (uint64_t a, SIMD1_64u const & b) { return b.cmpgt(a); }
        inline SIMDMask2 operator< (uint64_t a, SIMD2_64u const & b) { return b.cmpgt(a); }
        inline SIMDMask4 operator< (uint64_t a, SIMD4_64u const & b) { return b.cmpgt(a); }
        inline SIMDMask8 operator< (uint64_t a, SIMD8_64u const & b) { return b.cmpgt(a); }
        inline SIMDMask16 operator< (uint64_t a, SIMD16_64u const & b) { return b.cmpgt(a); }

        inline SIMDMask1 operator< (uint8_t a, SIMD1_8i const & b) { return b.cmpgt(a); }
        inline SIMDMask2 operator< (uint8_t a, SIMD2_8i const & b) { return b.cmpgt(a); }
        inline SIMDMask4 operator< (uint8_t a, SIMD4_8i const & b) { return b.cmpgt(a); }
        inline SIMDMask8 operator< (uint8_t a, SIMD8_8i const & b) { return b.cmpgt(a); }
        inline SIMDMask16 operator< (uint8_t a, SIMD16_8i const & b) { return b.cmpgt(a); }
        inline SIMDMask32 operator< (uint8_t a, SIMD32_8i const & b) { return b.cmpgt(a); }
        inline SIMDMask64 operator< (uint8_t a, SIMD64_8i const & b) { return b.cmpgt(a); }
        inline SIMDMask128 operator< (uint8_t a, SIMD128_8i const & b) { return b.cmpgt(a); }

        inline SIMDMask1 operator< (uint16_t a, SIMD1_16i const & b) { return b.cmpgt(a); }
        inline SIMDMask2 operator< (uint16_t a, SIMD2_16i const & b) { return b.cmpgt(a); }
        inline SIMDMask4 operator< (uint16_t a, SIMD4_16i const & b) { return b.cmpgt(a); }
        inline SIMDMask8 operator< (uint16_t a, SIMD8_16i const & b) { return b.cmpgt(a); }
        inline SIMDMask16 operator< (uint16_t a, SIMD16_16i const & b) { return b.cmpgt(a); }
        inline SIMDMask32 operator< (uint16_t a, SIMD32_16i const & b) { return b.cmpgt(a); }
        inline SIMDMask64 operator< (uint16_t a, SIMD64_16i const & b) { return b.cmpgt(a); }

        inline SIMDMask1 operator< (uint32_t a, SIMD1_32i const & b) { return b.cmpgt(a); }
        inline SIMDMask2 operator< (uint32_t a, SIMD2_32i const & b) { return b.cmpgt(a); }
        inline SIMDMask4 operator< (uint32_t a, SIMD4_32i const & b) { return b.cmpgt(a); }
        inline SIMDMask8 operator< (uint32_t a, SIMD8_32i const & b) { return b.cmpgt(a); }
        inline SIMDMask16 operator< (uint32_t a, SIMD16_32i const & b) { return b.cmpgt(a); }
        inline SIMDMask32 operator< (uint32_t a, SIMD32_32i const & b) { return b.cmpgt(a); }

        inline SIMDMask1 operator< (uint64_t a, SIMD1_64i const & b) { return b.cmpgt(a); }
        inline SIMDMask2 operator< (uint64_t a, SIMD2_64i const & b) { return b.cmpgt(a); }
        inline SIMDMask4 operator< (uint64_t a, SIMD4_64i const & b) { return b.cmpgt(a); }
        inline SIMDMask8 operator< (uint64_t a, SIMD8_64i const & b) { return b.cmpgt(a); }
        inline SIMDMask16 operator< (uint64_t a, SIMD16_64i const & b) { return b.cmpgt(a); }

        inline SIMDMask1 operator< (float a, SIMD1_32f const & b) { return b.cmpgt(a); }
        inline SIMDMask2 operator< (float a, SIMD2_32f const & b) { return b.cmpgt(a); }
        inline SIMDMask4 operator< (float a, SIMD4_32f const & b) { return b.cmpgt(a); }
        inline SIMDMask8 operator< (float a, SIMD8_32f const & b) { return b.cmpgt(a); }
        inline SIMDMask16 operator< (float a, SIMD16_32f const & b) { return b.cmpgt(a); }
        inline SIMDMask32 operator< (float a, SIMD32_32f const & b) { return b.cmpgt(a); }

        inline SIMDMask1 operator< (double a, SIMD1_64f const & b) { return b.cmpgt(a); }
        inline SIMDMask2 operator< (double a, SIMD2_64f const & b) { return b.cmpgt(a); }
        inline SIMDMask4 operator< (double a, SIMD4_64f const & b) { return b.cmpgt(a); }
        inline SIMDMask8 operator< (double a, SIMD8_64f const & b) { return b.cmpgt(a); }
        inline SIMDMask16 operator< (double a, SIMD16_64f const & b) { return b.cmpgt(a); }

        //CMPGES
        inline SIMDMask1 operator>= (uint8_t a, SIMD1_8u const & b) { return b.cmple(a); }
        inline SIMDMask2 operator>= (uint8_t a, SIMD2_8u const & b) { return b.cmple(a); }
        inline SIMDMask4 operator>= (uint8_t a, SIMD4_8u const & b) { return b.cmple(a); }
        inline SIMDMask8 operator>= (uint8_t a, SIMD8_8u const & b) { return b.cmple(a); }
        inline SIMDMask16 operator>= (uint8_t a, SIMD16_8u const & b) { return b.cmple(a); }
        inline SIMDMask32 operator>= (uint8_t a, SIMD32_8u const & b) { return b.cmple(a); }
        inline SIMDMask64 operator>= (uint8_t a, SIMD64_8u const & b) { return b.cmple(a); }
        inline SIMDMask128 operator>= (uint8_t a, SIMD128_8u const & b) { return b.cmple(a); }

        inline SIMDMask1 operator>= (uint16_t a, SIMD1_16u const & b) { return b.cmple(a); }
        inline SIMDMask2 operator>= (uint16_t a, SIMD2_16u const & b) { return b.cmple(a); }
        inline SIMDMask4 operator>= (uint16_t a, SIMD4_16u const & b) { return b.cmple(a); }
        inline SIMDMask8 operator>= (uint16_t a, SIMD8_16u const & b) { return b.cmple(a); }
        inline SIMDMask16 operator>= (uint16_t a, SIMD16_16u const & b) { return b.cmple(a); }
        inline SIMDMask32 operator>= (uint16_t a, SIMD32_16u const & b) { return b.cmple(a); }
        inline SIMDMask64 operator>= (uint16_t a, SIMD64_16u const & b) { return b.cmple(a); }

        inline SIMDMask1 operator>= (uint32_t a, SIMD1_32u const & b) { return b.cmple(a); }
        inline SIMDMask2 operator>= (uint32_t a, SIMD2_32u const & b) { return b.cmple(a); }
        inline SIMDMask4 operator>= (uint32_t a, SIMD4_32u const & b) { return b.cmple(a); }
        inline SIMDMask8 operator>= (uint32_t a, SIMD8_32u const & b) { return b.cmple(a); }
        inline SIMDMask16 operator>= (uint32_t a, SIMD16_32u const & b) { return b.cmple(a); }
        inline SIMDMask32 operator>= (uint32_t a, SIMD32_32u const & b) { return b.cmple(a); }

        inline SIMDMask1 operator>= (uint64_t a, SIMD1_64u const & b) { return b.cmple(a); }
        inline SIMDMask2 operator>= (uint64_t a, SIMD2_64u const & b) { return b.cmple(a); }
        inline SIMDMask4 operator>= (uint64_t a, SIMD4_64u const & b) { return b.cmple(a); }
        inline SIMDMask8 operator>= (uint64_t a, SIMD8_64u const & b) { return b.cmple(a); }
        inline SIMDMask16 operator>= (uint64_t a, SIMD16_64u const & b) { return b.cmple(a); }

        inline SIMDMask1 operator>= (uint8_t a, SIMD1_8i const & b) { return b.cmple(a); }
        inline SIMDMask2 operator>= (uint8_t a, SIMD2_8i const & b) { return b.cmple(a); }
        inline SIMDMask4 operator>= (uint8_t a, SIMD4_8i const & b) { return b.cmple(a); }
        inline SIMDMask8 operator>= (uint8_t a, SIMD8_8i const & b) { return b.cmple(a); }
        inline SIMDMask16 operator>= (uint8_t a, SIMD16_8i const & b) { return b.cmple(a); }
        inline SIMDMask32 operator>= (uint8_t a, SIMD32_8i const & b) { return b.cmple(a); }
        inline SIMDMask64 operator>= (uint8_t a, SIMD64_8i const & b) { return b.cmple(a); }
        inline SIMDMask128 operator>= (uint8_t a, SIMD128_8i const & b) { return b.cmple(a); }

        inline SIMDMask1 operator>= (uint16_t a, SIMD1_16i const & b) { return b.cmple(a); }
        inline SIMDMask2 operator>= (uint16_t a, SIMD2_16i const & b) { return b.cmple(a); }
        inline SIMDMask4 operator>= (uint16_t a, SIMD4_16i const & b) { return b.cmple(a); }
        inline SIMDMask8 operator>= (uint16_t a, SIMD8_16i const & b) { return b.cmple(a); }
        inline SIMDMask16 operator>= (uint16_t a, SIMD16_16i const & b) { return b.cmple(a); }
        inline SIMDMask32 operator>= (uint16_t a, SIMD32_16i const & b) { return b.cmple(a); }
        inline SIMDMask64 operator>= (uint16_t a, SIMD64_16i const & b) { return b.cmple(a); }

        inline SIMDMask1 operator>= (uint32_t a, SIMD1_32i const & b) { return b.cmple(a); }
        inline SIMDMask2 operator>= (uint32_t a, SIMD2_32i const & b) { return b.cmple(a); }
        inline SIMDMask4 operator>= (uint32_t a, SIMD4_32i const & b) { return b.cmple(a); }
        inline SIMDMask8 operator>= (uint32_t a, SIMD8_32i const & b) { return b.cmple(a); }
        inline SIMDMask16 operator>= (uint32_t a, SIMD16_32i const & b) { return b.cmple(a); }
        inline SIMDMask32 operator>= (uint32_t a, SIMD32_32i const & b) { return b.cmple(a); }

        inline SIMDMask1 operator>= (uint64_t a, SIMD1_64i const & b) { return b.cmple(a); }
        inline SIMDMask2 operator>= (uint64_t a, SIMD2_64i const & b) { return b.cmple(a); }
        inline SIMDMask4 operator>= (uint64_t a, SIMD4_64i const & b) { return b.cmple(a); }
        inline SIMDMask8 operator>= (uint64_t a, SIMD8_64i const & b) { return b.cmple(a); }
        inline SIMDMask16 operator>= (uint64_t a, SIMD16_64i const & b) { return b.cmple(a); }

        inline SIMDMask1 operator>= (float a, SIMD1_32f const & b) { return b.cmple(a); }
        inline SIMDMask2 operator>= (float a, SIMD2_32f const & b) { return b.cmple(a); }
        inline SIMDMask4 operator>= (float a, SIMD4_32f const & b) { return b.cmple(a); }
        inline SIMDMask8 operator>= (float a, SIMD8_32f const & b) { return b.cmple(a); }
        inline SIMDMask16 operator>= (float a, SIMD16_32f const & b) { return b.cmple(a); }
        inline SIMDMask32 operator>= (float a, SIMD32_32f const & b) { return b.cmple(a); }

        inline SIMDMask1 operator>= (double a, SIMD1_64f const & b) { return b.cmple(a); }
        inline SIMDMask2 operator>= (double a, SIMD2_64f const & b) { return b.cmple(a); }
        inline SIMDMask4 operator>= (double a, SIMD4_64f const & b) { return b.cmple(a); }
        inline SIMDMask8 operator>= (double a, SIMD8_64f const & b) { return b.cmple(a); }
        inline SIMDMask16 operator>= (double a, SIMD16_64f const & b) { return b.cmple(a); }

        //CMPLES
        inline SIMDMask1 operator<= (uint8_t a, SIMD1_8u const & b) { return b.cmpge(a); }
        inline SIMDMask2 operator<= (uint8_t a, SIMD2_8u const & b) { return b.cmpge(a); }
        inline SIMDMask4 operator<= (uint8_t a, SIMD4_8u const & b) { return b.cmpge(a); }
        inline SIMDMask8 operator<= (uint8_t a, SIMD8_8u const & b) { return b.cmpge(a); }
        inline SIMDMask16 operator<= (uint8_t a, SIMD16_8u const & b) { return b.cmpge(a); }
        inline SIMDMask32 operator<= (uint8_t a, SIMD32_8u const & b) { return b.cmpge(a); }
        inline SIMDMask64 operator<= (uint8_t a, SIMD64_8u const & b) { return b.cmpge(a); }
        inline SIMDMask128 operator<= (uint8_t a, SIMD128_8u const & b) { return b.cmpge(a); }

        inline SIMDMask1 operator<= (uint16_t a, SIMD1_16u const & b) { return b.cmpge(a); }
        inline SIMDMask2 operator<= (uint16_t a, SIMD2_16u const & b) { return b.cmpge(a); }
        inline SIMDMask4 operator<= (uint16_t a, SIMD4_16u const & b) { return b.cmpge(a); }
        inline SIMDMask8 operator<= (uint16_t a, SIMD8_16u const & b) { return b.cmpge(a); }
        inline SIMDMask16 operator<= (uint16_t a, SIMD16_16u const & b) { return b.cmpge(a); }
        inline SIMDMask32 operator<= (uint16_t a, SIMD32_16u const & b) { return b.cmpge(a); }
        inline SIMDMask64 operator<= (uint16_t a, SIMD64_16u const & b) { return b.cmpge(a); }

        inline SIMDMask1 operator<= (uint32_t a, SIMD1_32u const & b) { return b.cmpge(a); }
        inline SIMDMask2 operator<= (uint32_t a, SIMD2_32u const & b) { return b.cmpge(a); }
        inline SIMDMask4 operator<= (uint32_t a, SIMD4_32u const & b) { return b.cmpge(a); }
        inline SIMDMask8 operator<= (uint32_t a, SIMD8_32u const & b) { return b.cmpge(a); }
        inline SIMDMask16 operator<= (uint32_t a, SIMD16_32u const & b) { return b.cmpge(a); }
        inline SIMDMask32 operator<= (uint32_t a, SIMD32_32u const & b) { return b.cmpge(a); }

        inline SIMDMask1 operator<= (uint64_t a, SIMD1_64u const & b) { return b.cmpge(a); }
        inline SIMDMask2 operator<= (uint64_t a, SIMD2_64u const & b) { return b.cmpge(a); }
        inline SIMDMask4 operator<= (uint64_t a, SIMD4_64u const & b) { return b.cmpge(a); }
        inline SIMDMask8 operator<= (uint64_t a, SIMD8_64u const & b) { return b.cmpge(a); }
        inline SIMDMask16 operator<= (uint64_t a, SIMD16_64u const & b) { return b.cmpge(a); }

        inline SIMDMask1 operator<= (uint8_t a, SIMD1_8i const & b) { return b.cmpge(a); }
        inline SIMDMask2 operator<= (uint8_t a, SIMD2_8i const & b) { return b.cmpge(a); }
        inline SIMDMask4 operator<= (uint8_t a, SIMD4_8i const & b) { return b.cmpge(a); }
        inline SIMDMask8 operator<= (uint8_t a, SIMD8_8i const & b) { return b.cmpge(a); }
        inline SIMDMask16 operator<= (uint8_t a, SIMD16_8i const & b) { return b.cmpge(a); }
        inline SIMDMask32 operator<= (uint8_t a, SIMD32_8i const & b) { return b.cmpge(a); }
        inline SIMDMask64 operator<= (uint8_t a, SIMD64_8i const & b) { return b.cmpge(a); }
        inline SIMDMask128 operator<= (uint8_t a, SIMD128_8i const & b) { return b.cmpge(a); }

        inline SIMDMask1 operator<= (uint16_t a, SIMD1_16i const & b) { return b.cmpge(a); }
        inline SIMDMask2 operator<= (uint16_t a, SIMD2_16i const & b) { return b.cmpge(a); }
        inline SIMDMask4 operator<= (uint16_t a, SIMD4_16i const & b) { return b.cmpge(a); }
        inline SIMDMask8 operator<= (uint16_t a, SIMD8_16i const & b) { return b.cmpge(a); }
        inline SIMDMask16 operator<= (uint16_t a, SIMD16_16i const & b) { return b.cmpge(a); }
        inline SIMDMask32 operator<= (uint16_t a, SIMD32_16i const & b) { return b.cmpge(a); }
        inline SIMDMask64 operator<= (uint16_t a, SIMD64_16i const & b) { return b.cmpge(a); }

        inline SIMDMask1 operator<= (uint32_t a, SIMD1_32i const & b) { return b.cmpge(a); }
        inline SIMDMask2 operator<= (uint32_t a, SIMD2_32i const & b) { return b.cmpge(a); }
        inline SIMDMask4 operator<= (uint32_t a, SIMD4_32i const & b) { return b.cmpge(a); }
        inline SIMDMask8 operator<= (uint32_t a, SIMD8_32i const & b) { return b.cmpge(a); }
        inline SIMDMask16 operator<= (uint32_t a, SIMD16_32i const & b) { return b.cmpge(a); }
        inline SIMDMask32 operator<= (uint32_t a, SIMD32_32i const & b) { return b.cmpge(a); }

        inline SIMDMask1 operator<= (uint64_t a, SIMD1_64i const & b) { return b.cmpge(a); }
        inline SIMDMask2 operator<= (uint64_t a, SIMD2_64i const & b) { return b.cmpge(a); }
        inline SIMDMask4 operator<= (uint64_t a, SIMD4_64i const & b) { return b.cmpge(a); }
        inline SIMDMask8 operator<= (uint64_t a, SIMD8_64i const & b) { return b.cmpge(a); }
        inline SIMDMask16 operator<= (uint64_t a, SIMD16_64i const & b) { return b.cmpge(a); }

        inline SIMDMask1 operator<= (float a, SIMD1_32f const & b) { return b.cmpge(a); }
        inline SIMDMask2 operator<= (float a, SIMD2_32f const & b) { return b.cmpge(a); }
        inline SIMDMask4 operator<= (float a, SIMD4_32f const & b) { return b.cmpge(a); }
        inline SIMDMask8 operator<= (float a, SIMD8_32f const & b) { return b.cmpge(a); }
        inline SIMDMask16 operator<= (float a, SIMD16_32f const & b) { return b.cmpge(a); }
        inline SIMDMask32 operator<= (float a, SIMD32_32f const & b) { return b.cmpge(a); }

        inline SIMDMask1 operator<= (double a, SIMD1_64f const & b) { return b.cmpge(a); }
        inline SIMDMask2 operator<= (double a, SIMD2_64f const & b) { return b.cmpge(a); }
        inline SIMDMask4 operator<= (double a, SIMD4_64f const & b) { return b.cmpge(a); }
        inline SIMDMask8 operator<= (double a, SIMD8_64f const & b) { return b.cmpge(a); }
        inline SIMDMask16 operator<= (double a, SIMD16_64f const & b) { return b.cmpge(a); }
    }
}
#endif
