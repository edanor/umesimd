# default compilation flags for all compilers
# can be overwritten by compiler module if needed

add_compile_options(-W -Wall -Werror -Wstrict-aliasing -fstrict-aliasing -pedantic)

set(CMAKE_CXX_FLAGS_DEBUG          "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple|)Clang|GNU|Intel")
  include(${CMAKE_CXX_COMPILER_ID})
else()
  message(WARNING "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

if (DEFINED TARGET_ISA)
  string(TOUPPER "${TARGET_ISA}" TARGET_ISA)

  if (NOT "${TARGET_ISA}" MATCHES "^(NATIVE|SSE(2|3|4.(1|2))|AVX(|2|512)|KNC|KNL)$")
    message(FATAL_ERROR "Unknown instruction set architecture: ${TARGET_ISA}")
  endif()

  string(REPLACE "." "" ISA "${TARGET_ISA}")

  if (NOT DEFINED FLAGS_${ISA})
    set(COMPILER "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    message(FATAL_ERROR "${TARGET_ISA} not supported by ${COMPILER} compiler")
  else()
    add_compile_options(${FLAGS_${ISA}})
  endif()

  if (NOT DEFINED TARGET_ISA_FLAGS_SET)
    set(TARGET_ISA_FLAGS_SET True CACHE BOOL "")
    message(STATUS "Compiling for ${TARGET_ISA} instruction set architecture")
  endif()
endif()
