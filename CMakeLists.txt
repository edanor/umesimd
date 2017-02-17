cmake_minimum_required(VERSION 3.1)

project(umesimd LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ ISO Standard")
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(BUILD_TESTING    "Build test binaries")
option(BUILD_BENCHMARKS "Build benchmarking binaries")

set(UMESIMD_VERSION_FILE "include/ume/simd")
file(STRINGS "${UMESIMD_VERSION_FILE}" UMESIMD_VERSION_PARTS REGEX "#define UMESIMD_VERSION_[A-Z]+[ ]+" )
string(REGEX REPLACE ".+UMESIMD_VERSION_MAJOR[ ]+([0-9]+).*" "\\1" UMESIMD_VERSION_MAJOR "${UMESIMD_VERSION_PARTS}")
string(REGEX REPLACE ".+UMESIMD_VERSION_MINOR[ ]+([0-9]+).*" "\\1" UMESIMD_VERSION_MINOR "${UMESIMD_VERSION_PARTS}")
string(REGEX REPLACE ".+UMESIMD_VERSION_PATCH[ ]+([0-9]+).*" "\\1" UMESIMD_VERSION_PATCH "${UMESIMD_VERSION_PARTS}")
set(UMESIMD_VERSION "${UMESIMD_VERSION_MAJOR}.${UMESIMD_VERSION_MINOR}.${UMESIMD_VERSION_PATCH}")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Select build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo)
endif()

include(CompilerSetup)
include(GNUInstallDirs)

include_directories(include)

configure_file(cmake/UMESIMDConfig.cmake.in cmake/UMESIMDConfig.cmake)
configure_file(cmake/UMESIMDConfigVersion.cmake.in cmake/UMESIMDConfigVersion.cmake)

install(FILES LICENSE README.md DESTINATION ${CMAKE_INSTALL_DOCDIR})

install(DIRECTORY doc/     DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(DIRECTORY examples DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${PROJECT_BINARY_DIR}/cmake/
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
        FILES_MATCHING PATTERN *.cmake)

enable_testing()

if (BUILD_TESTING)
	add_subdirectory(unittest)
endif()

if (BUILD_BENCHMARKS)
	add_subdirectory(microbenchmarks)
endif()
