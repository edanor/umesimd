cmake_minimum_required (VERSION 2.6.4)
project(UMESIMDUnitTest)

# The version number
set(UMESIMDUnitTest_VERSION_MAJOR 1)
set(UMESIMDUnitTest_VERSION_MINOR 0)

set(CMAKE_BINARY_DIR ..)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ ${CMAKE_CURRENT_SOURCE_DIR}/../../)

FILE(GLOB header_files *.h)
FILE(GLOB body_files *.cpp)
FILE(GLOB ume_simd_files ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

source_group(include FILES ${header_files})
source_group(sources FILES ${body_files})

#add ume files in separate category
source_group(UME_SIMD FILES ${ume_simd_files})

add_executable( UMESIMDUnitTest UMESIMDUnitTest.cpp ${header_files} ${ume_files})
