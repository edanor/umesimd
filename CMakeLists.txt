cmake_minimum_required(VERSION 3.0)

project(umesimd)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/doc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/examples)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugins)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utilities)

file(GLOB top_files
        UMEBasicTypes.h
        UMEInline.h
        UMEMemory.h
        UMESimd.h
        UMESimdInterface.h
        UMESimdInterfaceFunctions.h
        UMESimdScalarEmulation.h
        UMESimdScalarOperators.h
        UMESimdTraits.h
        UMESimdVectorEmulation.h
        README.md
        LICENSE)

install(DIRECTORY doc DESTINATION include/umesimd)
install(DIRECTORY examples DESTINATION include/umesimd)
install(DIRECTORY plugins DESTINATION include/umesimd)
install(DIRECTORY utilities DESTINATION include/umesimd)
install(FILES ${top_files} DESTINATION include/umesimd)

include(CPack)
