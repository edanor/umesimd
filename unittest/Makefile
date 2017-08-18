# These three variables should be passed by users
# CXX={gcc, icc, clang)
# ISA={scalar, avx, avx2, core_avx512, mic_avx512, imci, arm, altivec}
# BUILD={debug, release, release_O3}
# {FORCE_OPENMP_PLUGIN=ON}
# {FORCE_SCALAR_PLUGIN=ON}

CXXFLAGS=-std=c++14
CXX=g++
ISA=mic_avx512
BUILD=release

ifneq (,$(findstring armclang,$(CXX)))
	CXXCOMPILER=armclang++
else
ifneq (,$(findstring clang,$(CXX)))
	CXXCOMPILER=clang++
else
ifneq (,$(findstring g++,$(CXX)))
	CXXCOMPILER=g++
else
ifneq (,$(findstring ic, $(CXX)))
	CXXCOMPILER=icc
endif
endif
endif
endif

#some predefined rules
ifeq ($(CXXCOMPILER), g++)
	COMPILER_PREFIX=gcc
#-W -Wall
	CXXFLAGS+= -pedantic -fstrict-aliasing -Wstrict-aliasing -Woverflow -Werror
endif
ifeq ($(CXXCOMPILER), clang++)
	COMPILER_PREFIX=clang
	CXXFLAGS+=-W -Wall -pedantic -fstrict-aliasing -Wstrict-aliasing -Wshorten-64-to-32 -Woverflow -Werror
endif
ifeq ($(CXXCOMPILER), armclang++)
	COMPILER_PREFIX=armclang
    CXXFLAGS+=-W -Wall -pedantic -fstrict-aliasing -Wstrict-aliasing -Wshorten-64-to-32 -Woverflow -Werror
endif
ifeq ($(CXXCOMPILER), icc)
	COMPILER_PREFIX=icc
	CXXFLAGS+=-DUME_USE_SVML -W -Wall -pedantic -fstrict-aliasing -Wstrict-aliasing -Werror
endif

#select proper build flags
ifeq ($(BUILD), debug)
	#for intel compiler use precise results generation
	ifeq ($(CXXCOMPILER), icc)
		CXXFLAGS+=-fp-model=precise
	endif
  
	CXXFLAGS+=-O0 -fno-inline -g
	BUILD_PREFIX=_O0
endif
ifeq ($(BUILD), release)
	#for intel compiler use precise results generation
	ifeq ($(CXXCOMPILER), icc)
		CXXFLAGS+=-fp-model=precise
	endif

	CXXFLAGS+=-O2
	BUILD_PREFIX=_O2
endif
ifeq ($(BUILD), release_o3)
	CXXFLAGS+=-O3
	BUILD_PREFIX=_O3
endif

ifeq ($(FORCE_OPENMP_PLUGIN), ON)
	CXXFLAGS+=-fopenmp
	CXXFLAGS+=-DFORCE_OPENMP
endif

ifeq ($(FORCE_SCALAR_PLUGIN), ON)
	CXXFLAGS+=-DFORCE_SCALAR
endif

# Select proper instruction set flags
ifeq ($(ISA), scalar)
	ISA_PREFIX+=_scalar
endif
ifeq ($(ISA), avx)
	CXXFLAGS+=-mavx
	ISA_PREFIX+=_avx
endif
ifeq ($(ISA), avx2)
	ifeq ($(CXXCOMPILER), icc)
		CXXFLAGS+=-xCORE-AVX2
	else
		CXXFLAGS+=-mavx2
	endif
	ISA_PREFIX+=_avx2
endif
ifeq ($(ISA), core_avx512)
	ifeq ($(CXXCOMPILER), icc)
		CXXFLAGS+=-xCORE-AVX512
	else
		CXXFLAGS+=-march=skylake-avx512
	endif
	ISA_PREFIX+=_core_avx512
endif
ifeq ($(ISA), mic_avx512) 
	ifeq ($(CXXCOMPILER), icc)
		CXXFLAGS+=-xMIC-AVX512
	else
		CXXFLAGS+=-march=knl
	endif
	ISA_PREFIX+=_mic_avx512
endif
ifeq ($(ISA), imci)
	CXXFLAGS+=-mmic
	ISA_PREFIX+=_imci
endif
ifeq ($(ISA), arm)
	ISA_PREFIX+=_arm
    ifeq ($(CXXCOMPILER), g++)
        CXXFLAGS+=-march=native
    endif
endif
ifeq ($(ISA), altivec)
	CXXFLAGS+=-maltivec -mtune=native -mcpu=native -mvsx -mabi=altivec
	ISA_PREFIX+=_altivec
endif

OUT_NAME=$(join $(join $(join $(COMPILER_PREFIX), $(ISA_PREFIX)), $(BUILD_PREFIX)), .out)

#all: executable

all:
	$(MAKE) clean
	$(MAKE) executable

ifndef CXX
  $(error CXX is not set)
endif
ifndef BUILD
	$(error BUILD is not set)
endif
ifndef ISA
	$(error ISA is not set)
endif

executable: data_sets tests
	$(CXX) $(CXXFLAGS) UMESIMDUnitTest.cpp UMEUnitTestDataSets8.o UMEUnitTestDataSets16.o UMEUnitTestDataSets32.o UMEUnitTestDataSets64.o UMEUnitTestCommon.o UMEUnitTestSimd8b.o UMEUnitTestSimd16b.o UMEUnitTestSimd32b.o UMEUnitTestSimd64b.o UMEUnitTestSimd128b.o UMEUnitTestSimd256b.o UMEUnitTestSimd512b.o UMEUnitTestSimd1024b.o -o $(OUT_NAME)

data_sets: UMEUnitTestDataSets8.o UMEUnitTestDataSets16.o UMEUnitTestDataSets32.o UMEUnitTestDataSets64.o

UMEUnitTestDataSets8.o: UMEUnitTestDataSets8.cpp
	$(CXX) $(CXXFLAGS) -c UMEUnitTestDataSets8.cpp
  
UMEUnitTestDataSets16.o: UMEUnitTestDataSets16.cpp
	$(CXX) $(CXXFLAGS) -c UMEUnitTestDataSets16.cpp

UMEUnitTestDataSets32.o: UMEUnitTestDataSets32.cpp
	$(CXX) $(CXXFLAGS) -c UMEUnitTestDataSets32.cpp

UMEUnitTestDataSets64.o: UMEUnitTestDataSets64.cpp
	$(CXX) $(CXXFLAGS) -c UMEUnitTestDataSets64.cpp

tests: UMEUnitTestCommon.o UMEUnitTestSimd8b.o UMEUnitTestSimd16b.o UMEUnitTestSimd32b.o UMEUnitTestSimd64b.o UMEUnitTestSimd128b.o UMEUnitTestSimd256b.o UMEUnitTestSimd512b.o UMEUnitTestSimd1024b.o
 
UMEUnitTestCommon.o: UMEUnitTestCommon.cpp
	$(CXX) $(CXXFLAGS) -c UMEUnitTestCommon.cpp

UMEUnitTestSimd8b.o: UMEUnitTestSimd8b.cpp
	$(CXX) $(CXXFLAGS) -c UMEUnitTestSimd8b.cpp

UMEUnitTestSimd16b.o: UMEUnitTestSimd16b.cpp
	$(CXX) $(CXXFLAGS) -c UMEUnitTestSimd16b.cpp

UMEUnitTestSimd32b.o: UMEUnitTestSimd32b.cpp
	$(CXX) $(CXXFLAGS) -c UMEUnitTestSimd32b.cpp

UMEUnitTestSimd64b.o: UMEUnitTestSimd64b.cpp
	$(CXX) $(CXXFLAGS) -c UMEUnitTestSimd64b.cpp

UMEUnitTestSimd128b.o: UMEUnitTestSimd128b.cpp
	$(CXX) $(CXXFLAGS) -c UMEUnitTestSimd128b.cpp

UMEUnitTestSimd256b.o: UMEUnitTestSimd256b.cpp
	$(CXX) $(CXXFLAGS) -c UMEUnitTestSimd256b.cpp

UMEUnitTestSimd512b.o: UMEUnitTestSimd512b.cpp
	$(CXX) $(CXXFLAGS) -c UMEUnitTestSimd512b.cpp

UMEUnitTestSimd1024b.o: UMEUnitTestSimd1024b.cpp
	$(CXX) $(CXXFLAGS) -c UMEUnitTestSimd1024b.cpp

clean:
	rm -f *.o 
