#!/bin/bash
#TODO: this file requires modifications to support GCC

FLAGS="-std=c++11 -fp-model=precise -O2"
CC="icpc"

if [ "$1" == "novec" ]; then
    FLAGS="$FLAGS -no-vec"
    BINARY_PREFIX="novec"
elif [ "$1" == "scalar" ]; then
    echo $1
    BINARY_PREFIX="scalar"
elif [ "$1" == "avx" ]; then
    echo $1
    FLAGS="$FLAGS -mavx"
    BINARY_PREFIX="avx"
elif [ "$1" == "avx2" ]; then
    echo $1
    FLAGS="$FLAGS -xCORE-AVX2"
    BINARY_PREFIX="avx2"
elif [ "$1" == "avx512" ]; then
    echo $1
    FLAGS="$FLAGS -xCORE-AVX512"
    BINARY_PREFIX="avx512"
fi

UINT8_BINARY_NAME=$BINARY_PREFIX"_8u.out"
UINT16_BINARY_NAME=$BINARY_PREFIX"_16u.out"
UINT32_BINARY_NAME=$BINARY_PREFIX"_32u.out"
UINT64_BINARY_NAME=$BINARY_PREFIX"_64u.out"
INT8_BINARY_NAME=$BINARY_PREFIX"_8i.out"
INT16_BINARY_NAME=$BINARY_PREFIX"_16i.out"
INT32_BINARY_NAME=$BINARY_PREFIX"_32i.out"
INT64_BINARY_NAME=$BINARY_PREFIX"_64i.out"
FLOAT32_BINARY_NAME=$BINARY_PREFIX"_32f.out"
FLOAT64_BINARY_NAME=$BINARY_PREFIX"_64f.out"

time $CC latencies_8u.cpp  $FLAGS -o $UINT8_BINARY_NAME
time $CC latencies_16u.cpp $FLAGS -o $UINT16_BINARY_NAME
time $CC latencies_32u.cpp $FLAGS -o $UINT32_BINARY_NAME
time $CC latencies_64u.cpp $FLAGS -o $UINT64_BINARY_NAME
time $CC latencies_8i.cpp  $FLAGS -o $INT8_BINARY_NAME
time $CC latencies_16i.cpp $FLAGS -o $INT16_BINARY_NAME
time $CC latencies_32i.cpp $FLAGS -o $INT32_BINARY_NAME
time $CC latencies_64i.cpp $FLAGS -o $INT64_BINARY_NAME
time $CC latencies_32f.cpp $FLAGS -o $FLOAT32_BINARY_NAME
time $CC latencies_64f.cpp $FLAGS -o $FLOAT64_BINARY_NAME
