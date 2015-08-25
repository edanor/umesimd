#!/bin/bash
source clean.sh
includePath=$PWD/src/libraries/ume/src/libraries/BOOST/boost_1_57_0

export CC=gcc
export CXX=g++

cmake -G "Unix Makefiles" -DCMAKE_CC_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_FLAGS:STRING="-std=c++11 -Wfatal-errors -O2 -I"$includePath
