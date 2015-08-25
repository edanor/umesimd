#!/bin/bash
rm -r -f build
rm -r -f CMakeFiles
rm cmake_install.cmake
rm CMakeCache.txt
rm Makefile
rm -r -f test/outputs
cd src && ./clean.sh && cd ..
