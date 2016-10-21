#!/bin/bash
# These three variables should be passed by users
# CXX={gcc, icc, clang)
# ISA={scalar, avx, avx2, core_avx512, mic_avx512, imci}
# BUILD={debug, release, release_O3}

COMPILER="CXX=$1"
ISA="ISA=$2"
BUILD="BUILD=$3"


RESULT="average_$1_$2_$3.txt"
cd average
make $COMPILER $ISA $BUILD
for i in *.out; do "./$i" > "../$RESULT"; done
rm *.out
cd ..

RESULT="explog_$1_$2_$3.txt"
cd explog
make $COMPILER $ISA $BUILD
for i in *.out; do "./$i" > "../$RESULT"; done
rm *.out
cd ..

RESULT="histogram1_$1_$2_$3.txt"
cd histogram1
make $COMPILER $ISA $BUILD
for i in *.out; do "./$i" > "../$RESULT"; done
rm *.out
cd ..

RESULT="histogram2_$1_$2_$3.txt"
cd histogram2
make $COMPILER $ISA $BUILD
for i in *.out; do "./$i" > "../$RESULT"; done
rm *.out
cd ..

RESULT="mandelbrot1_$1_$2_$3.txt"
cd mandelbrot1
make $COMPILER $ISA $BUILD
for i in *.out; do "./$i" > "../$RESULT"; done
rm *.out
rm *.bmp
cd ..

RESULT="mandelbrot2_$1_$2_$3.txt"
cd mandelbrot2
make $COMPILER $ISA $BUILD
for i in *.out; do "./$i" > "../$RESULT"; done
rm *.out
rm *.bmp
cd ..

#this benchmark is not complete yet!
#RESULT="matmul_$1_$2_$3.txt"
#cd matmul
#make $COMPILER $ISA $BUILD
#for i in *.out; do "./$i" > "../$RESULT"; done
#rm *.out
#cd ..

RESULT="polynomial_$1_$2_$3.txt"
cd polynomial
make $COMPILER $ISA $BUILD
for i in *.out; do "./$i" > "../$RESULT"; done
rm *.out
cd ..

RESULT="quadraticsolver_$1_$2_$3.txt"
cd QuadraticSolver
make $COMPILER $ISA $BUILD
for i in *.out; do "./$i" > "../$RESULT"; done
rm *.out
cd ..

RESULT="sincos_$1_$2_$3.txt"
cd sincos
make $COMPILER $ISA $BUILD
for i in *.out; do "./$i" > "../$RESULT"; done
rm *.out
cd ..
