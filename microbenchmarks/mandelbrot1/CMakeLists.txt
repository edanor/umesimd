file(GLOB MANDEL_SRCS *.cpp)

add_executable(mandelbrot1 ${MANDEL_SRCS} ../utilities/UMEBitmap.cpp)

add_test(Mandelbrot1 mandelbrot1)
