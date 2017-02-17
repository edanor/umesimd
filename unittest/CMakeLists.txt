
file(GLOB TEST_SRCS *.cpp)

add_executable(run-all-tests ${TEST_SRCS})

add_test(UnitTests run-all-tests)

