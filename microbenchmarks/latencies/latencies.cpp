
#include "latencies.h"
#include "latencies_uint.h"
#include "latencies_int.h"
#include "latencies_float.h"

int main()
{
    test_uint_latencies();
    test_int_latencies();
    test_float_latencies();

    return 0;
}