// 
// This piece of code comes from http://www.lomont.org/.
// This code is not a part of UME::SIMD library code and is used purely for
// performance measurement reference.
// 
// Modifications have been made to original files to fit them for benchmarking
// of UME::SIMD.

#include <complex>
void MandelbrotCPU(float x1, float y1, float x2, float y2, int width, int height, int maxIters, unsigned short * image)
{
    float dx = (x2 - x1) / width, dy = (y2 - y1) / height;
    for (int j = 0; j < height; ++j)
        for (int i = 0; i < width; ++i)
        {
            std::complex<float> c(x1 + dx*i, y1 + dy*j), z(0, 0);
            int count = -1;
            while ((++count < maxIters) && (std::norm(z) < 4.0))
                z = z*z + c;
            *image++ = count;
        }
}

void MandelbrotCPU2(float x1, float y1, float x2, float y2, int width, int height, int maxIters, unsigned short * image)
{
    float dx = (x2 - x1) / width, dy = (y2 - y1) / height;
    for (int j = 0; j < height; ++j)
        for (int i = 0; i < width; ++i)
        {
            float cx = x1 + dx*i, cy = y1 + dy*j;
            float x = cx, y = cy;
            int count = 0;
            for (count = 1; count < maxIters; ++count)
            {
                float x2 = x * x, y2 = y * y;
                if (x2 + y2 >= 4)
                    break;
                float xy = x*y;
                x = x2 - y2 + cx;
                y = 2 * xy + cy;
            }
            *image++ = count;
        }
}

void MandelbrotCPU2_64f(double x1, double y1, double x2, double y2, int width, int height, int maxIters, unsigned short * image)
{
    double dx = (x2 - x1) / width, dy = (y2 - y1) / height;
    for (int j = 0; j < height; ++j)
        for (int i = 0; i < width; ++i)
        {
            double cx = x1 + dx*i, cy = y1 + dy*j;
            double x = cx, y = cy;
            int count = 0;
            for (count = 1; count < maxIters; ++count)
            {
                double x2 = x * x, y2 = y * y;
                if (x2 + y2 >= 4)
                    break;
                double xy = x*y;
                x = x2 - y2 + cx;
                y = 2 * xy + cy;
            }
            *image++ = count;
        }
}
#if defined __SSE2__
// SSE based mandelbrot
/*void MandelbrotSSE2(float x1, float y1, float x2, float y2, int width, int height, int maxIters, unsigned short * image)
{
float dx = (x2 - x1) / width;
float dy = (y2 - y1) / height;
int widthEven = width & ~1UL; // todo - make work for odd height, width also
int heightEven = height & ~1UL;

float fours[4] = { 4.0f,4.0f,4.0f,4.0f };
float onesF[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
__m128 xmm5 = _mm_load_ps(fours);
__m128 ones = _mm_load_ps(onesF);

for (int j = 0; j < heightEven; j += 2)
{
for (int i = 0; i < widthEven; i += 2)
{
float xi[4], yi[4];
xi[0] = xi[2] = x1 + i*dx;
xi[1] = xi[3] = x1 + (i + 1)*dx;
yi[0] = yi[1] = y1 + j*dy;
yi[2] = yi[3] = y1 + (j + 1)*dy;
__m128 xmm6 = _mm_load_ps(xi);   // 4 x and y values on a 2x2 grid
__m128 xmm7 = _mm_load_ps(yi);

__m128 xmm0 = _mm_xor_ps(xmm6, xmm6); // zero out xmm0,xmm1,xmm3
__m128 xmm1 = xmm0;
__m128 xmm3 = xmm0;
__m128 xmm2, xmm4;

unsigned int test = 0;

int iter = 0;
do
{
xmm2 = xmm0;                           // xi
xmm2 = _mm_mul_ps(xmm2, xmm1);          // xi * yi
xmm0 = _mm_mul_ps(xmm0, xmm0);          // xi * xi
xmm1 = _mm_mul_ps(xmm1, xmm1);          // yi * yi
xmm4 = xmm0;                           // xi * xi
xmm4 = _mm_add_ps(xmm4, xmm1);          // xi*xi + yi*yi
xmm0 = _mm_sub_ps(xmm0, xmm1);          // xi*xi - yi*yi
xmm0 = _mm_add_ps(xmm0, xmm6);          // xi*xi - yi*yi + x0
xmm1 = xmm2;                           // xi*yi
xmm1 = _mm_add_ps(xmm1, xmm1);          // 2*xi*yi
xmm1 = _mm_add_ps(xmm1, xmm7);          // 2*xi*yi+y0

xmm4 = _mm_cmplt_ps(xmm4, xmm5);        // xi*xi+yi*yi < 4 in each slot
// now xmm4 has all 1s in the non overflowed locations
// xmm0 has the new x value
// xmm1 has the new y value
test = _mm_movemask_ps(xmm4) & 15;       // lower 4 bits are comparisons
xmm4 = _mm_and_ps(xmm4, ones);           // get 1.0f or 0.0f in each field
xmm3 = _mm_add_ps(xmm3, xmm4);          // xmm3 is counters for each pixel iteration

++iter;
} while ((test != 0) && (iter < maxIters));

image[i + j*width] = (unsigned short)xmm3.m128_f32[0];
image[i + 1 + j*width] = (unsigned short)xmm3.m128_f32[1];
image[i + (j + 1)*width] = (unsigned short)xmm3.m128_f32[2];
image[i + 1 + (j + 1)*width] = (unsigned short)xmm3.m128_f32[3];
}
}
}*/
#endif 

#if defined __AVX__
// Intel AVX based mandelbrot
void MandelbrotAVX(float x1, float y1, float x2, float y2, int width, int height, int maxIters, unsigned short * image)
{
    float dx = (x2 - x1) / width;
    float dy = (y2 - y1) / height;
    // round up width to next multiple of 8
    int roundedWidth = (width + 7) & ~7UL;

    float constants[] = { dx, dy, x1, y1, 1.0f, 4.0f };
    __m256 ymm0 = _mm256_broadcast_ss(constants);   // all dx
    __m256 ymm1 = _mm256_broadcast_ss(constants + 1); // all dy
    __m256 ymm2 = _mm256_broadcast_ss(constants + 2); // all x1
    __m256 ymm3 = _mm256_broadcast_ss(constants + 3); // all y1
    __m256 ymm4 = _mm256_broadcast_ss(constants + 4); // all 1's (iter increments)
    __m256 ymm5 = _mm256_broadcast_ss(constants + 5); // all 4's (comparisons)

    alignas(32) float incr[8] = { 0.0f,1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f }; // used to reset the i position when j increases
    __m256 ymm6 = _mm256_xor_ps(ymm0, ymm0); // zero out j counter (ymm0 is just a dummy)

    alignas(32) unsigned short raw_outputs[16];

    for (int j = 0; j < height; j += 1)
    {
        __m256 ymm7 = _mm256_load_ps(incr);  // i counter set to 0,1,2,..,7
        for (int i = 0; i < roundedWidth; i += 8)
        {
            __m256 ymm8 = _mm256_mul_ps(ymm7, ymm0);  // x0 = (i+k)*dx
            ymm8 = _mm256_add_ps(ymm8, ymm2);         // x0 = x1+(i+k)*dx
            __m256 ymm9 = _mm256_mul_ps(ymm6, ymm1);  // y0 = j*dy
            ymm9 = _mm256_add_ps(ymm9, ymm3);         // y0 = y1+j*dy
            __m256 ymm10 = _mm256_xor_ps(ymm0, ymm0);  // zero out iteration counter (ymm0 is just a dummy)
            __m256 ymm11 = ymm10, ymm12 = ymm10;        // set initial xi=0, yi=0

            unsigned int test = 0;
            int iter = 0;
            do
            {
                __m256 ymm13 = _mm256_mul_ps(ymm11, ymm11); // xi*xi
                __m256 ymm14 = _mm256_mul_ps(ymm12, ymm12); // yi*yi
                __m256 ymm15 = _mm256_add_ps(ymm13, ymm14); // xi*xi+yi*yi

                ymm15 = _mm256_cmp_ps(ymm15, ymm5, _CMP_LT_OQ);        // xi*xi+yi*yi < 4 in each slot
                                                                       // now ymm15 has all 1s in the non overflowed locations
                test = _mm256_movemask_ps(ymm15) & 255;      // lower 8 bits are comparisons
                ymm15 = _mm256_and_ps(ymm15, ymm4);           // get 1.0f or 0.0f in each field as counters
                ymm10 = _mm256_add_ps(ymm10, ymm15);        // counters for each pixel iteration

                ymm15 = _mm256_mul_ps(ymm11, ymm12);        // xi*yi

                ymm11 = _mm256_sub_ps(ymm13, ymm14);        // xi*xi-yi*yi
                ymm11 = _mm256_add_ps(ymm11, ymm8);         // xi <- xi*xi-yi*yi+x0 done!
                ymm12 = _mm256_add_ps(ymm15, ymm15);        // 2*xi*yi
                ymm12 = _mm256_add_ps(ymm12, ymm9);         // yi <- 2*xi*yi+y0

                ++iter;
            } while ((test != 0) && (iter < maxIters));

            // convert iterations to output values
            __m256i ymm10i = _mm256_cvtps_epi32(ymm10);

            // write only where needed
            _mm256_store_si256((__m256i*) raw_outputs, ymm10i);
            int top = (i + 7) < width ? 8 : width & 7;
            for (int k = 0; k < top; ++k)
                image[i + k + j*width] = raw_outputs[2 * k];

            // next i position - increment each slot by 8
            ymm7 = _mm256_add_ps(ymm7, ymm5);
            ymm7 = _mm256_add_ps(ymm7, ymm5);
        }
        ymm6 = _mm256_add_ps(ymm6, ymm4); // increment j counter
    }
}


// Intel AVX based mandelbrot
void MandelbrotAVX_double(double x1, double y1, double x2, double y2, int width, int height, int maxIters, unsigned short * image)
{
    double dx = (x2 - x1) / width;
    double dy = (y2 - y1) / height;
    // round up width to next multiple of 4
    int roundedWidth = (width + 3) & ~3UL;

    double constants[] = { dx, dy, x1, y1, 1.0, 4.0 };
    __m256d ymm0 = _mm256_broadcast_sd(constants);   // all dx
    __m256d ymm1 = _mm256_broadcast_sd(constants + 1); // all dy
    __m256d ymm2 = _mm256_broadcast_sd(constants + 2); // all x1
    __m256d ymm3 = _mm256_broadcast_sd(constants + 3); // all y1
    __m256d ymm4 = _mm256_broadcast_sd(constants + 4); // all 1's (iter increments)
    __m256d ymm5 = _mm256_broadcast_sd(constants + 5); // all 4's (comparisons)

    alignas(32) double incr[4] = { 0.0,1.0,2.0,3.0}; // used to reset the i position when j increases
    __m256d ymm6 = _mm256_xor_pd(ymm0, ymm0); // zero out j counter (ymm0 is just a dummy)

    alignas(32) unsigned short raw_outputs[16];

    for (int j = 0; j < height; j += 1)
    {
        __m256d ymm7 = _mm256_load_pd(incr);  // i counter set to 0,1,2,..,7
        for (int i = 0; i < roundedWidth; i += 4)
        {
            __m256d ymm8 = _mm256_mul_pd(ymm7, ymm0);  // x0 = (i+k)*dx
            ymm8 = _mm256_add_pd(ymm8, ymm2);         // x0 = x1+(i+k)*dx
            __m256d ymm9 = _mm256_mul_pd(ymm6, ymm1);  // y0 = j*dy
            ymm9 = _mm256_add_pd(ymm9, ymm3);         // y0 = y1+j*dy
            __m256d ymm10 = _mm256_xor_pd(ymm0, ymm0);  // zero out iteration counter (ymm0 is just a dummy)
            __m256d ymm11 = ymm10, ymm12 = ymm10;        // set initial xi=0, yi=0

            unsigned int test = 0;
            int iter = 0;
            do
            {
                __m256d ymm13 = _mm256_mul_pd(ymm11, ymm11); // xi*xi
                __m256d ymm14 = _mm256_mul_pd(ymm12, ymm12); // yi*yi
                __m256d ymm15 = _mm256_add_pd(ymm13, ymm14); // xi*xi+yi*yi

                ymm15 = _mm256_cmp_pd(ymm15, ymm5, _CMP_LT_OQ);        // xi*xi+yi*yi < 4 in each slot
                                                                       // now ymm15 has all 1s in the non overflowed locations
                test = _mm256_movemask_pd(ymm15) & 255;      // lower 8 bits are comparisons
                ymm15 = _mm256_and_pd(ymm15, ymm4);           // get 1.0f or 0.0f in each field as counters
                ymm10 = _mm256_add_pd(ymm10, ymm15);        // counters for each pixel iteration

                ymm15 = _mm256_mul_pd(ymm11, ymm12);        // xi*yi

                ymm11 = _mm256_sub_pd(ymm13, ymm14);        // xi*xi-yi*yi
                ymm11 = _mm256_add_pd(ymm11, ymm8);         // xi <- xi*xi-yi*yi+x0 done!
                ymm12 = _mm256_add_pd(ymm15, ymm15);        // 2*xi*yi
                ymm12 = _mm256_add_pd(ymm12, ymm9);         // yi <- 2*xi*yi+y0

                ++iter;
            } while ((test != 0) && (iter < maxIters));

            // convert iterations to output values
            __m128i ymm10i = _mm_cvtps_epi32(_mm256_cvtpd_ps(ymm10));

            // write only where needed
            _mm_store_si128((__m128i*) raw_outputs, ymm10i);
            int top = (i + 3) < width ? 4 : width & 3;
            for (int k = 0; k < top; ++k)
                image[i + k + j*width] = raw_outputs[2 * k];

            // next i position - increment each slot by 8
            ymm7 = _mm256_add_pd(ymm7, ymm5);
            ymm7 = _mm256_add_pd(ymm7, ymm5);
        }
        ymm6 = _mm256_add_pd(ymm6, ymm4); // increment j counter
    }
}
#endif

#if defined(__AVX512F__)

// Intel AVX512 based mandelbrot
void MandelbrotAVX512(float x1, float y1, float x2, float y2, int width, int height, int maxIters, unsigned short * image)
{
    float dx = (x2 - x1) / width;
    float dy = (y2 - y1) / height;
    // round up width to next multiple of 8
    int roundedWidth = (width + 16) & ~15UL;

    __m512 zmm0 = _mm512_set1_ps(dx);   // all dx
    __m512 zmm1 = _mm512_set1_ps(dy); // all dy
    __m512 zmm2 = _mm512_set1_ps(x1); // all x1
    __m512 zmm3 = _mm512_set1_ps(y1); // all y1
    __m512 zmm4 = _mm512_set1_ps(1.0f); // all 1's (iter increments)
    __m512 zmm5 = _mm512_set1_ps(4.0f); // all 4's (comparisons)

    alignas(64) float incr[16] = { 0.0f, 1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                                   8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f}; // used to reset the i position when j increases
    __m512 zmm6 = _mm512_set1_ps(0.0f); // zero out j counter (ymm0 is just a dummy)

    alignas(64) unsigned short raw_outputs[16];

    for (int j = 0; j < height; j += 1)
    {
        __m512 zmm7 = _mm512_load_ps(incr);  // i counter set to 0,1,2,..,7
        for (int i = 0; i < roundedWidth; i += 16)
        {
            __m512 zmm8 = _mm512_mul_ps(zmm7, zmm0);  // x0 = (i+k)*dx
            zmm8 = _mm512_add_ps(zmm8, zmm2);         // x0 = x1+(i+k)*dx
            __m512 zmm9 = _mm512_mul_ps(zmm6, zmm1);  // y0 = j*dy
            zmm9 = _mm512_add_ps(zmm9, zmm3);         // y0 = y1+j*dy
            __m512 zmm10 = _mm512_set1_ps(0.0f);      // zero out iteration counter (zmm0 is just a dummy)
            __m512 zmm11 = zmm10, zmm12 = zmm10;        // set initial xi=0, yi=0

            __mmask16 m0 = 0;
            int iter = 0;
            do
            {
                __m512 zmm13 = _mm512_mul_ps(zmm11, zmm11); // xi*xi
                __m512 zmm14 = _mm512_mul_ps(zmm12, zmm12); // yi*yi
                __m512 zmm15 = _mm512_add_ps(zmm13, zmm14); // xi*xi+yi*yi

                m0 = _mm512_cmp_ps_mask(zmm15, zmm5, _CMP_LT_OQ);        // xi*xi+yi*yi < 4 in each slot
                __m512 zmm16 = _mm512_set1_ps(0.0f);
                zmm16 = _mm512_mask_mov_ps(zmm16, m0, zmm4);
                zmm10 = _mm512_add_ps(zmm10, zmm16);        // counters for each pixel iteration

                zmm15 = _mm512_mul_ps(zmm11, zmm12);        // xi*yi

                zmm11 = _mm512_sub_ps(zmm13, zmm14);        // xi*xi-yi*yi
                zmm11 = _mm512_add_ps(zmm11, zmm8);         // xi <- xi*xi-yi*yi+x0 done!
                zmm12 = _mm512_add_ps(zmm15, zmm15);        // 2*xi*yi
                zmm12 = _mm512_add_ps(zmm12, zmm9);         // yi <- 2*xi*yi+y0

                ++iter;
            } while ((m0 != 0) && (iter < maxIters));

            // convert iterations to output values
            __m512i zmm10i = _mm512_cvtps_epi32(zmm10);

            // write only where needed
            _mm512_store_si512((__m512i*) raw_outputs, zmm10i);
            int top = (i + 15) < width ? 16 : width & 15;
            for (int k = 0; k < top; ++k)
                image[i + k + j*width] = raw_outputs[2 * k];

            // next i position - increment each slot by 8
            zmm7 = _mm512_add_ps(zmm7, zmm5);
            zmm7 = _mm512_add_ps(zmm7, zmm5);
        }
        zmm6 = _mm512_add_ps(zmm6, zmm4); // increment j counter
    }
}

// Intel AVX512 based mandelbrot
void MandelbrotAVX512_double(double x1, double y1, double x2, double y2, int width, int height, int maxIters, unsigned short * image)
{
    double dx = (x2 - x1) / width;
    double dy = (y2 - y1) / height;
    // round up width to next multiple of 8
    int roundedWidth = (width + 8) & ~7UL;

    __m512d zmm0 = _mm512_set1_pd(dx);   // all dx
    __m512d zmm1 = _mm512_set1_pd(dy); // all dy
    __m512d zmm2 = _mm512_set1_pd(x1); // all x1
    __m512d zmm3 = _mm512_set1_pd(y1); // all y1
    __m512d zmm4 = _mm512_set1_pd(1.0); // all 1's (iter increments)
    __m512d zmm5 = _mm512_set1_pd(4.0); // all 4's (comparisons)

    alignas(64) double incr[8] = { 0.0, 1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0}; // used to reset the i position when j increases
    __m512d zmm6 = _mm512_set1_pd(0.0); // zero out j counter (zmm0 is just a dummy)

    alignas(64) unsigned short raw_outputs[16];

    for (int j = 0; j < height; j += 1)
    {
        __m512d zmm7 = _mm512_load_pd(incr);  // i counter set to 0,1,2,..,7
        for (int i = 0; i < roundedWidth; i += 8)
        {
            __m512d zmm8 = _mm512_mul_pd(zmm7, zmm0);  // x0 = (i+k)*dx 
            zmm8 = _mm512_add_pd(ymm8, ymm2);         // x0 = x1+(i+k)*dx
            __m512d zmm9 = _mm512_mul_pd(zmm6, zmm1);  // y0 = j*dy
            zmm9 = _mm512_add_pd(zmm9, zmm3);         // y0 = y1+j*dy
            __m512d zmm10 = _mm512_set1_pd(0.0);      // zero out iteration counter (ymm0 is just a dummy)
            __m512d zmm11 = zmm10, zmm12 = zmm10;        // set initial xi=0, yi=0

            __mmask8 m0 = 0;
            int iter = 0;
            do
            {
                __m512d zmm13 = _mm512_mul_pd(zmm11, zmm11); // xi*xi
                __m512d zmm14 = _mm512_mul_pd(zmm12, zmm12); // yi*yi
                __m512d zmm15 = _mm512_add_pd(zmm13, zmm14); // xi*xi+yi*yi

                m0 = _mm512_cmp_pd_mask(zmm15, zmm5, _CMP_LT_OQ);        // xi*xi+yi*yi < 4 in each slot
                __m512d zmm16 = _mm512_set1_pd(0.0);
                zmm16 = _mm512_mask_mov_pd(zmm16, m0, zmm4);
                zmm10 = _mm512_add_pd(zmm10, zmm16);        // counters for each pixel iteration

                zmm15 = _mm512_mul_pd(zmm11, zmm12);        // xi*yi

                zmm11 = _mm512_sub_pd(zmm13, zmm14);        // xi*xi-yi*yi
                zmm11 = _mm512_add_pd(zmm11, zmm8);         // xi <- xi*xi-yi*yi+x0 done!
                zmm12 = _mm512_add_pd(zmm15, zmm15);        // 2*xi*yi
                zmm12 = _mm512_add_pd(zmm12, zmm9);         // yi <- 2*xi*yi+y0

                ++iter;
            } while ((m0 != 0) && (iter < maxIters));

            // convert iterations to output values
            __m256i ymm10i = _mm512_cvtpd_epi32(zmm10);

            // write only where needed
            _mm256_store_si256((__m256i*) raw_outputs, ymm10i);
            int top = (i + 8) < width ? 8 : width & 7;
            for (int k = 0; k < top; ++k)
                image[i + k + j*width] = raw_outputs[2 * k];

            // next i position - increment each slot by 8
            zmm7 = _mm512_add_pd(zmm7, zmm5);
            zmm7 = _mm512_add_pd(zmm7, zmm5);
        }
        zmm6 = _mm512_add_pd(zmm6, zmm4); // increment j counter
    }
}
#endif
