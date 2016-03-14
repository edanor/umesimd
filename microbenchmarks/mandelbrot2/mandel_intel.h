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
#endif