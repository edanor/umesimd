#pragma once

struct spec {
    /* Image Specification */
    int width;
    int height;
    int depth;
    /* Fractal Specification */
    float xlim[2];
    float ylim[2];
    int iterations;
};
