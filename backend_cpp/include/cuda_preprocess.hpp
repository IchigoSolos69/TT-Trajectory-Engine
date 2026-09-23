#ifndef CUDA_PREPROCESS_HPP
#define CUDA_PREPROCESS_HPP

#include <cuda_runtime.h>

// Launch CUDA kernel for parallel image letterboxing, BGR->RGB conversion, 0..1 normalization, and CHW planar layout
void launchLetterboxPreprocessKernel(
    const unsigned char* d_srcBGR, int srcW, int srcH, int srcStep,
    float* d_dstCHW, int dstW, int dstH,
    cudaStream_t stream
);

#endif // CUDA_PREPROCESS_HPP
