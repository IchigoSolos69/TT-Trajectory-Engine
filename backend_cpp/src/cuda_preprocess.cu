#include "cuda_preprocess.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>

__global__ void letterboxPreprocessKernel(
    const unsigned char* __restrict__ srcBGR, int srcW, int srcH, int srcStep,
    float* __restrict__ dstCHW, int dstW, int dstH,
    float scaleR, int padW, int padH, int unpadW, int unpadH
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dstW || dy >= dstH) return;

    int channelSize = dstW * dstH;
    int dstIdx = dy * dstW + dx;

    // Check if inside letterbox padding region
    if (dy < padH || dy >= (padH + unpadH) || dx < padW || dx >= (padW + unpadW)) {
        float padVal = 114.0f / 255.0f;
        dstCHW[0 * channelSize + dstIdx] = padVal;
        dstCHW[1 * channelSize + dstIdx] = padVal;
        dstCHW[2 * channelSize + dstIdx] = padVal;
        return;
    }

    // Bilinear or nearest sampling from source BGR image
    float srcX = (dx - padW) / scaleR;
    float srcY = (dy - padH) / scaleR;

    int sx = __float2int_rd(srcX);
    int sy = __float2int_rd(srcY);

    sx = min(max(sx, 0), srcW - 1);
    sy = min(max(sy, 0), srcH - 1);

    int srcIdx = sy * srcStep + sx * 3;

    float b = srcBGR[srcIdx + 0] / 255.0f;
    float g = srcBGR[srcIdx + 1] / 255.0f;
    float r = srcBGR[srcIdx + 2] / 255.0f;

    // Write Planar RGB (CHW)
    dstCHW[0 * channelSize + dstIdx] = r;
    dstCHW[1 * channelSize + dstIdx] = g;
    dstCHW[2 * channelSize + dstIdx] = b;
}

void launchLetterboxPreprocessKernel(
    const unsigned char* d_srcBGR, int srcW, int srcH, int srcStep,
    float* d_dstCHW, int dstW, int dstH,
    cudaStream_t stream
) {
    float scaleR = fminf(static_cast<float>(dstW) / srcW, static_cast<float>(dstH) / srcH);
    int unpadW = static_cast<int>(roundf(srcW * scaleR));
    int unpadH = static_cast<int>(roundf(srcH * scaleR));
    int padW = (dstW - unpadW) / 2;
    int padH = (dstH - unpadH) / 2;

    dim3 block(16, 16);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);

    letterboxPreprocessKernel<<<grid, block, 0, stream>>>(
        d_srcBGR, srcW, srcH, srcStep,
        d_dstCHW, dstW, dstH,
        scaleR, padW, padH, unpadW, unpadH
    );
}
