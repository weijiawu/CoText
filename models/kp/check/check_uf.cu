#include <assert.h>
#include <nppi_filtering_functions.h>
#include <stdio.h>
#define WIDTH 16
#define HEIGHT 16

template<typename T>
void my_print(T *data, int w, int h) {

    for (int i = 0; i < h; i++)

    {

        for (int j = 0; j < w; j++)

        {

            if (data[i * w + j] == 255) printf("  *");

            else
                printf("%3hd", data[i * w + j]);
        }

        printf("\n");
    }
}

template<typename T>
__global__ void bb(const T *__restrict__ i, int *__restrict__ maxh, int *__restrict__ minh, int *__restrict__ maxw, int *__restrict__ minw, int height, int width) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if ((idx < width) && (idy < height)) {
        T myval = i[idy * width + idx];
        if (myval > 0) {
            atomicMax(maxw + myval - 1, idx);
            atomicMin(minw + myval - 1, idx);
            atomicMax(maxh + myval - 1, idy);
            atomicMin(minh + myval - 1, idy);
        }
    }
}

int main() {
    Npp8u host_src[WIDTH * HEIGHT] =
            {
                    0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0,
                    0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0,
                    0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0,
                    0, 255, 255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255,
                    0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 255, 255,
                    0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255,
                    0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 255, 255, 255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0,
                    0, 255, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0,
                    0, 0, 255, 255, 255, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0,
                    0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 255, 255,
                    0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255,
                    0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255,
                    0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    Npp8u *device_src;
    Npp32u *device_dst;
    cudaMalloc((void **) &device_src, sizeof(Npp8u) * WIDTH * HEIGHT);
    cudaMalloc((void **) &device_dst, sizeof(Npp32u) * WIDTH * HEIGHT);
    cudaMemcpy(device_src, host_src, sizeof(Npp8u) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);

    int buffer_size;
    NppiSize source_roi = {WIDTH, HEIGHT};
    NppStatus e = nppiLabelMarkersUFGetBufferSize_32u_C1R(source_roi, &buffer_size);
    assert(e == NPP_NO_ERROR);
    Npp8u *buffer;
    cudaMalloc((void **) &buffer, buffer_size);
    int bs;
    e = nppiCompressMarkerLabelsGetBufferSize_32u_C1R(WIDTH * HEIGHT, &bs);
    assert(e == NPP_NO_ERROR);
    if (bs > buffer_size) {
        buffer_size = bs;
        cudaFree(buffer);
        cudaMalloc(&buffer, buffer_size);
    }


    e = nppiLabelMarkersUF_8u32u_C1R(device_src, sizeof(Npp8u) * WIDTH, device_dst, sizeof(Npp32u) * WIDTH, source_roi, nppiNormInf, buffer);
    assert(e == NPP_NO_ERROR);

    int max;
    e = nppiCompressMarkerLabelsUF_32u_C1IR(device_dst, sizeof(Npp32u) * WIDTH, source_roi, source_roi.width * source_roi.height, &max, buffer);

    assert(e == NPP_NO_ERROR);
    int *maxw, *maxh, *minw, *minh, *d_maxw, *d_maxh, *d_minw, *d_minh;
    maxw = new int[max];
    maxh = new int[max];
    minw = new int[max];
    minh = new int[max];
    cudaMalloc(&d_maxw, max * sizeof(int));
    cudaMalloc(&d_maxh, max * sizeof(int));
    cudaMalloc(&d_minw, max * sizeof(int));
    cudaMalloc(&d_minh, max * sizeof(int));
    for (int i = 0; i < max; i++) {
        maxw[i] = 0;
        maxh[i] = 0;
        minw[i] = WIDTH;
        minh[i] = HEIGHT;
    }
    cudaMemcpy(d_maxw, maxw, max * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxh, maxh, max * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minw, minw, max * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minh, minh, max * sizeof(int), cudaMemcpyHostToDevice);
    dim3 block(32, 32);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
    bb<<<grid, block>>>(device_src, d_maxh, d_minh, d_maxw, d_minw, HEIGHT, WIDTH);
    cudaMemcpy(maxw, d_maxw, max * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(maxh, d_maxh, max * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(minw, d_minw, max * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(minh, d_minh, max * sizeof(int), cudaMemcpyDeviceToHost);

    Npp32u *dst = new Npp32u[WIDTH * HEIGHT];
    cudaMemcpy(dst, device_dst, sizeof(Npp32u) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    printf("*******INPUT************\n");
    my_print(host_src, WIDTH, HEIGHT);
    printf("******OUTPUT************\n");
    my_print(dst, WIDTH, HEIGHT);
    printf("compressed max: %d\n", max);
    printf("bounding boxes:\n");
    for (int i = 0; i < max; i++)
        printf("label %d, maxh: %d, minh: %d, maxw: %d, minw: %d\n", i + 1, maxh[i], minh[i], maxw[i], minw[i]);
}