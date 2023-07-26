//
// Created by liupeng on 2021/9/10.
//
#pragma once
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>


#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"

#include <cassert>
#include <cstdlib>
#include <iostream>

#ifndef CUDA_CHECK
#define CUDA_CHECK(condition)                                                                                 \
    do {                                                                                                      \
        cudaError_t error = (condition);                                                                      \
        if (error != cudaSuccess) {                                                                           \
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "]"                                            \
                      << " cuda error status=" << error << " msg=" << cudaGetErrorString(error) << std::endl; \
        }                                                                                                     \
    } while (0)
#endif

#ifndef CUBLAS_CHECK
static inline const char *_cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

#define CUBLAS_CHECK(condition)                                                                 \
    do {                                                                                        \
        cublasStatus_t status = condition;                                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                  \
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "]"                              \
                      << " cublas error status=" << _cublasGetErrorString(status) << std::endl; \
            exit(1);                                                                            \
        }                                                                                       \
    } while (0)
#endif


int pixel_aggregation_kernel_launcher(int *kernel, const int *region, const int h, const int w);

int crop_resize_kernel_launcher(
        cudaStream_t stream,
        int n,
        const void *image,
        const void *rois,
        const int *mask_ids,
        const int *roi_mask_ids,
        int input_height,
        int input_width,
        int num_boxes,
        int crop_height,
        int crop_width,
        int depth,
        void *output);