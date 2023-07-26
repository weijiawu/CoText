//
// Created by liupeng on 2021/9/10.
//

#ifndef KPROCESS_KPROCESS_H
#define KPROCESS_KPROCESS_H
#include "npp.h"
#include <iostream>
#include <kernel.h>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

namespace cuda {
    at::Tensor pixel_aggregate(at::Tensor kernel, at::Tensor region, bool in_place=false);
    at::Tensor warp_perspective(
            at::Tensor image_cuda_tensor,
            at::Tensor src_quad_cpu_tensor,// const double src_quad[4][2],
            at::Tensor dst_quad_cpu_tensor,//  const double dst_quad[4][2],
            int64_t height,
            int64_t width,
            int64_t eInterpolation);
    at::Tensor connectedComponents(at::Tensor image, int64_t connectivity);

    at::Tensor masked_roi(at::Tensor image, at::Tensor bboxes, at::Tensor mask, at::Tensor labels, int64_t crop_h, int64_t crop_w);
}// namespace cuda

#endif//KPROCESS_KPROCESS_H
