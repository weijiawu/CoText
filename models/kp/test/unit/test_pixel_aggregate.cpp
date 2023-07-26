//
// Created by liupeng on 2021/9/11.
//
#include "kprocess.h"
#include <gtest/gtest.h>

TEST(test_pixel_aggregation, test_pixel_aggregation_shape) {
    std::vector<uint8_t> raw_data = {
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

    auto options = at::TensorOptions()
                           .dtype(torch::kUInt8)
                           .layout(torch::kStrided)
                           .requires_grad(false);
    auto cc_in_tensor = torch::from_blob(raw_data.data(), {16, 16}, options).cuda();
    auto kernel = cc_in_tensor.to(torch::kInt32);

    raw_data[1 * 16 + 0] = 255;
    raw_data[9] = 255;
    auto region_tensor = torch::from_blob(raw_data.data(), {16, 16}, options).cuda();
    auto region = (region_tensor > 0).to(torch::kInt32);
    auto out_kernel = cuda::pixel_aggregate(kernel, region, false);
    std::cout << "region" << region.cpu() << std::endl
              << "kernel" << kernel.cpu() << std::endl
              << "out_kernel " << out_kernel.cpu() << std::endl;

    ASSERT_TRUE(out_kernel.sizes() == kernel.sizes());
}