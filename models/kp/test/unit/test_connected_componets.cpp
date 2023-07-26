//
// Created by liupeng on 2021/9/11.
//
#include "kprocess.h"
#include <gtest/gtest.h>

TEST(test_connected_componets, test_connecter_componets_shape) {
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

    auto cc_out_tensor = cuda::connectedComponents(cc_in_tensor, 4);
    auto cc_out_data = cc_out_tensor.cpu();
    std::cout << cc_out_data << std::endl;
    ASSERT_TRUE(cc_out_tensor.sizes() == cc_in_tensor.sizes());
}