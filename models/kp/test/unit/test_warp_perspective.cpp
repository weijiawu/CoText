//
// Created by liupeng on 2021/9/11.
//

#include "kprocess.h"
#include <gtest/gtest.h>

TEST(test_warp_perspective, test_warp_perspective_shape) {
//    std::string image_path = "resources/bruce.jpg";
    // HWC
    torch::Tensor image_tensor = torch::rand({600, 400}).to(at::kByte).cuda();
    std::cout << "single deocde: " << std::endl
    << image_tensor.sizes() << std::endl;

    // warp_perspective
    auto height = 64 * 2;
    auto width = 128 * 2;
    std::vector<double> aSrcQuad = {125., 150., 562., 40, 562., 282., 54., 328.};
    auto src_quad_tensor = torch::from_blob(aSrcQuad.data(), {4, 2});
    std::vector<double> aDstQuad = {0., 0., double(width - 1), 0., double(width - 1), double(height - 1), 0., double(height - 1)};
    auto dst_quad_tensor = torch::from_blob(aDstQuad.data(), {4, 2});

    auto warp_out = cuda::warp_perspective(image_tensor, src_quad_tensor, dst_quad_tensor, height, width, 1);
    std::cout << "warp perspective output: " << std::endl << warp_out.sizes() << std::endl;
    ASSERT_TRUE(warp_out.sizes().vec() == std::vector<int64_t>({height, width}));
}