//
// Created by liupeng on 2021/9/12.
//
#include "kprocess.h"
#include <gtest/gtest.h>


TEST(test_masked_roi_aling, test_masked_roi_random) {

    //    masked_roi(at::Tensor image, at::Tensor bboxes, at::Tensor mask, at::Tensor labels, int64_t crop_h, int64_t crop_w);

    //    cnpy::NpyArray arr = cnpy::npy_load("/Users/liupeng/code/pan_pp/image.npy");
    //    auto arr = cnpy::npy_load("/Users/liupeng/code/pan_pp/label.npy");
    int c = 1;
    int h = 16;
    int w = 16;
    int num_boxes = 1;
    int crop_h = 4;
    int crop_w = 4;


    std::vector<float> raw_data = {
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1,
            0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    auto options = at::TensorOptions()
                           .dtype(torch::kFloat32)
                           .layout(torch::kStrided)
                           .requires_grad(false);
    auto image = torch::from_blob(raw_data.data(), {1, 16, 16}, options).cuda();

    std::vector<float> bboxex_vec({{1.0f / h, 1.0f / w, 3.0f / h, 3.0f / w}});
    at::Tensor bboxes = torch::from_blob(bboxex_vec.data(), {num_boxes, 4}, {at::kFloat}).cuda();

    at::Tensor mask = torch::ones({h, w}, {at::kInt}).cuda();
    at::Tensor labels = torch::ones({1}, {at::kInt}).cuda();
    at::Tensor out = cuda::masked_roi(image, bboxes, mask, labels, crop_h, crop_w);

    std::cout << "image: " << std::endl
              << image.cpu() << std::endl
              << "bbox: " << bboxes.cpu() << std::endl
              << "mask: " << mask.cpu() << std::endl
              << "labels: " << labels.cpu() << std::endl
              << "out: " << out.sizes() << std::endl
              << out.cpu() << std::endl;
    ASSERT_TRUE(true);
}

TEST(test_masked_roi_aling, test_masked_roi_align) {
    int c = 1;
    int h = 16;
    int w = 16;
    int num_boxes = 2;
    int crop_h = 2;
    int crop_w = 4;

    std::vector<float> raw_data = {
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 2, 3, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 4, 5, 6, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 7, 8, 9, 0, 0, 1, 1, 1, 1, 0, 0, 19, 0, 0, 0,
            0, 10, 11, 12, 0, 0, 0, 1, 1, 1, 0, 0, 0, 10, 11, 12,
            0, 13, 14, 15, 0, 0, 0, 0, 1, 0, 0, 0, 0, 13, 14, 15,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 17, 18,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1,
            0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    std::vector<int> mask_raw_data = {
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 2, 2, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 2, 2, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 2, 2, 2, 0, 0, 1, 1, 1, 1, 0, 0, 3, 0, 0, 0,
            0, 2, 2, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 3, 3, 3,
            0, 2, 2, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 3, 3,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1,
            0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    auto options = at::TensorOptions()
                           .dtype(torch::kFloat32)
                           .layout(torch::kStrided)
                           .requires_grad(false);
    auto image = torch::from_blob(raw_data.data(), {c, h, w}, options).cuda();

    //    std::vector<float> bboxex_vec({{1.0f / h, 1.0f / w, 3.0f / h, 3.0f / w}});
    std::vector<int> bboxex_vec({1, 1, 5, 3, 3, 12, 6, 15});
    at::Tensor bboxes = torch::from_blob(bboxex_vec.data(), {num_boxes, 4}, {at::kInt}).cuda();
    at::Tensor mask = torch::from_blob(mask_raw_data.data(), {h, w}, {at::kInt}).cuda();
    at::Tensor labels = torch::from_blob(std::vector<int32_t>({2, 3}).data(), {num_boxes}, {at::kInt}).cuda();
    at::Tensor out = cuda::masked_roi(image, bboxes, mask, labels, crop_h, crop_w);

    std::cout << "image: " << std::endl
              << image.cpu() << std::endl
              << "bbox: " << bboxes.cpu() << std::endl
              << "mask: " << std::endl
              << mask.cpu() << std::endl
              << "labels: " << labels.cpu() << std::endl
              << "out: " << out.sizes() << std::endl
              << out.cpu() << std::endl;
    ASSERT_TRUE(true);
}