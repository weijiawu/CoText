//
// Created by liupeng on 2021/8/3.
//
#include "../npp_exceptions.h"
#include "c10/cuda/CUDAStream.h"
#include <cuda_runtime.h>
#include <iostream>
#include <npp.h>
#include <opencv_cuda.h>
#include <torch/torch.h>

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"


namespace cuda_op {
    /**
     *
     * @param image_cuda_tensor  [H, W]
     * @param src_quad
     * @param dst_quad
     * @param height
     * @param width
     * @param eInterpolation
     * @return
     */
    at::Tensor torch_warp_perspective(
            at::Tensor image_cuda_tensor,
            at::Tensor src_quad_cpu_tensor,// const double src_quad[4][2],
            at::Tensor dst_quad_cpu_tensor,//  const double dst_quad[4][2],
            int64_t height,
            int64_t width,
            int64_t eInterpolation) {
        TORCH_CHECK(image_cuda_tensor.is_cuda() && !src_quad_cpu_tensor.is_cuda() && !dst_quad_cpu_tensor.is_cuda(), "tensor cuda")
        TORCH_CHECK((src_quad_cpu_tensor.sizes() == dst_quad_cpu_tensor.sizes()) && (dst_quad_cpu_tensor.sizes() == torch::IntArrayRef({4, 2})), "size [4, 2]")
        TORCH_CHECK(src_quad_cpu_tensor.dtype() == at::kFloat, "src_quad_cpu_tensor dtype should be float")

        int device_id = 0;
        auto options = at::TensorOptions()
                               .device(torch::kCUDA, device_id)
                               .dtype(torch::kUInt8)
                               .layout(torch::kStrided)
                               .requires_grad(false);

        int64_t in_channels = 1;
        std::vector<int64_t> output_tensor_shape = {height, width};
        if (image_cuda_tensor.sizes().size() == 3) {
            in_channels = image_cuda_tensor.size(-1);
            output_tensor_shape.push_back(in_channels);
        }

        auto output_tensor = at::empty(output_tensor_shape, options, at::MemoryFormat::Contiguous);
        auto in_shape = image_cuda_tensor.sizes().vec();
        std::vector<int> in_s(in_shape.begin(), in_shape.end());
        auto out_shape = output_tensor.sizes().vec();
        std::vector<int> out_s(out_shape.begin(), out_shape.end());

        cuda_op::DataShape max_input_shape = {1, in_s[2], in_s[0], in_s[1]};
        cuda_op::DataShape max_output_shape = {1, out_s[2], out_s[0], out_s[1]};

        cuda_op::WarpPerspective cuda_warp_perspective(max_input_shape, max_output_shape);
        cuda_op::DataType data_type = cuda_op::DataType::kCV_8U;
        size_t buffer_size = cuda_warp_perspective.calBufferSize(max_input_shape, max_output_shape, data_type);
        void *workspace = nullptr;
        checkCudaErrors(cudaMalloc(&workspace, buffer_size));

        std::vector<float> src_quad(src_quad_cpu_tensor.data_ptr<float>(), src_quad_cpu_tensor.data_ptr<float>() + src_quad_cpu_tensor.numel());
        std::vector<float> dst_quad(dst_quad_cpu_tensor.data_ptr<float>(), dst_quad_cpu_tensor.data_ptr<float>() + dst_quad_cpu_tensor.numel());
        cv::Point2f srcTri[4];
        cv::Point2f dstTri[4];
        for (int i = 0; i < 4; i++) {
            srcTri[i] = cv::Point2f(src_quad[i * 2], src_quad[i * 2 + 1]);
            dstTri[i] = cv::Point2f(dst_quad[i * 2], dst_quad[i * 2 + 1]);
        }
        cv::Mat warpPerspective_mat(3, 3, CV_64FC(in_channels));
        warpPerspective_mat = cv::getPerspectiveTransform(srcTri, dstTri);

        float *trans_matrix = (float *) malloc(9 * sizeof(float));
        float *matrix_workspace = (float *) malloc(9 * sizeof(float));
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                trans_matrix[i * 3 + j] = warpPerspective_mat.at<double>(i, j);
            }
        }

        // 0: INTER_NEAREST, 1: INTER_LINEAR, 2: INTER_CUBIC
        int flags = eInterpolation;
        int borderMode = cv::BORDER_CONSTANT;
        cuda_op::DataFormat data_format = cuda_op::DataFormat::kHWC;
        auto dsize = cv::Size(width, height);
        auto stream = c10::cuda::getCurrentCUDAStream(0).stream();
        int out_channels = in_channels;

        cuda_warp_perspective.infer((const void *const *) (image_cuda_tensor.data_ptr()),
                                    (void **) output_tensor.data_ptr(),
                                    workspace,
                                    trans_matrix,
                                    matrix_workspace,
                                    dsize,
                                    flags,
                                    borderMode,
                                    0,
                                    max_input_shape,
                                    data_format,
                                    data_type,
                                    stream);
        free(trans_matrix);
        free(matrix_workspace);
        cudaFree(workspace);
        return output_tensor;
    }
}// namespace cuda_op

//TORCH_LIBRARY(my_ops, m) {
//    m.def("warp_perspective", npp::warp_perspective);
//}

#ifdef PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_warp_perspective",
          &cuda::warp_perspective,
          py::call_guard<py::gil_scoped_release>(),
          py::return_value_policy::take_ownership,
          R"docdelimiter(
                    NPP warp_perspective
                    Parameters:
                        image_cuda_tensor: uint8, cuda tensor, shape [H, W]
                        src_quad_cpu_tensor: double, cpu tensor, shape [4, 2]
                        dst_quad_cpu_tensor: double, cpu tensor, shape [4, 2]
                        height: int
                        width: int
                        eInterpolation: int, 0: nn, 1: bilinear, 2: bicubic
                    Returns:
                        cuda tensor [height, width]
                )docdelimiter",
          py::arg("image_cuda_tensor"),
          py::arg("src_quad_cpu_tensor"),
          py::arg("dst_quad_cpu_tensor"),
          py::arg("height"),
          py::arg("width"),
          py::arg("eInterpolation"));
}
#endif

int main(int argc, const char **argv) {
    std::string image_path = "/home/liupeng/remote/torchnvjpeg/images/bruce.jpg";
    if (argc > 1) {
        image_path = argv[1];
    }

    // HWC
    torch::Tensor image_tensor = torch::rand({600, 400, 3}).to(at::kByte).cuda().contiguous();
    std::cout << "single deocde: " << std::endl
              << image_tensor.sizes() << std::endl;

    auto height = 64 * 2;
    auto width = 128 * 2;

    std::vector<float> aSrcQuad = {125., 150., 562., 40, 562., 282., 54., 328.};
    auto src_quad_tensor = torch::from_blob(aSrcQuad.data(), {4, 2});
    std::vector<float> aDstQuad = {0., 0., float(width - 1), 0., float(width - 1), float(height - 1), 0., float(height - 1)};
    auto dst_quad_tensor = torch::from_blob(aDstQuad.data(), {4, 2});

    auto out = cuda_op::torch_warp_perspective(image_tensor, src_quad_tensor, dst_quad_tensor, height, width, 1);
    assert(out.is_cuda());
    std::cout << "warp perspective output: " << std::endl
              << out.sizes() << std::endl;
    printf("Done.\n");
}
