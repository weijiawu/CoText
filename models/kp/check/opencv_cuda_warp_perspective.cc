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
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/cudawarping.hpp"


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

//        cuda_op::DataType data_type = cuda_op::DataType::kCV_8U;

        auto data_type = CV_8U;
        int device_id = 0;

        int64_t in_channels = 1;
        std::vector<int64_t> output_tensor_shape = {height, width};
        if (image_cuda_tensor.sizes().size() == 3) {
            in_channels = image_cuda_tensor.size(-1);
            output_tensor_shape.push_back(in_channels);
        }
        auto output_tensor = at::empty(output_tensor_shape, image_cuda_tensor.options(), at::MemoryFormat::Contiguous);
        int out_channels = in_channels;

        std::vector<float> src_quad(src_quad_cpu_tensor.data_ptr<float>(), src_quad_cpu_tensor.data_ptr<float>() + src_quad_cpu_tensor.numel());
        std::vector<float> dst_quad(dst_quad_cpu_tensor.data_ptr<float>(), dst_quad_cpu_tensor.data_ptr<float>() + dst_quad_cpu_tensor.numel());
        cv::Point2f srcTri[4];
        cv::Point2f dstTri[4];
        for (int i = 0; i < 4; i++) {
            srcTri[i] = cv::Point2f(src_quad[i * 2], src_quad[i * 2 + 1]);
            dstTri[i] = cv::Point2f(dst_quad[i * 2], dst_quad[i * 2 + 1]);
        }

        cv::Mat warpPerspective_mat(3, 3, CV_64FC(out_channels));
        warpPerspective_mat = cv::getPerspectiveTransform(srcTri, dstTri);

        // 0: INTER_NEAREST, 1: INTER_LINEAR, 2: INTER_CUBIC
        int flags = eInterpolation;
        int borderMode = cv::BORDER_CONSTANT;
//        cv::Scalar value(255);
//        if (out_channels == 3) {
//            cv::Scalar value(255, 255, 255);
//        }
        auto dsize = cv::Size(width, height);
        auto stream = c10::cuda::getCurrentCUDAStream(device_id).stream();

        cv::cuda::GpuMat in_mat(image_cuda_tensor.size(0), image_cuda_tensor.size(1), CV_MAKETYPE(data_type, out_channels), image_cuda_tensor.data_ptr<u_int8_t>());
        cv::cuda::GpuMat out_mat(output_tensor.size(0), output_tensor.size(1), CV_MAKETYPE(data_type, out_channels), output_tensor.data_ptr<u_int8_t>());
        cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);
        if (image_cuda_tensor.is_cuda()) {
            cv::cuda::warpPerspective(in_mat, out_mat, warpPerspective_mat, dsize, flags, borderMode, 0, cv_stream);
        }
        return output_tensor;
    }
}// namespace cuda_op

//TORCH_LIBRARY(my_ops, m) {
//    m.def("warp_perspective", npp::warp_perspective);
//}

#ifdef PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_warp_perspective",
          &cuda_op::torch_warp_perspective,
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
    torch::Tensor image_tensor = torch::rand({600, 400}).to(at::kByte).cuda().contiguous();
    std::cout << "single deocde: " << std::endl
              << image_tensor.sizes() << std::endl;

    auto height = 64 * 2;
    auto width = 128 * 2;

    std::vector<float> aSrcQuad = {125., 150., 562., 40, 562., 282., 54., 328.};
    auto src_quad_tensor = torch::from_blob(aSrcQuad.data(), {4, 2});
    std::vector<float> aDstQuad = {0., 0., float(width - 1), 0., float(width - 1), float(height - 1), 0., float(height - 1)};
    auto dst_quad_tensor = torch::from_blob(aDstQuad.data(), {4, 2});

    auto out = cuda_op::torch_warp_perspective(image_tensor, src_quad_tensor, dst_quad_tensor, height, width, 0);
    assert(out.is_cuda());
    std::cout << "warp perspective output: " << std::endl
              << out.sizes() << std::endl;
    printf("Done.\n");
}
