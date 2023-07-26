//
// Created by liupeng on 2021/9/10.
//
#include "kprocess.h"
#include "npp_exceptions.h"
#include <pybind11/pybind11.h>

/**
 * inplace op
 * @param kernel
 * @param region
 * @return
 */

namespace cuda {
    using namespace npp;
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
    at::Tensor warp_perspective(
            at::Tensor image_cuda_tensor,
            at::Tensor src_quad_cpu_tensor,// const double src_quad[4][2],
            at::Tensor dst_quad_cpu_tensor,//  const double dst_quad[4][2],
            int64_t height,
            int64_t width,
            int64_t eInterpolation) {
        TORCH_CHECK(image_cuda_tensor.is_cuda() && !src_quad_cpu_tensor.is_cuda() && !dst_quad_cpu_tensor.is_cuda(), "tensor cuda")
        TORCH_CHECK((src_quad_cpu_tensor.sizes() == dst_quad_cpu_tensor.sizes()) && (dst_quad_cpu_tensor.sizes() == torch::IntArrayRef({4, 2})), "size [4, 2]")

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


        NppiRect oSrcROI = {0, 0, int(image_cuda_tensor.size(1)), int(image_cuda_tensor.size(0))};
        NppiRect oDstROI = {0, 0, int(width), int(height)};

        src_quad_cpu_tensor = src_quad_cpu_tensor.to(at::kDouble);
        dst_quad_cpu_tensor = dst_quad_cpu_tensor.to(at::kDouble);
        auto src_quad = static_cast<const double(*)[2]>(src_quad_cpu_tensor.data_ptr());
        auto dst_quad = static_cast<const double(*)[2]>(dst_quad_cpu_tensor.data_ptr());


        if (in_channels == 1) {
            NppiSize oSrcSize = {(int) image_cuda_tensor.size(1), (int) image_cuda_tensor.size(0)};
            NPP_CHECK_NPP(nppiWarpPerspectiveQuad_8u_C1R(
                    (Npp8u *) image_cuda_tensor.data_ptr(),
                    oSrcSize,
                    image_cuda_tensor.size(1) * sizeof(Npp8u),
                    oSrcROI,
                    src_quad,
                    (Npp8u *) output_tensor.data_ptr(),
                    output_tensor.size(1) * sizeof(Npp8u),
                    oDstROI,
                    dst_quad,
                    (int) eInterpolation));
        } else if (in_channels == 3) {
            NppiSize oSrcSize = {(int) image_cuda_tensor.size(1), (int) image_cuda_tensor.size(0)};
            NPP_CHECK_NPP(nppiWarpPerspectiveQuad_8u_C3R(
                    (Npp8u *) image_cuda_tensor.data_ptr(),
                    oSrcSize,
                    image_cuda_tensor.size(1) * sizeof(Npp8u) * 3,
                    oSrcROI,
                    src_quad,
                    (Npp8u *) output_tensor.data_ptr(),
                    output_tensor.size(1) * sizeof(Npp8u) * 3,
                    oDstROI,
                    dst_quad,
                    (int) eInterpolation));
        } else {
            throw std::runtime_error("in_channels should be 1 or 3");
        }
        return output_tensor;
    }
    /**
     * map to cv::connectedComponents
     * @param image
     * @param connectivity: [4, 8], Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity.
     * @return
     */
    at::Tensor connectedComponents(at::Tensor image, int64_t connectivity) {
        TORCH_CHECK(image.dim() == 2, "image shape should be [h, w]")
        TORCH_CHECK(image.dtype() == torch::kUInt8, "image dtype should be uint8")
        int height = image.size(0);
        int width = image.size(1);

        image = image.contiguous(c10::MemoryFormat::Contiguous);
        int device_id = 0;
        auto options = at::TensorOptions()
                               .device(torch::kCUDA, device_id)
                               .dtype(torch::kInt32)
                               .layout(torch::kStrided)
                               .requires_grad(false);
        at::Tensor out_tensor = at::empty_like(image, options, c10::MemoryFormat::Contiguous);

        NppiSize source_roi = {width, height};
        int buffer_size;
        int compress_buffer_size;
        Npp8u *buffer;
#if CUDA_VERSION >= 11000
        NPP_CHECK_NPP(nppiLabelMarkersUFGetBufferSize_32u_C1R(source_roi, &buffer_size));
        NPP_CHECK_NPP(nppiCompressMarkerLabelsGetBufferSize_32u_C1R(width * height, &compress_buffer_size));
        NPP_CHECK_CUDA(cudaMalloc((void **) &buffer, std::max(buffer_size, compress_buffer_size)));
#else
        NPP_CHECK_NPP(nppiLabelMarkersGetBufferSize_8u32u_C1R(source_roi, &buffer_size));
        NPP_CHECK_CUDA(cudaMalloc((void **) &buffer, buffer_size));
#endif

        auto d_src = (Npp8u *) image.data_ptr();
        auto d_dst = (Npp32u *) out_tensor.data_ptr();
        /**
         * eNorm	Type of pixel connectivity test to use, nppiNormInf will use 8 way connectivity and nppiNormL1 will use 4 way connectivity.
         */
        NppiNorm eNorm;
        if (connectivity == 4) {
            eNorm = nppiNormL1;
        } else if (connectivity == 8) {
            eNorm = nppiNormInf;
        } else {
            throw std::runtime_error("connectivity should be 4 or 8.");
        }

        int maxLabel = 0;
#if CUDA_VERSION >= 11000
        NPP_CHECK_NPP(nppiLabelMarkersUF_8u32u_C1R(
                d_src,
                sizeof(Npp8u) * width,
                d_dst,
                sizeof(Npp32u) * width,
                source_roi,
                eNorm,
                buffer));
#else
        NPP_CHECK_NPP(nppiLabelMarkers_8u32u_C1R(
                d_src,
                sizeof(Npp8u) * width,
                d_dst,
                sizeof(Npp32u) * width,
                source_roi,
                0,
                eNorm,
                &maxLabel,
                buffer));
#endif

#if CUDA_VERSION >= 11000
        NPP_CHECK_NPP(nppiCompressMarkerLabelsUF_32u_C1IR(
                d_dst,
                sizeof(Npp32u) * width,
                source_roi,
                source_roi.width * source_roi.height,
                &maxLabel,
                buffer));
        NPP_CHECK_CUDA(cudaFree(buffer));
        return out_tensor;
#else
        // Get necessary scratch buffer size and allocate that much device memory
        NPP_CHECK_NPP(nppiCompressMarkerLabelsGetBufferSize_32u8u_C1R(maxLabel, &compress_buffer_size));
        if (compress_buffer_size > buffer_size) {
            NPP_CHECK_CUDA(cudaFree(buffer));
            NPP_CHECK_CUDA(cudaMalloc((void **) &buffer, compress_buffer_size));
        }

        auto u8_options = at::TensorOptions()
                                  .device(torch::kCUDA, device_id)
                                  .dtype(torch::kUInt8)
                                  .layout(torch::kStrided)
                                  .requires_grad(false);
        at::Tensor u8_out_tensor = at::empty_like(image, u8_options, c10::MemoryFormat::Contiguous);

        NPP_CHECK_NPP(nppiCompressMarkerLabels_32u8u_C1R(
                d_dst,
                sizeof(Npp32u) * width,
                (Npp8u *) u8_out_tensor.data_ptr(),
                sizeof(Npp8u) * width,
                source_roi,
                maxLabel,
                &maxLabel,
                buffer));

        NPP_CHECK_CUDA(cudaFree(buffer));
        return u8_out_tensor;
#endif
    }

    at::Tensor pixel_aggregate(at::Tensor kernel, at::Tensor region, bool in_place) {
        TORCH_CHECK((kernel.dim() == 2) && (region.dim() == 2), "kernel and region shape should be [H, W]")
        TORCH_CHECK(kernel.dtype() == torch::kInt32, "kernel dtype should be int")
        auto out_tensor = kernel;
        if (!in_place) {
            out_tensor = kernel.clone();
        }
        pixel_aggregation_kernel_launcher(out_tensor.data_ptr<int>(), region.data_ptr<int>(), kernel.size(0), kernel.size(1));
        return out_tensor;
    }

    at::Tensor masked_roi(at::Tensor image, at::Tensor bboxes, at::Tensor mask, at::Tensor labels, int64_t crop_h, int64_t crop_w) {
        TORCH_CHECK(image.dim() == 3 && bboxes.dim() == 2 && mask.dim() == 2 && labels.dim() == 1, "dim of image, bboxes, mask should be 2, dim of label should be 1");
        TORCH_CHECK(bboxes.size(1) == 4, "bboxes should be 4 dim")
        TORCH_CHECK(image.dtype() == torch::kFloat32 && bboxes.dtype() == torch::kInt32 && mask.dtype() == torch::kInt32 && labels.dtype() == torch::kInt32, "data type error")

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(0).stream();

        at::Tensor out_tensor = at::empty({bboxes.size(0), image.size(0), crop_h, crop_w}, image.options());
        int n = out_tensor.numel();
        crop_resize_kernel_launcher(stream,
                                    n,
                                    (float *) image.data_ptr(),
                                    (int *) bboxes.data_ptr(),
                                    (int *) mask.data_ptr(),
                                    (int *) labels.data_ptr(),
                                    image.size(1),
                                    image.size(2),
                                    bboxes.size(0),
                                    crop_h,
                                    crop_w,
                                    image.size(0),
                                    (float *) out_tensor.data_ptr());

        return out_tensor;
    }
};// namespace cuda

namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("warp_perspective",
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
          py::arg("eInterpolation"))
            .def("connectedComponents",
                 &cuda::connectedComponents,
                 py::call_guard<py::gil_scoped_release>(),
                 py::return_value_policy::take_ownership,
                 R"docdelimiter(
               NPP connectedComponents(cv2::connectedComponents)
               Parameters:
                   image: uint8, cuda tensor, shape [H, W]
                   connectivity: int, 4 or 8
               Returns:
                   cuda tensor [height, width]
               )docdelimiter",
                 py::arg("image"),
                 py::arg("connectivity") = 4)
            .def("pixel_aggregation",
                 &cuda::pixel_aggregate,
                 py::call_guard<py::gil_scoped_release>(),
                 py::return_value_policy::take_ownership,
                 R"docdelimiter(
                   pixel_aggregation)
                   Parameters:
                       kernel: int32, cuda tensor, shape [H, W]
                       region: int32, cuda tensor, shape [H, W]
                       in_place: bool, false by default
                   Returns:
                       int32 cuda tensor, shape [H,W]
                    )docdelimiter",
                 py::arg("kernel"),
                 py::arg("region"),
                 py::arg("in_place") = false)
            .def("masked_roi",
                 &cuda::masked_roi,
                 py::call_guard<py::gil_scoped_release>(),
                 py::return_value_policy::take_ownership,
                 R"docdelimiter(
                        masked roi
                        Parameters:
                            image: float32 cuda tensor, shape [C, H, W]
                            bboxes: float32 cuda tensor, shape [num_boxes, 4]
                            mask: int32 cuda tenssor, shape [H, W]
                            labels: int32 cuda tensor, shape [num_boxes]
                            crop_h: int64_t
                            crop_w: int64_t
                    )docdelimiter",
                 py::arg("image"),
                 py::arg("bboxes"),
                 py::arg("mask"),
                 py::arg("labels"),
                 py::arg("crop_h"),
                 py::arg("crop_w"));
}
