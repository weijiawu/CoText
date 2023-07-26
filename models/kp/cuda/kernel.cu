#include "kernel.h"
#include <stdio.h>

__global__ void qOnePass(int *kernel, const int *region, int *changed, const int h, const int w) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;

    __shared__ int changed_shm;
    if (threadIdx.x == 0)
        changed_shm = 0;
    __syncthreads();

    const int idx = y * w + x;
    //judge if cur pos should be changed based on its value
    if ((region[idx] > 0) && (kernel[idx] == 0)) {//region 中非背景，且非kernel区域
        //judge if cur pos should be changed based on neighbors
        int dx[4] = {0, -1, 1, 0};
        int dy[4] = {-1, 0, 0, 1};
        for (int i = 0; i < 4; i++) {
            const int neighbor_y = y + dy[i];
            const int neighbor_x = x + dx[i];
            if ((neighbor_x >= 0) && (neighbor_x < w) && (neighbor_y >= 0) && (neighbor_y < h)) {
                const int neighbor_val = kernel[neighbor_y * w + neighbor_x];
                if (neighbor_val > 0) {
                    kernel[idx] = neighbor_val;
                    changed_shm = 1;
                    break;
                }
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && changed_shm != 0) {
        *changed = 1;
    }
}


int pixel_aggregation_kernel_launcher(int *kernel, const int *region, const int h, const int w) {
    int *d_changed;
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    const int block_size = 8;
    dim3 block(block_size, block_size);
    dim3 grid((w + block_size - 1) / block_size, (h + block_size - 1) / block_size);
    int h_changed = 1;
    while (h_changed == 1) {
        cudaMemset(d_changed, 0, sizeof(int));
        qOnePass<<<grid, block>>>(kernel, region, d_changed, h, w);
        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        //        printf("iter %d\n", i++);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in pixel_aggregation_kernel_launcher: %s\n", cudaGetErrorString(err));
    }
    CUDA_CHECK(cudaFree(d_changed));
    return 0;
}

///////////////////////////////////crop_resize_kernel/////////////////////////////////////////////
template<typename T>
__global__ void crop_resize_kernel(
        const int nthreads,
        const T *image_ptr,
        const int *boxes_ptr,// int
        const int *mask_ids_ptr,
        const int *roi_mask_ids_ptr,
        int num_boxes,
        int batch,
        int image_height,
        int image_width,
        int crop_height,
        int crop_width,
        int depth,
        float extrapolation_value,
        float *crops_ptr) {
    for (int out_idx = threadIdx.x + blockIdx.x * blockDim.x; out_idx < nthreads; out_idx += blockDim.x * gridDim.x) {
        int idx = out_idx;
        int out_x = idx % crop_width;
        idx /= crop_width;
        int out_y = idx % crop_height;
        idx /= crop_height;
        const int d = idx % depth;
        const int b = idx / depth;
        const int y1 = boxes_ptr[b * 4];
        const int x1 = boxes_ptr[b * 4 + 1];
        const int y2 = boxes_ptr[b * 4 + 2];
        const int x2 = boxes_ptr[b * 4 + 3];
        // each image has num_boxes of boxes, so we simply divide to get the box index.
        const int b_in = b / num_boxes;

        if (b_in < 0 || b_in >= batch) {
            continue;
        }

        float bbox_height = float(y2 - y1) + 1;
        float bbox_width = float(x2 - x1) + 1;
        if (bbox_height > bbox_width * 1.5f) {
            // if h > w * 1.5:  # 判定是否为竖排文字
            int temp = out_y;
            out_y = out_x;
            out_x = temp;

            int c_temp = crop_height;
            crop_height = crop_width;
            crop_width = c_temp;
        }


        float height_scale = (crop_height > 1) ? float(y2 - y1) / float(crop_height - 1) : 0.0f;
        float width_scale = (crop_width > 1) ? float(x2 - x1) / float(crop_width - 1) : 0.0f;

        float in_y = (crop_height > 1) ? y1 + out_y * height_scale : 0.5 * (y1 + y2) * (image_height - 1);
        if (in_y < 0 || in_y > image_height - 1) {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }
        float in_x = (crop_width > 1) ? x1 + out_x * width_scale : 0.5 * (x1 + x2) * (image_width - 1);
        if (in_x < 0 || in_x > image_width - 1) {
            crops_ptr[out_idx] = extrapolation_value;
            continue;
        }

        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;
        const int left_x_index = floorf(in_x);
        const int right_x_index = ceilf(in_x);
        const float x_lerp = in_x - left_x_index;

        int cur_roi_mask_id = roi_mask_ids_ptr[b];

        float top_left = static_cast<float>(image_ptr[((b_in * depth + d) * image_height + top_y_index) * image_width + left_x_index]);
        float top_left_mask_value = (mask_ids_ptr[(b_in * image_height + top_y_index) * image_width + left_x_index] == cur_roi_mask_id ? 1.0f : 0.0f);
        top_left *= top_left_mask_value;

        float top_right = static_cast<float>(image_ptr[((b_in * depth + d) * image_height + top_y_index) * image_width + right_x_index]);
        float top_right_mask_value = (mask_ids_ptr[(b_in * image_height + top_y_index) * image_width + right_x_index] == cur_roi_mask_id ? 1.0f : 0.0f);
        top_right *= top_right_mask_value;

        float bottom_left = static_cast<float>(image_ptr[((b_in * depth + d) * image_height + bottom_y_index) * image_width + left_x_index]);
        float bottom_left_mask_value = (mask_ids_ptr[(b_in * image_height + bottom_y_index) * image_width + left_x_index] == cur_roi_mask_id ? 1.0f : 0.0f);
        bottom_left *= bottom_left_mask_value;

        float bottom_right = static_cast<float>(image_ptr[((b_in * depth + d) * image_height + bottom_y_index) * image_width + right_x_index]);
        float bottom_right_mask_value = (mask_ids_ptr[(b_in * image_height + bottom_y_index) * image_width + right_x_index] == cur_roi_mask_id ? 1.0f : 0.0f);
        bottom_right *= bottom_right_mask_value;

        //        printf("debug kernel: %d - %d - out %d - %d - in %f - %f - scale(%f - %f)- tlbr(%d - %d - %d - %d) - index (%d - %d - %d - %d) - mask(%f - %f - %f - %f)\n", threadIdx.x, cur_roi_mask_id,
        //               out_y, out_x, in_y, in_x, height_scale, width_scale,
        //               top_y_index, left_x_index, bottom_y_index, right_x_index, // tlbr
        //               (b_in * image_height + top_y_index) * image_width + left_x_index,
        //               (b_in * image_height + top_y_index) * image_width + right_x_index,
        //               (b_in * image_height + bottom_y_index) * image_width + left_x_index,
        //               (b_in * image_height + bottom_y_index) * image_width + right_x_index,
        //               top_left_mask_value, top_right_mask_value, bottom_left_mask_value, bottom_right_mask_value);

        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
        crops_ptr[out_idx] = top + (bottom - top) * y_lerp;
    }
}

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
        void *output) {
    // todo: support batch
    constexpr int batch_size = 1;
    int output_volume = batch_size * num_boxes * depth * crop_height * crop_width;
    int block_size = 128;
    int grid_size = (output_volume + block_size - 1) / block_size;
    crop_resize_kernel<float><<<grid_size, block_size, 0, stream>>>(
            output_volume,
            static_cast<const float *>(image),
            static_cast<const int *>(rois),
            mask_ids,
            roi_mask_ids,
            num_boxes,
            batch_size,
            input_height,
            input_width,
            crop_height,
            crop_width,
            depth,
            0.0f,
            static_cast<float *>(output));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in crop_resize_kernel_launcher: %s\n", cudaGetErrorString(err));
    }
    return 0;
}
