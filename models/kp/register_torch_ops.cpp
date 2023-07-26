//
// Created by liupeng on 2021/6/11.
//

#include <torch/torch.h>
static auto registry_tfm_mha = torch::RegisterOperators().op(
        "tfm::mha(Tensor input, int num_heads, int head_dim) -> Tensor",
        [](const torch::Tensor &input, int64_t N, int64_t H) {
            auto input_shape = input.sizes();
            auto B = input_shape[0];
            auto S = input_shape[1];

            auto x = input.reshape({B, S, 3, N, H}).permute({2, 0, 3, 1, 4});
            auto q = x[0];
            auto k = x[1];
            auto v = x[2];
            auto p = q.matmul(k.transpose(-2, -1)) * (1 / sqrt(H));
            p = p.softmax(-1);
            return p.matmul(v).transpose(1, 2).reshape({B, S, -1});
        });

static auto registry_tfm_skipln = torch::RegisterOperators().op(
        "tfm::skip_ln(Tensor input, Tensor skip, int[] normalized_shape, Tensor? gamma, Tensor? beta, float eps) -> Tensor",
        [](const torch::Tensor &input,
           const torch::Tensor &skip,
           torch::IntArrayRef normalized_shape,
           const c10::optional<torch::Tensor> &weight = {},
           const c10::optional<torch::Tensor> &bias = {},
           double eps = 1e-05) {
            auto x = input + skip;
            return torch::layer_norm(x, normalized_shape, weight, bias, eps, true);
        });

/**
 * Using in TSM model.
 * def shift(x, n_segment, fold_div=3):
    nt, c, h, w = x.size()
    n_batch = int(nt / n_segment)
    x = x.view(n_batch, n_segment, c, h, w)
    fold = int(c / fold_div)
    left_side = torch.cat((x[:, 1:, :fold], torch.zeros(n_batch, 1, fold, h, w).to(x.device)), dim=1)
    middle_side = torch.cat((torch.zeros(n_batch, 1, fold, h, w).to(x.device), x[:, :n_segment - 1, fold: 2 * fold]),
 dim=1) out = torch.cat((left_side, middle_side, x[:, :, 2 * fold:]), dim=2) return out.view(nt, c, h, w)
 */
static auto registry_tsm_temporal_shift = torch::RegisterOperators().op(
        "tsm::temporal_shift(Tensor input, int n_segment, int fold_div) -> Tensor",
        [](const torch::Tensor &in, int64_t n_segment, int64_t fold_div) {
            auto in_shape = in.sizes();
            int nt = in_shape.at(0);
            int c = in_shape.at(1);
            int h = in_shape.at(2);
            int w = in_shape.at(3);
            auto n_batch = nt / n_segment;
            auto x = in.view({n_batch, n_segment, c, h, w});
            auto fold = c / fold_div;

            using namespace torch::indexing;
            auto left_side = torch::cat(
                    {x.index({Slice(), Slice(1), Slice(0, fold)}), torch::zeros({n_batch, 1, fold, h, w}, x.options())}, 1);
            auto middle_side = torch::cat(
                    {torch::zeros({n_batch, 1, fold, h, w}, x.options()),
                     x.index({Slice(), Slice(0, n_segment - 1), Slice(fold, fold * 2)})},
                    1);
            auto out = torch::cat({left_side, middle_side, x.index({Slice(), Slice(), Slice(2 * fold)})}, 2);
            return out.view({nt, c, h, w});
        });

/**
 * # x shape [N, C, H, W] => [N, 4C, H/2, W/2]
 * torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
 */
static auto registry_yolo_focus =
        torch::RegisterOperators().op("yolo::focus(Tensor input) -> Tensor", [](const torch::Tensor &x) {
            using namespace torch::indexing;
            return torch::cat(
                    {
                            x.index({Slice(), Slice(), Slice(None, None, 2), Slice(None, None, 2)}),
                            x.index({Slice(), Slice(), Slice(1, None, 2), Slice(None, None, 2)}),
                            x.index({Slice(), Slice(), Slice(None, None, 2), Slice(1, None, 2)}),
                            x.index({Slice(), Slice(), Slice(1, None, 2), Slice(1, None, 2)}),
                    },
                    1);
        });

/**
* def anchor_decode(x: List[torch.Tensor], anchor_grid, grid, stride, na, no) -> torch.Tensor:
    z = []
    for i in range(len(x)):
        bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x[i] = x[i].view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        y = x[i].sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i].to(x[i].device)) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        z.append(y.view(bs, -1, no))
    return torch.cat(z, 1)
*/
static auto registry_yolo_anchor_decode = torch::RegisterOperators().op(
        "yolo::anchor_decode(Tensor[] inputs, Tensor anchor_grid, Tensor[] grid, Tensor stride, int na, int no) -> Tensor",
        [](const torch::TensorList in_tensor_list,
           torch::Tensor &anchor_grid,
           torch::TensorList grid_list,
           torch::Tensor &stride,
           int64_t na,
           int64_t no) {
            std::vector<torch::Tensor> results;
            results.reserve((in_tensor_list.size()));
            for (int i = 0; i < in_tensor_list.size(); i++) {
                auto xi = in_tensor_list[i];
                auto y = xi.sigmoid();
                auto in_shape = y.sizes();
                int bs = in_shape[0];
                int ny = in_shape[2];
                int nx = in_shape[3];

                y = y.view({bs, na, no, ny, nx}).permute({0, 1, 3, 4, 2}).contiguous();

                using namespace torch::indexing;

                auto a0 =
                        (y.index({Slice(), Slice(), Slice(), Slice(), Slice(None, 2)}) * 2.0 - 0.5 + grid_list[i].to(xi.device())) *
                        stride[i];
                auto a1 = (y.index({Slice(), Slice(), Slice(), Slice(), Slice(2, 4)}) * 2).pow(2) * anchor_grid[i];
                auto a2 = y.index({Slice(), Slice(), Slice(), Slice(), Slice(4)});

                results.push_back(torch::cat({a0, a1, a2}, -1).view({bs, -1, no}));
            }
            return torch::cat(results, 1);
        });

//#include <ops/deform_conv2d.h>
//static auto registry_dcn = torch::RegisterOperators().op(
//        "kinfer::deform_conv2d(Tensor input, Tensor weight, Tensor offset_mask, Tensor? bias, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups, int offset_groups, bool use_mask) -> (Tensor)",
//        [](const torch::Tensor &input,
//           const torch::Tensor &weight,
//           const torch::Tensor &offset_mask,
//           const c10::optional<torch::Tensor> &bias_opt,
//           int64_t stride_h,
//           int64_t stride_w,
//           int64_t pad_h,
//           int64_t pad_w,
//           int64_t dilation_h,
//           int64_t dilation_w,
//           int64_t groups,
//           int64_t offset_groups,
//           bool use_mask) {
//            auto offset_mask_shape = offset_mask.sizes();
//            int c = offset_mask_shape[1];
//            at::Tensor offset;
//            at::Tensor mask;
//
//            if (use_mask) {
//                TORCH_CHECK(c % 3 == 0, "offset_mask chanel should be divided by 3.")
//                using namespace torch::indexing;
//                auto offset_channel = c / 3 * 2;
//                offset = offset_mask.index({Slice(), Slice(0, offset_channel), Slice(), Slice()});
//                mask = offset_mask.index({Slice(), Slice(offset_channel), Slice(), Slice()}).sigmoid();
//            } else {
//                offset = offset_mask;
//                mask = at::ones({offset_mask_shape[0], c / 3, offset_mask_shape[2], offset_mask_shape[3]}, offset.options());
//            }
//
//            //      auto bias_tensor = c10::value_or_else(bias_opt, [] { return torch::zeros({weight.size(0)}, weight.options());});
//            auto bias_tensor = bias_opt.has_value() ? *bias_opt : torch::zeros({weight.size(0)}, weight.options());
//
//            return vision::ops::deform_conv2d(
//                    input,
//                    weight,
//                    offset,
//                    mask,
//                    bias_tensor,
//                    stride_h,
//                    stride_w,
//                    pad_h,
//                    pad_w,
//                    dilation_h,
//                    dilation_w,
//                    groups,
//                    offset_groups,
//                    use_mask);
//        });
