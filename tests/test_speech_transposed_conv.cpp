#include "test_assert.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstring>
#include <vector>

namespace {

ggml_tensor* transpose_2d(ggml_context* ctx, ggml_tensor* input) {
    return ggml_cont(ctx, ggml_transpose(ctx, input));
}

ggml_tensor* repeat_rows(
    ggml_context* ctx,
    ggml_tensor* vector,
    ggml_tensor* like) {
    ggml_tensor* row = ggml_reshape_2d(ctx, vector, 1, vector->ne[0]);
    return ggml_repeat(ctx, row, like);
}

ggml_tensor* original_conv(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* kernel,
    ggml_tensor* bias,
    int stride) {
    ggml_tensor* raw = ggml_conv_transpose_1d(
        ctx, kernel, input, stride, 0, 1);
    const int64_t output_length = input->ne[0] * stride;
    ggml_tensor* output = ggml_cont_2d(
        ctx,
        ggml_view_2d(
            ctx,
            raw,
            output_length,
            raw->ne[1],
            raw->nb[1],
            0),
        output_length,
        raw->ne[1]);
    return ggml_add(ctx, output, repeat_rows(ctx, bias, output));
}

ggml_tensor* decomposed_conv(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* kernel,
    ggml_tensor* bias,
    int stride) {
    const int64_t input_length = input->ne[0];
    const int64_t input_channels = input->ne[1];
    const int64_t output_length = input_length * stride;
    const int64_t output_channels = kernel->ne[1];
    ggml_tensor* input_transposed = transpose_2d(ctx, input);
    ggml_tensor* phases = nullptr;
    for (int phase = 0; phase < stride; ++phase) {
        const auto tap = [&](int k, ggml_tensor* source) {
            ggml_tensor* kernel_tap = ggml_view_2d(
                ctx,
                kernel,
                input_channels,
                output_channels,
                kernel->nb[1],
                static_cast<size_t>(k) * kernel->nb[2]);
            return ggml_mul_mat(ctx, kernel_tap, source);
        };

        ggml_tensor* phase_output = nullptr;
        int offset = 0;
        for (int k = phase; k < kernel->ne[2]; k += stride, ++offset) {
            if (offset >= input_length) {
                break;
            }
            ggml_tensor* source = offset == 0
                ? input_transposed
                : ggml_view_2d(
                    ctx,
                    input_transposed,
                    input_channels,
                    input_length - offset,
                    input_transposed->nb[1],
                    0);
            ggml_tensor* contribution = tap(k, source);
            if (offset > 0) {
                contribution = ggml_pad_ext(
                    ctx,
                    contribution,
                    0,
                    0,
                    offset,
                    0,
                    0,
                    0,
                    0,
                    0);
            }
            phase_output = phase_output == nullptr
                ? contribution
                : ggml_add(ctx, phase_output, contribution);
        }
        phase_output = transpose_2d(ctx, phase_output);
        phases = phases == nullptr
            ? phase_output
            : ggml_concat(ctx, phases, phase_output, 2);
    }
    ggml_tensor* output = ggml_cont_2d(
        ctx,
        ggml_permute(ctx, phases, 1, 2, 0, 3),
        output_length,
        output_channels);
    return ggml_add(ctx, output, repeat_rows(ctx, bias, output));
}

} // namespace

int main() {
    constexpr int64_t input_length = 3;
    constexpr int64_t input_channels = 2;
    constexpr int64_t output_channels = 2;
    constexpr int stride = 2;
    constexpr int64_t kernel_size = 4;

    const float input_values[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    const float kernel_values[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f,
    };
    const float bias_values[] = {0.5f, -0.5f};
    std::vector<float> transformed_kernel(
        input_channels * output_channels * kernel_size);
    for (int64_t input_channel = 0;
         input_channel < input_channels;
         ++input_channel) {
        for (int64_t output_channel = 0;
             output_channel < output_channels;
             ++output_channel) {
            for (int64_t k = 0; k < kernel_size; ++k) {
                transformed_kernel[
                    input_channel +
                    input_channels *
                        (output_channel + output_channels * k)] =
                    kernel_values[
                        k +
                        kernel_size *
                            (output_channel +
                             output_channels * input_channel)];
            }
        }
    }

    ggml_init_params params{};
    params.mem_size = 64 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context* ctx = ggml_init(params);
    CHECK(ctx != nullptr);

    ggml_tensor* input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, input_length, input_channels);
    ggml_tensor* kernel = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, kernel_size, output_channels, input_channels);
    ggml_tensor* transformed = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, input_channels, output_channels, kernel_size);
    ggml_tensor* bias = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, output_channels);

    ggml_tensor* expected = original_conv(
        ctx, input, kernel, bias, stride);
    ggml_tensor* actual = decomposed_conv(
        ctx, input, transformed, bias, stride);
    ggml_tensor* difference = ggml_sub(ctx, expected, actual);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, difference);

    // 所有张量建好之后再一次性分配：alloc_ctx_tensors 只覆盖此刻已存在的张量。
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(
        GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(backend != nullptr);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer != nullptr);

    std::memcpy(input->data, input_values, sizeof(input_values));
    std::memcpy(kernel->data, kernel_values, sizeof(kernel_values));
    std::memcpy(
        transformed->data,
        transformed_kernel.data(),
        transformed_kernel.size() * sizeof(float));
    std::memcpy(bias->data, bias_values, sizeof(bias_values));

    CHECK_EQ_I64(
        ggml_backend_graph_compute(backend, graph),
        GGML_STATUS_SUCCESS);
    CHECK_EQ_I64(expected->ne[0], actual->ne[0]);
    CHECK_EQ_I64(expected->ne[1], actual->ne[1]);

    const float* expected_values =
        static_cast<const float*>(expected->data);
    const float* actual_values =
        static_cast<const float*>(actual->data);
    for (int64_t i = 0; i < ggml_nelements(expected); ++i) {
        CHECK_NEAR(actual_values[i], expected_values[i], 1e-5f);
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
