// LibTorchProject.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <immintrin.h>

// ----- Convolution algorithms implementations -----
#include "SMM_Conv.hpp"

using namespace torch;
using namespace std;

//#define repeat_count 1
#define repeat_count 50

// ----------------- Testing methods -----------------


void Test_PartialPackAncConv_Stride1_SingleThread_NoSimd()
{
    printf("Test_PartialPackAncConv_Stride1_SingleThread_NoSimd: ");

    const auto options = torch::TensorOptions().dtype(torch::kF32).requires_grad(false);
    const int in_channels = 10;
    const int out_channels = 10; 
    const int input_dim = 10; 
    const int kernel_dim = 3;

    const int out_dim = input_dim - kernel_dim + 1;
    const int kernel_height = kernel_dim;
    const int kernel_width = kernel_dim;

    auto input = torch::randn({ 1, in_channels, input_dim, input_dim }, options).contiguous();
    float* input_ptr = input.data_ptr<float>();
    auto partial_packing_weights = torch::randn({ in_channels, kernel_width, kernel_height, out_channels }, options).contiguous();
    float* partial_packing_weights_ptr = partial_packing_weights.data_ptr<float>();

    // Convert the weights to torch format
    auto torch_weights = torch::zeros({ out_channels, in_channels, kernel_height, kernel_width });
    float* torch_weights_ptr = torch_weights.data_ptr<float>();
    for (int out_channel = 0; out_channel < out_channels; out_channel++)
    {
        for (int in_channel = 0; in_channel < in_channels; in_channel++)
        {
            for (int y = 0; y < kernel_height; y++)
            {
                for (int x = 0; x < kernel_width; x++)
                {
                    float weight = partial_packing_weights[in_channel][x][y][out_channel].item<float>();
                    torch_weights[out_channel][in_channel][y][x] = weight;
                }
            }
        }
    }

    auto partial_packing_results = SMMConv_Stride1_SingleThread_NoSimd(input, partial_packing_weights);

    auto torch_results = torch::nn::functional::conv2d(IN_ARG input, IN_ARG torch_weights);

    float* partial_packing_results_ptr = partial_packing_results.data_ptr<float>();
    float* torch_results_ptr = torch_results.data_ptr<float>();

    for (int i = 0; i < out_channels* out_dim* out_dim; i++)
    {
        if (abs( partial_packing_results_ptr[i] - torch_results_ptr[i] ) > 0.00001)
        {
            printf("Failed\n");
            exit(-1);
        }
    }
    printf("Passed\n");
}


float Duration_Partial_Stride1_MultiThread(const Tensor& input, const Tensor& weights)
{
    // Weights shape (in channels, kernel_height, kernel_width, out channels)
    const int64_t batch_size = input.size(0);
    
    const int64_t input_height = input.size(2);
    const int64_t input_width = input.size(3);
    const int64_t kernel_height = weights.size(1);
    const int64_t kernel_width = weights.size(2);
    const int64_t out_channels = weights.size(3);
    SMMConv_Multithread_Init(batch_size, input_height, input_width, kernel_width, kernel_height, out_channels);

    {
        auto results = SMMConv_Stride1_MultiThread_NoSimd(input, weights);
    }

    auto start = std::clock();
    for (size_t i = 0; i < repeat_count; i++)
    {
        auto results = SMMConv_Stride1_MultiThread_NoSimd(input, weights);
    }
    auto end = std::clock();
    float timeI2C = 1000.0 * (end - start) / CLOCKS_PER_SEC / 1000.0 / repeat_count;
    return timeI2C;
}

float Duration_Partial_Stride1_SingleThread(const Tensor& input, const Tensor& weights)
{
    // Weights shape (in channels, kernel_height, kernel_width, out channels)

    {
        auto results = SMMConv_Stride1_SingleThread_NoSimd(input, weights);
    }

    auto start = std::clock();
    for (size_t i = 0; i < repeat_count; i++)
    {
        auto results = SMMConv_Stride1_SingleThread_NoSimd(input, weights);
    }
    auto end = std::clock();
    float timeI2C = 1000.0 * (end - start) / CLOCKS_PER_SEC / 1000.0 / repeat_count;
    return timeI2C;
}

float Duration_WholePacking_Stride1_SingleThread(const Tensor& input, const Tensor& weights)
{
    // Weights shape (in channels, kernel_height, kernel_width, out channels)

    {
        auto results = WholePackAncConv_Stride1_SingleThread_NoSimd(input, weights);
    }

    auto start = std::clock();
    for (size_t i = 0; i < repeat_count; i++)
    {
        auto results = WholePackAncConv_Stride1_SingleThread_NoSimd(input, weights);
    }
    auto end = std::clock();
    float timeI2C = 1000.0 * (end - start) / CLOCKS_PER_SEC / 1000.0 / repeat_count;
    return timeI2C;
}

float Duration_MatMul_SingleThread(const Tensor& packedTensor, const Tensor& weights)
{
    // Input shape(batch, in channels, height, width)
    // Weights shape(in channels, out channels)

    {
        auto results = PointwiseConvolution_SingleThread_NoSimd(packedTensor, weights);
    }

    auto start = std::clock();
    for (size_t i = 0; i < repeat_count; i++)
    {
        auto results = PointwiseConvolution_SingleThread_NoSimd(packedTensor, weights);
    }
    auto end = std::clock();
    float timeMatMul = 1000.0 * (end - start) / CLOCKS_PER_SEC / 1000.0 / repeat_count;
    return timeMatMul;
}

float Duration_Partial_Packing_without_mul_Stride1_SingleThread(const Tensor& input, int kernel_dim)
{
    int mem_buffer_length = 256 * 256;
    float* mem_buffer = new float[mem_buffer_length];
    auto start = std::clock();
    for (size_t i = 0; i < repeat_count; i++)
    {
        const int batch_size = input.size(0);
        const int input_channels = input.size(1);
        const int H = input.size(2);
        const int W = input.size(3);
        //const int H_out = H - kernel_dim + 1;
        const int W_out = W - kernel_dim + 1;
        const int copy_size = W_out * sizeof(float);

        const auto options = torch::TensorOptions().dtype(torch::kF32).requires_grad(false);

        // Memory allocation
        if (H * W_out > mem_buffer_length)
        {
            delete[] mem_buffer;
            mem_buffer_length = H * W_out;
            mem_buffer = new float[mem_buffer_length];
        }

        // Result tensor (reuses memory buffer)
        auto result = torch::from_blob(mem_buffer, { H, W_out }, options);
        float* out_ptr = mem_buffer;

        for (int img_ix = 0; img_ix < batch_size; img_ix++)
        {
            for (int channel_ix = 0; channel_ix < input_channels; ++channel_ix)
            {
                float* input_matrix_ptr = input[img_ix][channel_ix].data_ptr<float>();

                for (int w_start = 0; w_start < kernel_dim + 1; w_start++)
                {
                    Pack(input_matrix_ptr + w_start, mem_buffer, H, W, W_out, copy_size);
                }
            }
        }
    }
    auto end = std::clock();
    float timeI2C = 1000.0 * (end - start) / CLOCKS_PER_SEC / 1000.0 / repeat_count;

    return timeI2C;
}

// ----------------- Comparing functions -----------------

void PackingSameBuffer_vs_CompareDiffBuffer(int kernel_dim, int in_channels, int out_channels, int input_dim)
{
    std::cout << in_channels << ", " << out_channels << ", " << kernel_dim << ", " << input_dim << ", ";
    auto options = torch::TensorOptions().dtype(torch::kF32).requires_grad(false);

    auto input = torch::randn({ 1, in_channels, input_dim, input_dim }, options).contiguous();
    auto weights = torch::randn({ in_channels, kernel_dim, kernel_dim, out_channels }, options).contiguous();

    float duration_partial_packing = Duration_Partial_Stride1_SingleThread(input, weights);
    std::cout << duration_partial_packing << ", ";

    float duration_whole_packing = Duration_WholePacking_Stride1_SingleThread(input, weights);
    std::cout << duration_whole_packing << std::endl;
}
// ----------------- Insights and general compares -----------------
void PrintHeader()
{
    std::cout << "kernel dim, input channels, out channels, input dim, direct channels last, direct channels first, Im2Col Conv, MEC im2col origin, MEC im2col, Partial packing, Faster" << endl;
}


void PackingSameBuffer_vs_CompareDiffBuffer()
{
    std::cout << "In channels, Out channels, Kernel_dim, Input Dim, Time partial Packing Conv, Time  Whole Packing Conv" << std::endl;
    PackingSameBuffer_vs_CompareDiffBuffer(3, 1, 1, 40);
    PackingSameBuffer_vs_CompareDiffBuffer(3, 1, 1, 80);
    PackingSameBuffer_vs_CompareDiffBuffer(3, 1, 1, 160);

    PackingSameBuffer_vs_CompareDiffBuffer(3, 1, 10, 40);
    PackingSameBuffer_vs_CompareDiffBuffer(3, 1, 10, 80);
    PackingSameBuffer_vs_CompareDiffBuffer(3, 1, 10, 160);

    PackingSameBuffer_vs_CompareDiffBuffer(3, 10, 1, 40);
    PackingSameBuffer_vs_CompareDiffBuffer(3, 10, 1, 80);
    PackingSameBuffer_vs_CompareDiffBuffer(3, 10, 1, 160);

    PackingSameBuffer_vs_CompareDiffBuffer(3, 10, 10, 40);
    PackingSameBuffer_vs_CompareDiffBuffer(3, 10, 10, 80);
    PackingSameBuffer_vs_CompareDiffBuffer(3, 10, 10, 160);
}


// ----------------- Profiling methods -----------------

void profile()
{
    int kernel_dim = 3;
    int in_channels = 32;
    int out_channels = 128;
    int input_dim = 64;

    cout << kernel_dim << ", " << in_channels << ", " << out_channels << ", " << input_dim << ", ";

    auto options = torch::TensorOptions().dtype(torch::kF32).requires_grad(false);
    auto input = torch::randn({ 1, in_channels, input_dim, input_dim }, options).contiguous();
    auto weights = torch::randn({ in_channels * kernel_dim * kernel_dim, out_channels }, options).contiguous();

    int out_pixels = input_dim - kernel_dim + 1;
    
    weights = torch::randn({ in_channels, kernel_dim, kernel_dim, out_channels }, options).contiguous();

    {
        auto results = SMMConv_Stride1_SingleThread_NoSimd(input, weights);
    }

    for (size_t i = 0; i < 500; i++)
    {
        auto results = SMMConv_Stride1_SingleThread_NoSimd(input, weights);
    }

}


int main()
{
    std::cout << "repeat count: " << repeat_count << std::endl;
    std::cout << "Uncomment for different time compare methods." << std::endl;

    // Test results
    Test_PartialPackAncConv_Stride1_SingleThread_NoSimd();
    
    // Atomic operations
    PackingSameBuffer_vs_CompareDiffBuffer();



    /*at::set_num_threads(1);
    at::set_num_interop_threads(1);
    omp_set_num_threads(1);*/
    //profile();
    //system("pause");
}