#pragma once
#include <thread>
#include <mutex>
#include <vector>
#include <future>
#include <xmmintrin.h>
#include <torch/torch.h>
#include <condition_variable>

#define IN_ARG
#define OUT_ARG
#define IN_OUT_ARG

#ifndef min_
#define min_(a,b)            (((a) > (b)) ? (b) : (a))
#define max_(a,b)            (((a) > (b)) ? (a) : (b))
#endif

using namespace torch;


inline void MulAdd(float* target, const float* src, float scalar, int64_t length)
{
    for (int64_t l = 0; l < length; ++l)
    {
        target[l] += src[l] * scalar;
    }
}

/*
Multiplies and adds into N vectors a vector with N scalars
Multiplies and adds into output channel result a matrix with scalar
*/
inline void MulAddMultiple(float* targets, const float* src, const float *scalars, int64_t num_targets, int64_t src_size)
{
    while (num_targets--)
    {
        MulAdd(targets, src, *scalars, src_size);

        targets += src_size;
        scalars++;
    }
}

inline void MulAddMultiple_SIMD(float* target_arrays, const float* source_array, const float* scalars, int64_t num_targets, int64_t src_size)
{
    int mul_in_256 = src_size / 8;
    __m256* dst = reinterpret_cast<__m256*>(target_arrays);
    const __m256* src = reinterpret_cast<const __m256*>(source_array);

    while (num_targets--)
    {
        float scalar = scalars[0];
        scalars += 1;
        __m256 scalar_vec = _mm256_set_ps(scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar);

        //__m256 tmp;
        for (int i = 0; i < mul_in_256; ++i)
        {
            //__m256 dst_256 = *dst;
            //dst_256 = _mm256_add_ps(dst_256, _mm256_mul_ps(scalar_vec, src[i]));
            dst[i] = _mm256_fmadd_ps(src[i], scalar_vec, dst[i]);
            //dst[0] = dst_256;
            //dst += 1;
        }
    }
}

inline void MulAddMultiple_SIMD(__m256* target_arrays, const __m256* source_array, const float* scalars, int64_t num_targets, int64_t src_size)
{
    while (num_targets--)
    {
        float scalar = scalars[0];
        scalars += 1;
        __m256 scalar_vec = _mm256_set_ps(scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar);

        for (int i = 0; i < src_size; ++i)
        {
            target_arrays[i] = _mm256_fmadd_ps(source_array[i], scalar_vec, target_arrays[i]);
        }
        target_arrays += src_size;
    }
}

/* Pack
* *****
* Packing method for Partial Packing
  This method could be used for convolution with transposed standard unit vector ([1,0,0], [0,1,0], [0,0,1])
  
  input_ptr:    In a matrix, the address of the first cell to copy. For example matrix_ptr+1 for convolution with [0,1,0] 
  output_ptr:   Where to store the data. It should have height * width_out * sizeof(float) buffer size. AKA smm_pack_buffer
  height:       The height of the input.
  width:        The width of the input.
  width_out:    The width of the result matrix.
  copy_size:    Calculated to width_out * sizeof(float).
*/
inline void Pack(const float *input_ptr, float* output_ptr, int64_t height, int64_t width, int64_t width_out, size_t copy_size)
{
    for (int64_t h = 0; h < height; ++h)
    {
        std::memcpy(output_ptr, input_ptr, copy_size);

        // Pointer arithmetics
        output_ptr += width_out;
        input_ptr += width;
    }
}


/* PartialPackAndConv_Stride1_SingleThread_NoSimd
* ************************************************
* Input shape (batch, in channels, height, width)
* Weights shape (in channels, kernel_width, kernel_height, out channels)
*/
// Members used for SMMConv_Stride1_SingleThread_NoSimd
int64_t smm_pack_buffer_length = 512 * 512;
float* smm_pack_buffer = new float[smm_pack_buffer_length];
Tensor SMMConv_Stride1_SingleThread_NoSimd(const Tensor& input, const Tensor& weights)
{
    const int64_t batch_size = input.size(0);
    const int64_t in_channels = input.size(1);
    const int64_t height = input.size(2);
    const int64_t width = input.size(3);
    const int64_t kernel_height = weights.size(1);
    const int64_t kernel_width = weights.size(2);
    const int64_t out_channels = weights.size(3);
    const int64_t width_out = width - kernel_width + 1;
    const int64_t height_out = height - kernel_height + 1;
    const size_t copy_size = sizeof(float) * width_out;

    const int64_t in_length = height * width;
    const int64_t out_length = height_out * width_out;

    auto options = torch::TensorOptions().dtype(torch::kF32).requires_grad(false);
    Tensor result = torch::zeros({ batch_size, out_channels, height_out, width_out }, options).contiguous();

    if (height * width_out > smm_pack_buffer_length)
    {
        _freea(smm_pack_buffer);
        smm_pack_buffer_length = height * width_out;
        smm_pack_buffer = (float*)_malloca(sizeof(float) * smm_pack_buffer_length);
    }

    const float* bufferCurr;
    for (int64_t batch = 0; batch < batch_size; ++batch)
    {
        const float* currWeight = weights.data_ptr<float>();
        float* outPtr = result[batch].data_ptr<float>();
        float* inputPtr = input[batch].data_ptr<float>();
        for (int64_t in_channel = 0; in_channel < in_channels; ++in_channel)
        {
            // Seperable convolution: For one of [1,0,0], [0,1,0], [0,0,1]
            for (int64_t offset_x = 0; offset_x < kernel_width; ++offset_x)
            {
                // ===== Packing =====
                Pack(inputPtr + offset_x, smm_pack_buffer, height, width, width_out, copy_size);

                // ===== Multiplying =====
                bufferCurr = smm_pack_buffer;
                for (int64_t offset_y = 0; offset_y < kernel_height; ++offset_y)
                {
                    // Multiply and add
                    // For every in channel we will run on every out channel
                    MulAddMultiple(OUT_ARG outPtr, IN_ARG bufferCurr, IN_ARG currWeight, out_channels, out_length);

                    // Pointer arithmetics
                    currWeight += out_channels;
                    bufferCurr += width_out;
                }
            }

            inputPtr += in_length;
        }
    }

    return result;
}


/* WholePackAndConv_Stride1_SingleThread_NoSimd
* ************************************************
* The difference is that we completly pack (i.e. K H*W' matrices)
* Input shape(batch, in channels, height, width)
* Weights shape (in channels, kernel_width, kernel_height, out channels)
*/
// Members used for WholePackAncConv_Stride1_SingleThread_NoSimd
int64_t wholepack_buffer_length = 160 * 158 * 3 * 3;
float* wholepack_buffer = new float[wholepack_buffer_length];
Tensor WholePackAncConv_Stride1_SingleThread_NoSimd(const Tensor& input, const Tensor& weights)
{
    const int64_t batch_size = input.size(0);
    const int64_t in_channels = input.size(1);
    const int64_t height = input.size(2);
    const int64_t width = input.size(3);
    const int64_t kernel_height = weights.size(1);
    const int64_t kernel_width = weights.size(2);
    const int64_t out_channels = weights.size(3);
    const int64_t width_out = width - kernel_width + 1;
    const int64_t height_out = height - kernel_height + 1;
    const int64_t eol_jump = width_out - width;
    const size_t copy_size = sizeof(float) * width_out;

    const int64_t in_length = height * width;
    const int64_t single_pack_length = height * width_out;
    const int64_t out_length = height_out * width_out;

    auto options = torch::TensorOptions().dtype(torch::kF32).requires_grad(false);
    Tensor result = torch::zeros({ batch_size, out_channels, height_out, width_out }, options).contiguous();


    if (single_pack_length * kernel_width * in_channels > wholepack_buffer_length)
    {
        _freea(wholepack_buffer);
        
        wholepack_buffer_length = single_pack_length * kernel_width * in_channels;
        wholepack_buffer = (float*)_malloca(sizeof(float) * wholepack_buffer_length);
    }

    const float* currWeight = weights.data_ptr<float>();
    for (int64_t batch = 0; batch < batch_size; ++batch)
    {
        // ===== Packing =====
        const float* inputPtr = input[batch].data_ptr<float>();
        float* bufferCurr = wholepack_buffer;
        for (int64_t in_channel = 0; in_channel < in_channels; ++in_channel)
        {
            // Seperable convolution: For one of [1,0,0], [0,1,0], [0,0,1]
            for (int64_t offset_x = 0; offset_x < kernel_width; ++offset_x)
            {
                // Pack
                Pack(inputPtr + offset_x, bufferCurr, height, width, width_out, copy_size);
                bufferCurr += single_pack_length;
            }

            inputPtr += in_length;
        }

        // ===== Multiplying ====
        float* outPtr = result[batch].data_ptr<float>();
        bufferCurr = wholepack_buffer;
        for (int64_t in_channel = 0; in_channel < in_channels; ++in_channel)
        {
            for (int64_t offset_x = 0; offset_x < kernel_width; ++offset_x)
            {
                const float* bufferForPackedCurrentLine = bufferCurr;
                for (int64_t offset_y = 0; offset_y < kernel_height; ++offset_y)
                {
                    // Multiply and add
                    // For every in channel we will run on every out channel
                    MulAddMultiple(OUT_ARG outPtr, IN_ARG bufferForPackedCurrentLine, IN_ARG currWeight, out_channels, out_length);

                    // Pointer arithmetics
                    currWeight += out_channels;
                    bufferForPackedCurrentLine += width_out;
                }

                bufferCurr += single_pack_length;
            }
        }
    }

    return result;
}







/* PointwiseConvolution_SingleThread_NoSimd
* *****************************************
Also for simulating MatMul 
*/
// Members used for PointwiseConvolution_SingleThread_NoSimd
int pointwiseconvolution_buffer_length = 512 * 512;
float* pointwiseconvolution_buffer = new float[pointwiseconvolution_buffer_length];

/*
* PointwiseConvolution_SingleThread_NoSimd
* Input shape (batch, in channels, height, width)
* Weights shape (in channels, out channels)
*/
Tensor PointwiseConvolution_SingleThread_NoSimd(const Tensor& input, const Tensor& weights)
{
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int length = height * width;
    const int out_channels = weights.size(1);
    const int copy_size = sizeof(float) * length;

    auto options = torch::TensorOptions().dtype(torch::kF32).requires_grad(false);
    Tensor result = torch::zeros({ batch_size, out_channels, height, width }, options).contiguous();

    if (length > pointwiseconvolution_buffer_length)
    {
        delete[] pointwiseconvolution_buffer;
        pointwiseconvolution_buffer_length = length;
        pointwiseconvolution_buffer = new float[pointwiseconvolution_buffer_length];
    }

    const float* weightPtr = weights.data_ptr<float>();

    for (int batch = 0; batch < batch_size; ++batch)
    {
        float* outPtr = result[batch].data_ptr<float>();
        const float* inputPtr = input[batch].data_ptr<float>();

        for (int in_channel = 0; in_channel < in_channels; ++in_channel)
        {
            memcpy(pointwiseconvolution_buffer, inputPtr, copy_size);
            inputPtr += length;

            // Multiply and add
            // For every in channel we will run on every out channel
            MulAddMultiple(outPtr, pointwiseconvolution_buffer, weightPtr, out_channels, length);
            weightPtr += out_channels;
        }
    }
    return result;
}


/*
* FusedProjPointwiseConvolution_SingleThread_NoSimd
* Input shape(batch, in channels, height, width)
* Weights shape (in channels, kernel_width, kernel_height, out channels)
*/

float** mt_partialpack_buffers = NULL;
// Array of pointers, pointing to the weight at [in_channel, kernel_width, :, :]
float* mt_result_buffer_preallocated = NULL;

void SMMConv_Multithread_Init(int batch_size, int input_height, int input_width, int kernel_width, int kernel_height, int out_channels)
{
    // Clear
    if (mt_partialpack_buffers != NULL)
    {
        _freea(mt_partialpack_buffers[0]);
        delete[] mt_partialpack_buffers;
    }
    int num_of_buffers = kernel_width;
    int output_width = input_width - kernel_width + 1;
    int output_height = input_height - kernel_height + 1;
    int packing_size = input_height * output_width;
    int buffer_length = packing_size * num_of_buffers * sizeof(float);

    // Allocate buffer
    float* buffer = (float*)_malloca(buffer_length);

    // Allocate pointers array
    mt_partialpack_buffers = new float* [kernel_width];

    // Set pointers to buffer locations
    for (size_t i = 0; i < num_of_buffers; i++)
    {
        mt_partialpack_buffers[i] = buffer + (i * packing_size);
    }

    if (mt_result_buffer_preallocated)
        _freea(mt_result_buffer_preallocated);
    mt_result_buffer_preallocated = (float*)_malloca(sizeof(float) * batch_size * out_channels * output_height * output_width);
}

void free_aligned(void* data)
{
    if (data)
    {
        if (mt_result_buffer_preallocated)
            _freea(data);
        else
            mt_result_buffer_preallocated = (float*)data;

    }
}

Tensor SMMConv_Stride1_MultiThread_NoSimd(const Tensor& input, const Tensor& weights)
{
    const int64_t batch_size = input.size(0);
    const int64_t in_channels = input.size(1);
    const int64_t height = input.size(2);
    const int64_t width = input.size(3);
    const int64_t input_matrix_size = height * width;

    const int64_t kernel_height = weights.size(1);
    const int64_t kernel_width = weights.size(2);
    const int64_t out_channels = weights.size(3);

    const int64_t width_out = width - kernel_width + 1;
    const int64_t height_out = height - kernel_height + 1;
    const int64_t out_matrix_size = height_out * width_out;

    // Variables for packing
    const int64_t copy_size = sizeof(float) * width_out;

    // Variables for shifting & multiplying
    const int64_t in_channel_num_of_weights = out_channels * kernel_width * kernel_height;
    const int num_of_threads = omp_get_max_threads();
    const int out_channels_per_thread = (float)out_channels / num_of_threads;


    auto options = torch::TensorOptions().dtype(torch::kF32).requires_grad(false);
    float* out_mem_buffer;
    if (mt_result_buffer_preallocated)
    {
        out_mem_buffer = mt_result_buffer_preallocated;
        mt_result_buffer_preallocated = NULL;
    }
    else
        out_mem_buffer = (float*)_malloca(sizeof(float) * batch_size * out_channels * out_matrix_size);
    Tensor result = torch::from_blob(out_mem_buffer, { batch_size, out_channels, height_out, width_out }, free_aligned, options);
    
    const float* weightPtr = weights.data_ptr<float>();
    for (int batch = 0; batch < batch_size; ++batch)
    {
        float* inputPtr = input[batch].data_ptr<float>();

        // Prepare array of the outputs for faster lookup. Each thread has output channels associated with it
        float* outPtr = result[batch].data_ptr<float>();
        float** thread_out_ptrs = (float**)alloca(sizeof(void*) * num_of_threads);
        for (int num_thread = 0; num_thread < num_of_threads; ++num_thread)
            thread_out_ptrs[num_thread] = outPtr + (out_channels_per_thread * num_thread * out_matrix_size);

        for (int64_t in_channel = 0; in_channel < in_channels; ++in_channel)
        {
            // Parallel Packing
            #pragma omp parallel for 
            for (int64_t offset_x = 0; offset_x < kernel_width; ++offset_x)
            {
                int num_thread = omp_get_thread_num();
                float* bufferCurr = mt_partialpack_buffers[num_thread];
                float* inputCurr = inputPtr + (in_channel * input_matrix_size) + offset_x;
                Pack(inputCurr, bufferCurr, height, width, width_out, copy_size);
            }

            #pragma omp parallel
            {
                int num_thread = omp_get_thread_num();
                float* thread_out_ptr = thread_out_ptrs[num_thread];
                const float* currWeight = weightPtr; // current thread
                for (int buffer_num = 0; buffer_num < kernel_width; ++buffer_num)
                {
                    auto bufferCurr = mt_partialpack_buffers[buffer_num];
                    
                    for (int64_t offset_y = 0; offset_y < kernel_height; ++offset_y)
                    {
                        // Multiply and add
                        // For every in channel we will run on every out channel
                        MulAddMultiple_SIMD(thread_out_ptr, bufferCurr, currWeight, out_channels_per_thread, out_matrix_size);

                        // Pointer arithmetics
                        currWeight += out_channels;
                        bufferCurr += width_out;
                    }
                }
            }

            weightPtr += in_channel_num_of_weights;
        }
    }

    return result;
}
