#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <vector>

#define INF 1e20

typedef double floattype;
typedef unsigned int uint;

// CUDA doesn't support atomicAdd for double by default
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void computeEnvelope(const floattype* array,
                                uint length,
                                uint constraint,
                                floattype* maxvalues,
                                floattype* minvalues) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= length)
        return;

    __shared__ floattype s_array[1024];
    s_array[threadIdx.x] = array[idx];
    __syncthreads();

    if (idx < length) {
        int start = max(0, (int)idx - (int)constraint);
        int end = min((int)length, (int)idx + (int)constraint + 1);

        floattype maxval = -INF;
        floattype minval = INF;

        for (int i = start; i < end; ++i) {
            if (array[i] > maxval)
                maxval = array[i];
            if (array[i] < minval)
                minval = array[i];
        }

        maxvalues[idx] = maxval;
        minvalues[idx] = minval;
    }
}

__global__ void calculateLB(const floattype* V,
                            const floattype* candidate,
                            const floattype* U,
                            const floattype* L,
                            const floattype* U2,
                            const floattype* L2,
                            floattype* error,
                            uint length) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length)
        return;

    floattype e = 0.0;
    if (candidate[idx] > U[idx])
        e = candidate[idx] - U[idx];
    else if (candidate[idx] < L[idx])
        e = L[idx] - candidate[idx];

    if (V[idx] > U2[idx])
        e += V[idx] - U2[idx];
    else if (V[idx] < L2[idx])
        e += L2[idx] - V[idx];

    atomicAddDouble(error, e);
}

__global__ void fastDynamic(const floattype* v,
                            const floattype* w,
                            floattype* gamma,
                            int mN,
                            int constraint) {
    int i = blockIdx.x;
    int j = threadIdx.x;

    if (i >= mN || j >= mN)
        return;

    if (abs(i - j) > constraint) {
        gamma[i * mN + j] = INF;
        return;
    }

    floattype Best = INF;
    if (i > 0)
        Best = gamma[(i - 1) * mN + j];
    if (j > 0)
        Best = min(Best, gamma[i * mN + (j - 1)]);
    if (i > 0 && j > 0)
        Best = min(Best, gamma[(i - 1) * mN + (j - 1)]);

    if (i == 0 && j == 0)
        gamma[i * mN + j] = fabs(v[i] - w[j]);
    else
        gamma[i * mN + j] = Best + fabs(v[i] - w[j]);
}

void LB_Improved(const std::vector<floattype>& V,
                 const std::vector<floattype>& candidate,
                 int constraint,
                 floattype* error) {
    uint length = V.size();
    floattype *d_V, *d_candidate, *d_U, *d_L, *d_U2, *d_L2, *d_error;

    cudaMalloc((void**)&d_V, length * sizeof(floattype));
    cudaMalloc((void**)&d_candidate, length * sizeof(floattype));
    cudaMalloc((void**)&d_U, length * sizeof(floattype));
    cudaMalloc((void**)&d_L, length * sizeof(floattype));
    cudaMalloc((void**)&d_U2, length * sizeof(floattype));
    cudaMalloc((void**)&d_L2, length * sizeof(floattype));
    cudaMalloc((void**)&d_error, sizeof(floattype));

    cudaMemcpy(d_V, V.data(), length * sizeof(floattype),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_candidate, candidate.data(), length * sizeof(floattype),
               cudaMemcpyHostToDevice);
    cudaMemset(d_error, 0, sizeof(floattype));

    uint blockSize = 1024;
    uint numBlocks = (length + blockSize - 1) / blockSize;

    computeEnvelope<<<numBlocks, blockSize>>>(d_V, length, constraint, d_U,
                                              d_L);
    computeEnvelope<<<numBlocks, blockSize>>>(d_candidate, length, constraint,
                                              d_U2, d_L2);

    calculateLB<<<numBlocks, blockSize>>>(d_V, d_candidate, d_U, d_L, d_U2,
                                          d_L2, d_error, length);

    cudaMemcpy(error, d_error, sizeof(floattype), cudaMemcpyDeviceToHost);

    cudaFree(d_V);
    cudaFree(d_candidate);
    cudaFree(d_U);
    cudaFree(d_L);
    cudaFree(d_U2);
    cudaFree(d_L2);
    cudaFree(d_error);
}

// int main() {
//     std::vector<floattype> V = {1.0, 2.0, 3.0, 4.0, 5.0};
//     std::vector<floattype> candidate = {1.0, 2.0, 3.5, 4.5, 5.0};
//     int constraint = 1;
//     floattype error = 0.0;

//     LB_Improved(V, candidate, constraint, &error);

//     std::cout << "Lower Bound Error: " << error << std::endl;

//     return 0;
// }
