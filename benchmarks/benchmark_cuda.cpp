#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>
#include "dtw.h"

using namespace std;

// __global__ void calculateDTW(const double* reference,
//                              const double* test,
//                              double* result,
//                              int size) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < size) {
//         // Calculate DTW here for each test sequence with the reference
//         sequence
//         // Update result array with DTW distances
//     }
// }

vector<double> getrandomwalk(uint size) {
    vector<double> data(size);
    data[0] = 0.0;
    for (uint k = 1; k < size; ++k)
        data[k] = (1.0 * rand() / (RAND_MAX)) - 0.5 + data[k - 1];
    return data;
}

vector<double> getcin() {
    float val;
    cin >> val;
    vector<double> v;
    while (cin) {
        v.push_back(val);
        cin >> val;
    }
    cout << "# Read " << v.size() << " data points. " << endl;
    return v;
}

template <class NN>
vector<uint> RunMe(vector<vector<double>>& collection,
                   vector<vector<double>>& testcollection) {
    vector<uint> bestmatches;
    clock_t start, finish;
    start = clock();
    // Prepare GPU memory
    double *d_collection, *d_test;
    cudaError_t err;

    err = cudaMalloc(&d_collection,
                     sizeof(double) * collection.size() * collection[0].size());
    if (err != cudaSuccess) {
        cout << "cudaMalloc for d_collection failed: "
             << cudaGetErrorString(err) << endl;
        return bestmatches;
    }

    err = cudaMemcpy(d_collection, collection.data(),
                     sizeof(double) * collection.size() * collection[0].size(),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cout << "cudaMemcpy for d_collection failed: "
             << cudaGetErrorString(err) << endl;
        return bestmatches;
    }

    err = cudaMalloc(&d_test, sizeof(double) * testcollection.size() *
                                  testcollection[0].size());
    if (err != cudaSuccess) {
        cout << "cudaMalloc for d_test failed: " << cudaGetErrorString(err)
             << endl;
        return bestmatches;
    }

    err = cudaMemcpy(
        d_test, testcollection.data(),
        sizeof(double) * testcollection.size() * testcollection[0].size(),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cout << "cudaMemcpy for d_test failed: " << cudaGetErrorString(err)
             << endl;
        return bestmatches;
    }

    // cout << "cudaMalloc success" << endl;

    // 找到 testcollection 中每个元素在 collection 中的最佳匹配
    for (uint k = 0; k < testcollection.size(); ++k) {
        NN n(testcollection[k],
             testcollection[k].size() / 10);  // window set at 10%, arbitrarily
        double current = n.getLowestCost();
        uint bestmatch = 0;
        for (uint z = 0; z < collection.size(); ++z) {
            // cout << "build d_collection begin" << endl;
            // Allocate host memory
            double* h_collection = new double[collection[0].size()];

            // Copy data from device to host
            cudaError_t err = cudaMemcpy(
                h_collection, d_collection + z * collection[0].size(),
                sizeof(double) * collection[0].size(), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                cout << "cudaMemcpy for h_collection failed: "
                     << cudaGetErrorString(err) << endl;
                delete[] h_collection;
                return bestmatches;
            }

            // Create vector from host memory
            std::vector<double> d_collection_vec(
                h_collection, h_collection + collection[0].size());

            // Free host memory
            delete[] h_collection;
            // cout << "build d_collection end" << endl;
            double newc = n.test(d_collection_vec);
            if (newc < current) {  // best candidate so far
                current = newc;
                bestmatch = z;
            }
        }
        bestmatches.push_back(bestmatch);
    }
    cout << "cuda Free begin..." << endl;
    cudaFree(d_collection);
    cudaFree(d_test);
    finish = clock();
    cout << "GPU time = "
         << static_cast<double>(finish - start) / CLOCKS_PER_SEC << endl;

    return bestmatches;
}

void runbenchmark() {
    uint N = 128;
    vector<vector<double>> collection;
    vector<vector<double>> testcollection;

    for (uint i = 0; i < 512; ++i) {
        collection.push_back(getrandomwalk(N));
        testcollection.push_back(getrandomwalk(N));
    }
    cout << "LB Keogh" << endl;
    RunMe<LB_Keogh>(collection, testcollection);
    cout << "LB Keogh (early)" << endl;
    RunMe<LB_KeoghEarly>(collection, testcollection);
    cout << "LB Improved" << endl;
    RunMe<LB_Improved>(collection, testcollection);
    cout << "LB Improved (early)" << endl;
    RunMe<LB_ImprovedEarly>(collection, testcollection);
}

int main() {
    srand(time(NULL));
    runbenchmark();
    return 0;
}
