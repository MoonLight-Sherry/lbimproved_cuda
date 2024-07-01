#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "dtw_cuda.h"

using namespace std;

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

void run_dtw_cuda(vector<vector<double>>& collection,
                  vector<vector<double>>& testcollection) {
    vector<uint> bestmatches;
    clock_t start, finish;
    start = clock();

    for (uint k = 0; k < testcollection.size(); ++k) {
        double current = INFINITY;
        uint bestmatch = 0;
        for (uint z = 0; z < collection.size(); ++z) {
            double result;
            LB_Improved(testcollection[k], collection[z],
                        testcollection[k].size() / 10, &result);
            if (result < current) {
                current = result;
                bestmatch = z;
            }
        }
        bestmatches.push_back(bestmatch);
    }
    finish = clock();
    cout << "CUDA DTW time = "
         << static_cast<double>(finish - start) / CLOCKS_PER_SEC << endl;
}

void runbenchmark() {
    uint N = 128;
    vector<vector<double>> collection;
    vector<vector<double>> testcollection;

    for (uint i = 0; i < 512; ++i) {
        collection.push_back(getrandomwalk(N));
        testcollection.push_back(getrandomwalk(N));
    }

    cout << "CUDA DTW" << endl;
    run_dtw_cuda(collection, testcollection);
}

int main() {
    runbenchmark();
    return 0;
}
