#ifndef DTW_CUDA_H
#define DTW_CUDA_H

#include <vector>

void LB_Improved(const std::vector<double>& V,
                 const std::vector<double>& candidate,
                 int constraint,
                 double* error);

#endif  // DTW_CUDA_H
