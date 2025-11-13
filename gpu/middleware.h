//
// Created by romro on 25/08/2025.
//

#ifndef VISIBILITYLATTICES_MIDDLEWARE_H
#define VISIBILITYLATTICES_MIDDLEWARE_H

#include <DGtal/helpers/StdDefs.h>
#include "testgpucpu.h"
#include "main_gpu.cuh"

HostVisibility computeVisibilityGpu(int radius, std::vector<int> &digital_dimensions,
                                    std::vector<DGtal::Z3i::Point> &pointels);

HostVisibilityCPU computeVisibilityGpuCPU(int radius, std::vector<int> &digital_dimensions,
                                          std::vector<DGtal::Z3i::Point> &pointels);

#endif //VISIBILITYLATTICES_MIDDLEWARE_H
