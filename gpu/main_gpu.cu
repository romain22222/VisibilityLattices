#include <iostream>
#include <random>
#include <string>
#include <DGtal/arithmetic/IntegerComputer.h>
#include <DGtal/base/Common.h>
#include <DGtal/geometry/meshes/CorrectedNormalCurrentComputer.h>
#include <DGtal/geometry/volumes/DigitalConvexity.h>
#include <DGtal/geometry/volumes/TangencyComputer.h>
#include <DGtal/helpers/StdDefs.h>
#include <DGtal/helpers/Shortcuts.h>
#include <DGtal/helpers/ShortcutsGeometry.h>
#include "../additionnalClasses/LinearKDTree.h"
#include "../CLI11.hpp"
#include "main_gpu.cuh"

using namespace DGtal;
using namespace Z3i;

// Typedefs
typedef std::vector <Vec3i> Vec3is;
typedef DigitalConvexity <KSpace> Convexity;
typedef typename Convexity::LatticeSet LatticeSet;

// Global variables
IntegerComputer <Integer> icgpu;

Vec3i pointVectorToVec3i(const Point &vector) {
  return {vector[0], vector[1], vector[2]};
}


struct LatticeFoundResult {
  int keyIndex{};
  IntervalList intervals;
};

__host__ __device__ int myMax(int a, int b) {
  return (a > b) ? a : b;
}

__host__ __device__ int myMin(int a, int b) {
  return (a < b) ? a : b;
}

template<typename Type>
struct Buffer {
  Type *data;
  size_t capacity;
  size_t size;

  __host__ __device__ Buffer(size_t cap) : capacity(cap), size(0) {
    cudaMalloc(&data, sizeof(Type) * capacity);
  }

  __host__ __device__ ~Buffer() {
    if (data) {
      cudaFree(data);
    }
  }

  __device__ void push_back(const Type &v) {
    data[size++] = v;
  }

  __host__ __device__ Type &operator[](size_t index) {
    return data[index];
  }
};

struct MyLatticeSet {
  Vec3i *d_keys{};
  IntervalList *d_intervals{};
  int myAxis;

  size_t numKeys = 0;

  __host__ void toGPU(LatticeSet &cpuLattice) {
    std::vector <Vec3i> keys;
    std::vector <size_t> offsets;
    std::vector <IntervalList> allIntervals;

    size_t offset = 0;

    for (auto [key, intervals]: cpuLattice.data()) {
      keys.push_back(pointVectorToVec3i(key));
      offsets.push_back(offset);
      const auto &ivs = intervals.data();
      IntervalList intervalList;
      intervalList.data = new IntervalGpu[ivs.size()];
      intervalList.capacity = ivs.size();
      intervalList.size = ivs.size();
      for (size_t i = 0; i < ivs.size(); ++i) {
        intervalList.data[i] = {ivs[i].first, ivs[i].second};
      }
      allIntervals.push_back(intervalList);
      offset += intervalList.size;
    }

    numKeys = keys.size();

    cudaMalloc(&d_keys, sizeof(Vec3i) * numKeys);
    cudaMalloc(&d_intervals, sizeof(IntervalList) * numKeys);

    cudaMemcpy(d_keys, keys.data(), sizeof(Vec3i) * numKeys, cudaMemcpyHostToDevice);
    cudaMemcpy(d_intervals, allIntervals.data(), sizeof(IntervalList) * numKeys, cudaMemcpyHostToDevice);
  }

  __host__ MyLatticeSet(LatticeSet &l)
      : myAxis(l.axis()) {
    toGPU(l);
  }

  __device__ MyLatticeSet(const Vec3i segment, int axis)
      : myAxis(axis) {
    int otherAxis1 = (axis + 1) % 3;
    int otherAxis2 = (axis + 2) % 3;
    int currentIndex = 0;
    Buffer<Vec3i> mainPointsBuf(2 * (segment[0] + segment[1] + segment[2]) + 1);
    mainPointsBuf.push_back(Vec3i(0, 0, 0));
    for (int k = 0; k < 3; k++) {
      const int n = (segment[k] >= 0) ? segment[k] : -segment[k];
      const int d = (segment[k] >= 0) ? 1 : -1;
      if (n == 0) continue;
      Vec3i kc;
      for (int i = 1; i < n; i++) {
        for (int j = 0; j < 3; j++) {
          if (j == k) kc[k] = 2 * (d * i);
          else {
            const auto v = segment[j];
            const auto q = (v * i) / n;
            const auto r = (v * i) % n; // might be negative
            kc[j] = 2 * q;
            if (r < 0) kc[j] -= 1;
            else if (r > 0) kc[j] += 1;
          }
        }
        mainPointsBuf.push_back(kc);
      }
    }
    if (segment != Vec3i()) mainPointsBuf.push_back(segment * 2);

    // Allocate memory for keys and intervals
    int allocateAmount = (2 * segment[otherAxis1] + 3) * (2 * segment[otherAxis2] + 3);
    cudaMalloc(&d_keys, sizeof(Vec3i) * allocateAmount);
    cudaMalloc(&d_intervals, sizeof(IntervalList) * allocateAmount);

    int currentMaxKey = 0;

    for (size_t i = 0; i < currentIndex; ++i) {
      for (int j = -1; j < 2; ++j) {
        for (int k = -1; k < 2; ++k) {
          Vec3i key = mainPointsBuf[i];
          key[otherAxis1] += j;
          key[otherAxis2] += k;
          auto alreadyExists = this->find(key);
          if (alreadyExists.keyIndex == -1) {
            d_keys[currentMaxKey] = key;
            cudaMalloc(&d_intervals[currentMaxKey].data, sizeof(IntervalGpu));
            d_intervals[currentMaxKey].size = 1;
            d_intervals[currentMaxKey].capacity = 1;
            d_intervals[currentMaxKey].data[0] = {key[axis] - 1, key[axis] + 1};
            currentMaxKey++;
          } else {
            // If the key already exists, merge intervals
            auto &existingIntervals = d_intervals[alreadyExists.keyIndex];
            existingIntervals.data[0].start = myMin(existingIntervals.data[existingIntervals.size - 1].start,
                                                    key[axis] - 1);
            existingIntervals.data[0].end = myMax(existingIntervals.data[existingIntervals.size - 1].end,
                                                  key[axis] + 1);
          }
        }
      }
    }

    numKeys = currentMaxKey;
  }

  __device__ LatticeFoundResult find(const Vec3i &p) const {
    // Search for the point p in the lattice set
    for (size_t i = 0; i < numKeys; ++i) {
      if (d_keys[i] == p) {
        return {i, d_intervals[i]};
      }
    }
    return {-1, d_intervals[0]};
  }
};

/**
 * Arbitrary consistent order for points
 * @param p1
 * @param p2
 * @return
 */
bool isPointLowerThan(const Vec3i &p1, const Vec3i &p2) {
  return p1[0] < p2[0] || (p1[0] == p2[0] && p1[1] < p2[1]) || (p1[0] == p2[0] && p1[1] == p2[1] && p1[2] < p2[2]);
}

inline int gcd(int a, int b) {
  while (b != 0) {
    int tmp = b;
    b = a % b;
    a = tmp;
  }
  return a;
}

inline int gcd3(int a, int b, int c) {
  return gcd(a, gcd(b, c));
}

/**
 * Get the main axis of the digital figure
 * @return
 */
int getLargeAxis(std::vector<int> &digital_dimensions) {
  auto axis = 0;
  for (auto i = 1; i < digital_dimensions.size() / 3; i++) {
    if (digital_dimensions[i] > digital_dimensions[axis]) axis = i;
  }
  return axis;
}

/**
 * Get all the unique integer vectors of maximum coordinate r
 * Only the vectors with gcd(x, y, z) = 1 are considered
 * @param r
 * @return
 */
Vec3is getAllVectorsVec3i(int radius) {
  Vec3is vectors;
  for (auto x = radius; x >= 1; x--) {
    for (auto y = radius; y >= -radius; y--) {
      for (auto z = radius; z >= -radius; z--) {
        if (icgpu.gcd(icgpu.gcd(x, y), z) != 1) continue;
        vectors.emplace_back(x, y, z);
      }
    }
  }
  for (auto y = radius; y >= 1; y--) {
    for (auto z = radius; z >= -radius; z--) {
      if (icgpu.gcd(y, z) != 1) continue;
      vectors.emplace_back(0, y, z);
    }
  }
  vectors.emplace_back(0, 0, 1);
  return vectors;
}

/**
 * Check if the interval toCheck is contained in figIntervals
 * @param toCheck
 * @param figIntervals
 * @return
 */
__device__ IntervalList checkInterval(IntervalList &buf, const IntervalGpu &toCheck, const IntervalList &figIntervals) {
//  IntervalList result(figIntervals.size);
  buf.size = 0;
  const auto toCheckSize = toCheck.end - toCheck.start;
  for (const auto &interval: figIntervals) {
    if (interval.end - interval.start >= toCheckSize) {
      buf.data[buf.size++] = {interval.start - toCheck.start, interval.end - toCheck.end};
    }
  }
  return buf;
}

__device__ IntervalList intersect(IntervalList &buf, const IntervalList &l1, const IntervalList &l2) {
//  IntervalList result(l1.capacity);
  buf.size = 0;
  int k1 = 0, k2 = 0;
  while (k1 < l1.size && k2 < l2.size) {
    const auto interval1 = l1[k1];
    const auto interval2 = l2[k2];
    const auto i = myMax(interval1.start, interval2.start);
    const auto j = myMin(interval1.end, interval2.end);
    if (i <= j) buf.data[buf.size++] = {i, j};
    if (interval1.end <= interval2.end) k1++;
    if (interval1.end >= interval2.end) k2++;
  }
  return buf;
}

/**
 * From the current interval toCheck, find the shifts of latticeVector that makes latticeVector contained in figLattices
 * @param toCheck
 * @param latticeVector
 * @param figLattices
 * @return
 */
__device__ IntervalList matchVector(IntervalList &toCheck,
                                    const IntervalList &vectorIntervals,
                                    const IntervalList &figIntervals,
                                    const int axisDim) {
  IntervalList buf(2 * axisDim + 1);
  for (const auto &vInterval: vectorIntervals) {
    toCheck = intersect(buf, toCheck, checkInterval(buf, vInterval, figIntervals));
    if (toCheck.empty()) break;
  }
  return toCheck;
}

__global__ void computeVisibilityKernel(
    int axis, int *digital_dimensions, int *axises_idx,
    MyLatticeSet figLattices, GpuVisibility visibility,
    Vec3i *segmentList, int segmentSize
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= segmentSize) return;

  Vec3i segment = segmentList[idx];
  IntervalList eligibles(2 * digital_dimensions[axis] + 1);
  MyLatticeSet latticeVector(segment, axis);
  int minTx = digital_dimensions[axises_idx[1] + 3] - myMin(0, segment[axises_idx[1]]);
  int maxTx = digital_dimensions[axises_idx[1] + 6] + 1 - myMax(0, segment[axises_idx[1]]);
  int minTy = digital_dimensions[axises_idx[2] + 3] - myMin(0, segment[axises_idx[2]]);
  int maxTy = digital_dimensions[axises_idx[2] + 6] + 1 - myMax(0, segment[axises_idx[2]]);
  for (auto tx = minTx; tx < maxTx; tx++) {
    for (auto ty = minTy; ty < maxTy; ty++) {
      eligibles.size = 1;
      eligibles.data[0] = {2 * digital_dimensions[axises_idx[1] + 3] - 1,
                           2 * digital_dimensions[axises_idx[1] + 6] + 1};
      const Vec3i pInterest(axis == 0 ? 0 : 2 * tx, axis == 1 ? 0 : 2 * (axis == 0 ? tx : ty),
                            axis == 2 ? 0 : 2 * ty);
      for (auto i = 0; i < latticeVector.numKeys; i++) {
        const auto key = latticeVector.d_keys[i];
        const auto value = latticeVector.d_intervals[i];
        const auto res = figLattices.find(pInterest + key);
        if (res.keyIndex >= 0)
          eligibles = matchVector(eligibles, value, res.intervals, digital_dimensions[axis]);
        else
          eligibles = res.intervals;
        if (eligibles.empty()) break;
      }
      if (!eligibles.empty()) {
        visibility.set(pInterest / 2, eligibles, idx);
      }
    }
  }
}

HostVisibility computeVisibilityGpu(int radius, std::vector<int> &digital_dimensions,
                                             std::vector <Point> &pointels) {
  std::cout << "Computing visibility GPU" << std::endl;
  auto axis = getLargeAxis(digital_dimensions);
  auto tmpL = LatticeSetByIntervals<Space>(pointels.cbegin(), pointels.cend(), axis).starOfPoints();
  std::cout << "Lattice set computed" << std::endl;
  MyLatticeSet figLattices(tmpL);

//  const auto axises_idx = std::vector < int > {axis, axis == 0 ? 1 : 0, axis == 2 ? 1 : 2};
  int *axises_idx = new int[3]{axis, axis == 0 ? 1 : 0, axis == 2 ? 1 : 2};
  auto segmentList = getAllVectorsVec3i(radius);

  std::cout << "Segment list computed with " << segmentList.size() << " segments" << std::endl;

  auto *pointelsData = new Vec3i[pointels.size()];
  for (size_t i = 0; i < pointels.size(); ++i) {
    pointelsData[i] = pointVectorToVec3i(pointels[i]);
  }

  std::cout << "Pointels digitized" << std::endl;

  GpuVisibility tmpVisibility(axis, segmentList.data(), segmentList.size(), pointelsData, pointels.size());
  delete[] pointelsData;

  std::cout << "Visibility initialized" << std::endl;

  std::cout << "Launching kernel with " << (segmentList.size() + 255) / 256 << " blocks and 256 threads per block"
            << std::endl;
  computeVisibilityKernel<<<(segmentList.size() + 255) / 256, 256>>>(
      axis,
      digital_dimensions.data(),
      axises_idx,
      figLattices, tmpVisibility, segmentList.data(), segmentList.size()
  );
  cudaDeviceSynchronize();

  std::cout << "Kernel finished" << std::endl;

  HostVisibility visibility(tmpVisibility);

  delete[] axises_idx;
  std::cout << "Visibility computed" << std::endl;
  return visibility;
}