#include <iostream>
#include <string>
#include "../CLI11.hpp"
#include "main_gpu.cuh"

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

  __host__ __device__ explicit Buffer(size_t cap) : capacity(cap), size(0) {
    cudaMalloc(&data, sizeof(Type) * capacity);
  }

  __host__ __device__ ~Buffer() {
    if (data) {
      cudaFree(data);
    }
  }

  __device__ void push_back(const Type &v) {
    if (size >= capacity) {
      printf("Buffer overflow in push_back\n");
    }
    data[size++] = v;
  }

  __host__ __device__ Type &operator[](size_t index) {
    if (index >= size) {
      printf("Buffer index out of range in operator[]\n");
    }
    return data[index];
  }
};

__host__ void MyLatticeSet::toGPU(std::vector<Vec3i> &keys, std::vector<IntervalList> &allIntervals) {
  numKeys = keys.size();

  cudaMalloc(&d_keys, sizeof(Vec3i) * numKeys);
  cudaMalloc(&d_intervals, sizeof(IntervalList) * numKeys);

  cudaMemcpy(d_keys, keys.data(), sizeof(Vec3i) * numKeys, cudaMemcpyHostToDevice);
  cudaMemcpy(d_intervals, allIntervals.data(), sizeof(IntervalList) * numKeys, cudaMemcpyHostToDevice);

}

__host__  MyLatticeSet::MyLatticeSet(int axis, std::vector<Vec3i> &keys, std::vector<IntervalList> &allIntervals)
    : myAxis(axis) {
  toGPU(keys, allIntervals);
}

__device__ MyLatticeSet::MyLatticeSet(const Vec3i segment, int axis)
    : myAxis(axis) {
  int otherAxis1 = (axis + 1) % 3;
  int otherAxis2 = (axis + 2) % 3;
  Buffer<Vec3i> mainPointsBuf(abs(segment[0]) + abs(segment[1]) + abs(segment[2]) + 1);
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

  for (size_t i = 0; i < mainPointsBuf.size; ++i) {
    for (int j = -1; j < 2; ++j) {
      for (int k = -1; k < 2; ++k) {
        Vec3i key = mainPointsBuf[i];
        key[otherAxis1] += j;
        key[otherAxis2] += k;
        auto alreadyExists = this->findWithoutAxis(key, axis);
        if (alreadyExists.keyIndex == -1) {
          if (numKeys >= allocateAmount) {
            printf("Exceeded allocated amount for lattice keys\n");
          }
          d_keys[numKeys] = key;
          cudaMalloc(&d_intervals[numKeys].data, sizeof(IntervalGpu));
          d_intervals[numKeys].size = 1;
          d_intervals[numKeys].capacity = 1;
          d_intervals[numKeys].data[0] = {key[axis] - 1, key[axis] + 1};
          numKeys++;
        } else {
          // If the key already exists, merge intervals
          auto &existingIntervals = d_intervals[alreadyExists.keyIndex];
          existingIntervals.data[0].start = myMin(existingIntervals.data[0].start,
                                                  key[axis] - 1);
          existingIntervals.data[0].end = myMax(existingIntervals.data[0].end,
                                                key[axis] + 1);
        }
      }
    }
  }
}

__device__ MyLatticeSet::~MyLatticeSet() {
  if (d_keys) {
    cudaFree(d_keys);
  }
  if (d_intervals) {
    for (size_t i = 0; i < numKeys; ++i) {
      if (d_intervals[i].data) {
        cudaFree(d_intervals[i].data);
      }
    }
    cudaFree(d_intervals);
  }
}

__device__ LatticeFoundResult MyLatticeSet::find(const Vec3i &p) const {
  // Search for the point p in the lattice set
  for (size_t i = 0; i < numKeys; ++i) {
    if (d_keys[i] == p) {
      return {static_cast<int>(i), d_intervals[i]};
    }
  }
  return {-1, d_intervals[0]};
}

__device__ LatticeFoundResult MyLatticeSet::findWithoutAxis(const Vec3i &p, int axis) const {
  // Search for the point p in the lattice set
  int otherAxis1 = (axis + 1) % 3;
  int otherAxis2 = (axis + 2) % 3;
  for (size_t i = 0; i < numKeys; ++i) {
    if (d_keys[i][otherAxis1] == p[otherAxis1] && d_keys[i][otherAxis2] == p[otherAxis2]) {
      return {static_cast<int>(i), d_intervals[i]};
    }
  }
  return {-1, d_intervals[0]};
}

/**
 * Arbitrary consistent order for points
 * @param p1
 * @param p2
 * @return
 */
bool isPointLowerThan(const Vec3i &p1, const Vec3i &p2) {
  return p1.x < p2.x || (p1.x == p2.x && (p1.y < p2.y || (p1.y == p2.y && p1.z < p2.z)));
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
__device__ IntervalList matchVector(
    IntervalList &buf,
    IntervalList &toCheck,
    const IntervalList &vectorIntervals,
    const IntervalList &figIntervals) {
  for (const auto &vInterval: vectorIntervals) {
    toCheck = intersect(buf, toCheck, checkInterval(buf, vInterval, figIntervals));
    if (toCheck.empty()) break;
  }
  return toCheck;
}

__global__ void computeVisibilityKernel(
    int axis, const int *digital_dimensions, const int *axises_idx,
    const MyLatticeSet& figLattices, GpuVisibility visibility,
    Vec3i *segmentList, int segmentSize
) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= segmentSize) return;

  Vec3i segment = segmentList[idx];
  IntervalList buf(2 * digital_dimensions[axis] + 1);
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
        const auto res = figLattices.findWithoutAxis(pInterest + key, axis);
        if (res.keyIndex < 0) {
          eligibles.size = 0;
          break;
        }
        eligibles = matchVector(buf, eligibles, value, res.intervals);
        if (eligibles.empty()) break;
      }
      if (!eligibles.empty()) {
        visibility.set(pInterest / 2, eligibles, idx);
      }
    }
  }
}

HostVisibility computeVisibility(
    int chunkAmount, int chunkSize,
    int axis, int *digital_dimensions, int *axises_idx,
    const MyLatticeSet& figLattices,
    Vec3i *segmentList, int segmentSize,
    Vec3i *pointels, int pointelsSize
) {

  // Malloc every list on GPU
  int *d_digital_dimensions;
  int *d_axises_idx;
  Vec3i *d_segmentList;
  Vec3i *d_pointels;
  auto err = cudaMalloc(&d_digital_dimensions, sizeof(int) * 9);
  if (err != cudaSuccess) {
    std::cerr << "Error allocating memory for digital_dimensions: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc(&d_axises_idx, sizeof(int) * 3);
  if (err != cudaSuccess) {
    std::cerr << "Error allocating memory for axises_idx: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc(&d_segmentList, sizeof(Vec3i) * segmentSize);
  if (err != cudaSuccess) {
    std::cerr << "Error allocating memory for segmentList: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc(&d_pointels, sizeof(Vec3i) * pointelsSize);
  if (err != cudaSuccess) {
    std::cerr << "Error allocating memory for pointels: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(d_digital_dimensions, digital_dimensions, sizeof(int) * 9, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Error copying digital_dimensions to device: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(d_axises_idx, axises_idx, sizeof(int) * 3, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Error copying axises_idx to device: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(d_segmentList, segmentList, sizeof(Vec3i) * segmentSize, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Error copying segmentList to device: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(d_pointels, pointels, sizeof(Vec3i) * pointelsSize, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Error copying pointels to device: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }


  GpuVisibility tmpVisibility(axis, d_segmentList, segmentSize, d_pointels, pointelsSize);

  std::cout << "Visibility initialized" << std::endl;
  std::cout << "Launching kernel with " << chunkAmount << " blocks and " << chunkSize << " threads per block"
            << std::endl;

  computeVisibilityKernel<<<chunkAmount, chunkSize>>>(
      axis, d_digital_dimensions, d_axises_idx,
      figLattices, tmpVisibility,
      d_segmentList, segmentSize
  );
  cudaDeviceSynchronize();

  std::cout << "Kernel finished" << std::endl;

  HostVisibility visibility(tmpVisibility);

  cudaFree(d_digital_dimensions);
  cudaFree(d_axises_idx);
  cudaFree(d_segmentList);
  cudaFree(d_pointels);

  return visibility;
}
