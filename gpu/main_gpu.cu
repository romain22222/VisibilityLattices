#include <iostream>
#include <vector>
#include <DGtal/base/Trace.h>
#include "main_gpu.cuh"


// helper macro for brevity (or use proper error handling)
#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE); \
  } \
} while(0)

#define CUDA_CHECK_DEVICE(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    return; \
  } \
} while(0)

#define USETIMERS
enum {
  tid_this = 0,
  tid_that,
  tid_count
};
__device__ float cuda_timers[tid_count];
#ifdef USETIMERS
#define TIMER_TIC clock_t tic; if ( threadIdx.x == 0 ) tic = clock();
#define TIMER_TOC(tid) clock_t toc = clock(); if ( threadIdx.x == 0 ) atomicAdd( &cuda_timers[tid] , ( toc > tic ) ? (toc - tic) : ( toc + (0xffffffff - tic) ) );
#else
#define TIMER_TIC
#define TIMER_TOC(tid)
#endif

DGtal::TraceWriterTerm traceWriterTerm(std::cerr);
DGtal::Trace trace(traceWriterTerm);

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
    CUDA_CHECK_DEVICE(cudaMalloc(&data, sizeof(Type) * capacity));
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

__host__  MyLatticeSet::MyLatticeSet(int axis, std::vector<Vec3i> &keys, std::vector<IntervalList *> &allIntervals)
    : myAxis(axis), myOtherAxis1((axis + 1) % 3), myOtherAxis2((axis + 2) % 3), numKeys(keys.size()) {
  d_keys = new Vec3i[numKeys];
  d_intervals = new IntervalList *[numKeys];
  for (auto i = 0; i < keys.size(); i++) {
    d_keys[i] = keys[i];
    d_intervals[i] = allIntervals[i];
  }
}

__device__ MyLatticeSet::MyLatticeSet(const Vec3i segment, int axis)
    : myAxis(axis), myOtherAxis1((axis + 1) % 3), myOtherAxis2((axis + 2) % 3) {
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
  int allocateAmount = (2 * abs(segment[myOtherAxis1]) + 3) * (2 * abs(segment[myOtherAxis2]) + 3);
  CUDA_CHECK_DEVICE(cudaMalloc(&d_keys, sizeof(Vec3i) * allocateAmount));
  CUDA_CHECK_DEVICE(cudaMalloc(&d_intervals, sizeof(IntervalList *) * allocateAmount));

  for (size_t i = 0; i < mainPointsBuf.size; ++i) {
    for (int j = -1; j < 2; ++j) {
      for (int k = -1; k < 2; ++k) {
        Vec3i key = mainPointsBuf[i];
        key[myOtherAxis1] += j;
        key[myOtherAxis2] += k;
        auto alreadyExists = this->findWithoutAxis(key);
        if (alreadyExists.keyIndex == -1) {
          if (numKeys >= allocateAmount) {
            printf("Exceeded allocated amount for lattice keys\n");
            printf("segment: (%d, %d, %d)\n", segment.x, segment.y, segment.z);
          }
          d_keys[numKeys] = key;
          CUDA_CHECK_DEVICE(cudaMalloc(&d_intervals[numKeys], sizeof(IntervalList)));
          CUDA_CHECK_DEVICE(cudaMalloc(&d_intervals[numKeys]->data, sizeof(IntervalGpu)));
          d_intervals[numKeys]->size = 1;
          d_intervals[numKeys]->capacity = 1;
          d_intervals[numKeys]->data[0] = {key[axis] - 1, key[axis] + 1};
          numKeys++;
        } else {
          // If the key already exists, merge intervals
          alreadyExists.intervals->data[0].start = myMin(alreadyExists.intervals->data[0].start,
                                                         key[axis] - 1);
          alreadyExists.intervals->data[0].end = myMax(alreadyExists.intervals->data[0].end,
                                                       key[axis] + 1);
        }
      }
    }
  }
}

/*__device__ MyLatticeSet::~MyLatticeSet() {
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
}*/

__device__ LatticeFoundResult MyLatticeSet::find(const Vec3i &p) const {
  // Search for the point p in the lattice set
  for (size_t i = 0; i < numKeys; ++i) {
    if (d_keys[i] == p) {
      return {static_cast<int>(i), d_intervals[i]};
    }
  }
  return {-1, nullptr};
}

__device__ LatticeFoundResult MyLatticeSet::findWithoutAxis(const Vec3i &p) const {
  // Search for the point p in the lattice set
  for (size_t i = 0; i < numKeys; ++i) {
    if (d_keys[i][myOtherAxis1] == p[myOtherAxis1] && d_keys[i][myOtherAxis2] == p[myOtherAxis2]) {
      return {static_cast<int>(i), d_intervals[i]};
    }
  }
  return {-1, nullptr};
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
__device__ IntervalList checkInterval(IntervalList &buf, IntervalGpu *toCheck, const IntervalList *figIntervals) {
//  IntervalList result(figIntervals.size);
  buf.size = 0;
  int err = 0;
  const auto toCheckSize = toCheck->end - toCheck->start;
  for (auto i = 0; i < figIntervals->size; i++) {
    if (figIntervals->data[i].end - figIntervals->data[i].start >= toCheckSize) {
      err = buf.push_back({figIntervals->data[i].start - toCheck->start, figIntervals->data[i].end - toCheck->end});
      if (err) {
        printf("Error pushing back in checkInterval\n");
      }
    }
  }
  return buf;
}

__device__ void intersect(IntervalList &buf, IntervalList &l1, const IntervalList &l2) {
  buf.size = 0;
  int err = 0;
  int k1 = 0, k2 = 0;
  while (k1 < l1.size && k2 < l2.size) {
    const auto interval1 = l1[k1];
    const auto interval2 = l2[k2];
    const auto i = myMax(interval1.start, interval2.start);
    const auto j = myMin(interval1.end, interval2.end);
    if (i <= j) {
      err = buf.push_back({i, j});
      if (err) printf("Error pushing back in intersect\n");
    }
    if (interval1.end <= interval2.end) k1++;
    if (interval1.end >= interval2.end) k2++;
  }
  l1.size = 0;
  for (int i = 0; i < buf.size; i++) {
    l1.push_back(buf[i]);
  }
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
    IntervalList &buf2,
    IntervalList &toCheck,
    const IntervalList *vectorIntervals,
    const IntervalList *figIntervals) {
  for (auto i = 0; i < vectorIntervals->size; i++) {
    intersect(buf2, toCheck, checkInterval(buf, &vectorIntervals->data[i], figIntervals));
    if (toCheck.empty()) break;
  }
  return toCheck;
}

__global__ void computeVisibilityKernel(
    int axis, const int *digital_dimensions, const int *axises_idx,
    MyLatticeSet *figLattices, GpuVisibility visibility,
    Vec3i *segmentList, int segmentSize
) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= segmentSize) return;

  Vec3i segment = segmentList[idx];
  IntervalList buf(2 * digital_dimensions[axis] + 1);
  IntervalList buf2(2 * digital_dimensions[axis] + 1);
  IntervalList eligibles(2 * digital_dimensions[axis] + 1);

  MyLatticeSet latticeVector(segment, axis);
  int minTx = digital_dimensions[axises_idx[1] + 3] - myMin(0, segment[axises_idx[1]]);
  int maxTx = digital_dimensions[axises_idx[1] + 6] + 1 - myMax(0, segment[axises_idx[1]]);
  int minTy = digital_dimensions[axises_idx[2] + 3] - myMin(0, segment[axises_idx[2]]);
  int maxTy = digital_dimensions[axises_idx[2] + 6] + 1 - myMax(0, segment[axises_idx[2]]);
  TIMER_TIC
  for (auto tx = minTx; tx < maxTx; tx++) {
    for (auto ty = minTy; ty < maxTy; ty++) {
      eligibles.size = 1;
      eligibles.data[0] = {2 * digital_dimensions[axis + 3] - 1,
                           2 * digital_dimensions[axis + 6] + 1};
      const Vec3i pInterest(axis == 0 ? 0 : 2 * tx, 2 * (axis == 0 ? tx : ty),
                            axis == 2 ? 0 : 2 * ty);
      for (auto i = 0; i < latticeVector.numKeys; i++) {
        const auto key = latticeVector.d_keys[i];
        const auto value = latticeVector.d_intervals[i];
        const auto res = figLattices->findWithoutAxis(pInterest + key);
        if (res.keyIndex < 0) {
          eligibles.size = 0;
          break;
        }
        if (idx == 0) {
          TIMER_TOC(tid_this)
        }
        eligibles = matchVector(buf, buf2, eligibles, value, res.intervals);
//        eligibles.size = 0; // test no visibility
        if (idx == 0) {
          TIMER_TOC(tid_that)
          printf("Thread %d key %d/%d, time so far: %f\n", idx, i, static_cast<int>(latticeVector.numKeys),
                 cuda_timers[tid_that] / 1000000.0f - cuda_timers[tid_that-1] / 1000000.0f );
        }
        if (eligibles.empty()) break;
      }
      if (!eligibles.empty())
        visibility.set(pInterest / 2, eligibles, idx);
    }
  }
  CUDA_CHECK_DEVICE(cudaFree(buf.data));
  CUDA_CHECK_DEVICE(cudaFree(buf2.data));
  CUDA_CHECK_DEVICE(cudaFree(eligibles.data));
  CUDA_CHECK_DEVICE(cudaFree(latticeVector.d_keys));
  for (size_t i = 0; i < latticeVector.numKeys; ++i) {
    CUDA_CHECK_DEVICE(cudaFree(latticeVector.d_intervals[i]->data));
  }
  CUDA_CHECK_DEVICE(cudaFree(latticeVector.d_intervals));
}

HostVisibility computeVisibility(
    int chunkAmount, int chunkSize,
    int axis, const int *digital_dimensions, const int *axises_idx,
    const MyLatticeSet &figLattices,
    Vec3i *segmentList, int segmentSize,
    Vec3i *pointels, int pointelsSize
) {

  trace.beginBlock("GPU memory allocation and copy");
  // Malloc every list on GPU
  int *d_digital_dimensions;
  int *d_axises_idx;
  Vec3i *d_segmentList;
  Vec3i *d_pointels;
  CUDA_CHECK(cudaMalloc(&d_digital_dimensions, sizeof(int) * 9));
  CUDA_CHECK(cudaMalloc(&d_axises_idx, sizeof(int) * 3));
  CUDA_CHECK(cudaMalloc(&d_segmentList, sizeof(Vec3i) * segmentSize));
  CUDA_CHECK(cudaMalloc(&d_pointels, sizeof(Vec3i) * pointelsSize));
  CUDA_CHECK(cudaMemcpy(d_digital_dimensions, digital_dimensions, sizeof(int) * 9, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_axises_idx, axises_idx, sizeof(int) * 3, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_segmentList, segmentList, sizeof(Vec3i) * segmentSize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pointels, pointels, sizeof(Vec3i) * pointelsSize, cudaMemcpyHostToDevice));
  trace.endBlock();

  trace.beginBlock("FigLattices copy to GPU");

// 1) Copy keys
  Vec3i *d_keys = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, sizeof(Vec3i) * figLattices.numKeys));
  CUDA_CHECK(cudaMemcpy(d_keys,       // dst (device)
                        figLattices.d_keys, // src (host) -- ensure this is actually host memory
                        sizeof(Vec3i) * figLattices.numKeys,
                        cudaMemcpyHostToDevice));

// 2) Build IntervalList array: for each key allocate device IntervalGpu[] and device IntervalList, store device IntervalList* in a host array
  std::vector<IntervalList> hostIntervals(figLattices.numKeys); // local copies containing sizes + device data pointer
  std::vector<IntervalList *> hostIntervalPtrs(figLattices.numKeys, nullptr); // host array of device pointers

  for (size_t i = 0; i < figLattices.numKeys; ++i) {
    // copy capacity/size into hostIntervals[i]
    hostIntervals[i].capacity = figLattices.d_intervals[i]->capacity; // ensure figLattices.d_intervals[i] is host pointer
    hostIntervals[i].size = figLattices.d_intervals[i]->size;

    // allocate device buffer for IntervalGpu data
    IntervalGpu *d_data = nullptr;
    if (hostIntervals[i].size > 0) {
      CUDA_CHECK(cudaMalloc(&d_data, sizeof(IntervalGpu) * hostIntervals[i].size));
      CUDA_CHECK(cudaMemcpy(d_data,
                            figLattices.d_intervals[i]->data, // src host data pointer
                            sizeof(IntervalGpu) * hostIntervals[i].size,
                            cudaMemcpyHostToDevice));
    } else {
      d_data = nullptr;
    }

    // set host copy's data pointer to point to device memory
    hostIntervals[i].data = d_data;

    // allocate a device IntervalList and copy the hostIntervals[i] structure to device
    IntervalList *d_interval = nullptr;
    CUDA_CHECK(cudaMalloc(&d_interval, sizeof(IntervalList)));
    CUDA_CHECK(cudaMemcpy(d_interval,
                          &hostIntervals[i], // NOTE the & here
                          sizeof(IntervalList),
                          cudaMemcpyHostToDevice));

    // store the device pointer in the host pointer-array so we can later copy the array to device
    hostIntervalPtrs[i] = d_interval;
  }

// Allocate device array of IntervalList* and copy hostIntervalPtrs to it in one go
  IntervalList **d_intervals = nullptr;
  CUDA_CHECK(cudaMalloc(&d_intervals, sizeof(IntervalList *) * figLattices.numKeys));
  CUDA_CHECK(cudaMemcpy(d_intervals,
                        hostIntervalPtrs.data(), // src host array of device pointers
                        sizeof(IntervalList *) * figLattices.numKeys,
                        cudaMemcpyHostToDevice));

// 3) Build host mirror of MyLatticeSet with patched device pointers
  MyLatticeSet hostFig;
  hostFig.d_keys = d_keys;
  hostFig.d_intervals = d_intervals;
  hostFig.myAxis = figLattices.myAxis;
  hostFig.myOtherAxis1 = figLattices.myOtherAxis1;
  hostFig.myOtherAxis2 = figLattices.myOtherAxis2;
  hostFig.numKeys = figLattices.numKeys;

// 4) Copy the MyLatticeSet struct to device
  MyLatticeSet *d_figLattices = nullptr;
  CUDA_CHECK(cudaMalloc(&d_figLattices, sizeof(MyLatticeSet)));
  CUDA_CHECK(cudaMemcpy(d_figLattices, &hostFig,
                        sizeof(MyLatticeSet), cudaMemcpyHostToDevice));

// optional: sync & check
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaError_t lastErr = cudaGetLastError();
  if (lastErr != cudaSuccess) {
    fprintf(stderr, "Post-copy CUDA error: %s\n", cudaGetErrorString(lastErr));
    // handle/exit as appropriate
    exit(EXIT_FAILURE);
  }

  trace.endBlock();

  GpuVisibility tmpVisibility(axis, d_segmentList, segmentSize, d_pointels, pointelsSize);

  std::cout << "Visibility initialized" << std::endl;
  std::cout << "Launching kernel with " << chunkAmount << " blocks and " << chunkSize << " threads per block"
            << std::endl;

  computeVisibilityKernel<<<chunkAmount, chunkSize>>>(
      axis, d_digital_dimensions, d_axises_idx,
      d_figLattices, tmpVisibility,
      d_segmentList, segmentSize
  );
  CUDA_CHECK(cudaDeviceSynchronize());

  std::cout << "Kernel finished" << std::endl;

  // Check for kernel launch errors
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }

  HostVisibility visibility(tmpVisibility);

  CUDA_CHECK(cudaFree(d_digital_dimensions));
  CUDA_CHECK(cudaFree(d_axises_idx));
  CUDA_CHECK(cudaFree(d_segmentList));
  CUDA_CHECK(cudaFree(d_pointels));
  CUDA_CHECK(cudaFree(d_keys));
  for (size_t i = 0; i < figLattices.numKeys; ++i) {
    CUDA_CHECK(cudaFree(hostIntervals[i].data));
  }
  CUDA_CHECK(cudaFree(d_intervals));
  CUDA_CHECK(cudaFree(d_figLattices));

  return visibility;
}
