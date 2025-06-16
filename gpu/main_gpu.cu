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

using namespace DGtal;
using namespace Z3i;

struct Vec3i {
  int x, y, z;

  __host__ __device__
  Vec3i() : x(0), y(0), z(0) {}

  __host__ __device__
  Vec3i(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}

  __host__ __device__
  Vec3i operator+(const Vec3i &other) const {
    return {x + other.x, y + other.y, z + other.z};
  }

  __host__ __device__
  Vec3i operator+=(const Vec3i &other) const {
    return {x + other.x, y + other.y, z + other.z};
  }

  __host__ __device__
  Vec3i operator-(const Vec3i &other) const {
    return {x - other.x, y - other.y, z - other.z};
  }

  __host__ __device__
  Vec3i operator*(int val) const {
    return {x * val, y * val, z * val};
  }

  __host__ __device__
  Vec3i operator/(int val) const {
    return {x / val, y / val, z / val};
  }

  __host__ __device__
  bool operator==(const Vec3i &other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  __host__ __device__
  bool operator!=(const Vec3i &other) const {
    return x != other.x || y != other.y || z != other.z;
  }

  __host__ __device__
  int &operator[](int index) {
    if (index == 0) return x;
    if (index == 1) return y;
    // Assume index == 2
    return z;
  }

  __host__ __device__
  const int &operator[](int index) const {
    if (index == 0) return x;
    if (index == 1) return y;
    // Assume index == 2
    return z;
  }
};


// Typedefs
typedef std::vector <Vec3i> Vec3is;
typedef Shortcuts <KSpace> SH3;
typedef ShortcutsGeometry <KSpace> SHG3;
typedef DigitalConvexity <KSpace> Convexity;
typedef typename Convexity::LatticeSet LatticeSet;
typedef CorrectedNormalCurrentComputer <RealPoint, RealVector> CNC;

// Polyscope
//polyscope::PointCloud *psVisibility = nullptr;
//polyscope::PointCloud *psVisibilityStart = nullptr;
//polyscope::SurfaceMesh *psPrimalMesh = nullptr;

// Global variables
KSpace K;
IntegerComputer <Integer> ic;
std::vector<int> digital_dimensions;
SH3::SurfelRange surfels;
std::vector <Point> pointels;
std::size_t pointel_idx = 0;
Point selected_point;
CountedPtr <SH3::ImplicitShape3D> implicit_shape(nullptr);
CountedPtr <SH3::BinaryImage> binary_image(nullptr);
CountedPtr <SH3::DigitalSurface> digital_surface(nullptr);
CountedPtr <SH3::SurfaceMesh> primal_surface(nullptr);

std::vector <RealPoint> visibility_normals;
std::vector <RealPoint> normalVisibilityColors;
std::vector <RealPoint> normalIIColors;
std::vector <RealPoint> visibility_sharps;

std::vector <RealPoint> true_normals;
std::vector <RealPoint> trivial_normals;
std::vector <RealPoint> surfel_true_normals;
std::vector <RealPoint> ii_normals;
std::vector <RealPoint> surfel_ii_normals;

// Curvature variables
CountedPtr <CNC> pCNC;
CNC::ScalarMeasure mu0;
CNC::ScalarMeasure mu1;
CNC::ScalarMeasure mu2;
CNC::TensorMeasure muXY;
std::vector<double> H;
std::vector<double> G;
std::vector<double> K1;
std::vector<double> K2;
std::vector <RealVector> D1;
std::vector <RealVector> D2;
float MaxCurv = 0.2;

// Parameters
int VisibilityRadius = 10;
double iiRadius = 3;
double gridstep = 1.0;
int OMP_max_nb_threads = 1;
double Time = 0.0;
bool noInterface = false;

struct Interval {
  int start;
  int end;
};

struct IntervalList {
  Interval *data;
  int capacity;
  int size;

  __host__ __device__ IntervalList() : data(nullptr), capacity(0), size(0) {}

  __host__ IntervalList(int maxCapacity) : data(nullptr), capacity(maxCapacity),
                                           size(0) {
    data = new Interval[capacity];
  }

  __host__ __device__ Interval *begin() const {
    return data;
  }

  __host__ __device__ Interval *end() const {
    return data + size;
  }

  __host__ __device__ bool empty() const {
    return size == 0;
  }

  __host__ __device__ Interval &operator[](int index) {
    ASSERT(index >= 0 && index < size && size <= capacity);
    return data[index];
  }

  __host__ __device__ const Interval &operator[](int index) const {
    ASSERT(index >= 0 && index < size && size <= capacity);
    return data[index];
  }
};

// Constants
const auto emptyIE = Vec3i();

Vec3i pointVectorToVec3i(const Point &vector) {
  return Vec3i(vector[0], vector[1], vector[2]);
}

Point vec3iToPointVector(const Vec3i &vector) {
  return Point(vector.x, vector.y, vector.z);
}

struct LatticeFoundResult {
  bool found;
  IntervalList intervals;
};

__host__ __device__ int myMax(int a, int b) {
  return (a > b) ? a : b;
}

__host__ __device__ int myMax3(int a, int b, int c) {
  return (a > b) ? myMax(a, c) : myMax(b, c);
}

__host__ __device__ int myMin(int a, int b) {
  return (a < b) ? a : b;
}

__host__ __device__ int myMin3(int a, int b, int c) {
  return (a < b) ? myMin(a, c) : myMin(b, c);
}

template<typename Type>
struct Buffer {
  Type *data;
  size_t capacity;
  size_t size;

  __host__ Buffer(size_t cap) : data(nullptr), capacity(cap), size(0) {
    cudaMalloc(&data, sizeof(Type) * capacity);
  }

  __host__ ~Buffer() {
    if (data) {
      cudaFree(data);
      data = nullptr;
    }
  }

  __device__ void push_back(const Type &v) {
    ASSERT(size < capacity);
    data[size++] = v;
  }

  __host__ __device__ Type &operator[](size_t index) {
    ASSERT(index < size);
    return data[index];
  }
};

struct MyLatticeSet {
  Vec3i *d_keys = nullptr;
  IntervalList *d_intervals = nullptr;
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
      intervalList.data = new Interval[ivs.size()];
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

  __device__ MyLatticeSet(const Vec3i segment, Buffer<Vec3i> &mainPointsBuf, Buffer<Vec3i> &keysBuf,
                          Buffer<IntervalList> &intervalsBuf, int axis)
      : myAxis(axis) {
    int otherAxis1 = (axis + 1) % 3;
    int otherAxis2 = (axis + 2) % 3;
    int currentIndex = 0;
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
    d_keys = keysBuf.data;
    d_intervals = intervalsBuf.data;

    int currentMaxKey = 0;

    for (size_t i = 0; i < currentIndex; ++i) {
      for (int j = -1; j < 2; ++j) {
        for (int k = -1; k < 2; ++k) {
          Vec3i key = mainPointsBuf[i];
          key[otherAxis1] += j;
          key[otherAxis2] += k;
          auto alreadyExists = this->find(key);
          if (!alreadyExists.found) {
            keysBuf.push_back(key);
            d_intervals[currentMaxKey - 1].size = 1;
            d_intervals[currentMaxKey - 1].capacity = 1;
            d_intervals[currentMaxKey - 1].data[0] = {key[axis] - 1, key[axis] + 1};
          } else {
            // If the key already exists, merge intervals
            auto &existingIntervals = d_intervals[alreadyExists.intervals.size - 1];
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

  __device__ LatticeFoundResult find(const Vec3i &p) {
    // Search for the point p in the lattice set
    for (size_t i = 0; i < numKeys; ++i) {
      if (d_keys[i] == p) {
        return {true, d_intervals[i]};
      }
    }
    return {false, d_intervals[0]};
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

struct FlatVisibility {
  int mainAxis;
  int vectorsSize;
  int pointsSize;
  bool *visibles;

  Vec3i *vectorList;  // flattened IntegerVectors
  Vec3i *pointList;

  FlatVisibility() = default;

  FlatVisibility(int mainAxis_, Vec3i *vectors, int vectorsSz, Vec3i *points, int pointsSz)
      : mainAxis(mainAxis_), vectorList(vectors), vectorsSize(vectorsSz),
        pointList(points), pointsSize(pointsSz) {
    size_t totalSize = pointsSize * vectorsSize;
    cudaMalloc(&visibles, totalSize * sizeof(bool));
    cudaMemset(visibles, 0, totalSize * sizeof(bool));
  }

  __device__
  void set(const Vec3i &offset, const IntervalList &value, size_t vectorIdx) const {
    Vec3i p = offset;
    for (const auto &interval: value) {
      for (int i = interval.start / 2; i <= interval.end / 2; ++i) {
        p[mainAxis] = i;
        int pointIdx = getPointIdx(p);
        if (pointIdx < pointsSize && vectorIdx < vectorsSize) {
          visibles[pointIdx * vectorsSize + vectorIdx] = true;
        }
      }
    }
  }

  __host__ __device__
  int getPointIdx(const Vec3i &p) const {
    // Search pointList for matching point (or use a flat hash map if performance needed)
    for (int i = 0; i < pointsSize; ++i) {
      if (pointList[i] == p) return i;
    }
    return pointsSize; // Not found
  }

  __host__ __device__
  int getVectorIdx(const Vec3i &v) const {
    for (int i = 0; i < vectorsSize; ++i) {
      if (vectorList[i] == v) return i;
    }
    return vectorsSize;
  }

  __host__ bool isVisible(const Vec3i &p1, const Vec3i &p2) const {
    if (p1 == p2) return true;

    if (!isPointLowerThan(p1, p2)) return isVisible(p2, p1);

    Vec3i v = p2 - p1;
    int d = gcd3(v.x, v.y, v.z);
    v = v / d;

    int vIdx = getVectorIdx(v);
    if (vIdx == vectorsSize) return false;

    for (Vec3i p = p1; p != p2; p += v) {
      int pIdx = getPointIdx(p);
      if (pIdx == pointsSize) return false;
      if (!visibles[pIdx * vectorsSize + vIdx]) return false;
    }
    return true;
  }
};

FlatVisibility visibility = FlatVisibility();

void embedPointels(const std::vector <Point> &vq, std::vector <RealPoint> &vp) {
  vp.clear();
  vp.reserve(vq.size());
  for (const auto &i: vq)
    vp.emplace_back(gridstep * (i[0] - 0.5),
                    gridstep * (i[1] - 0.5),
                    gridstep * (i[2] - 0.5));
}

void digitizePointels(const std::vector <RealPoint> &vp, std::vector <Point> &vq) {
  vq.clear();
  vq.reserve(vp.size());
  for (const auto &i: vp)
    vq.emplace_back(round(i[0] / gridstep + 0.5),
                    round(i[1] / gridstep + 0.5),
                    round(i[2] / gridstep + 0.5));
}

void listPolynomials() {
  auto L = SH3::getPolynomialList();
  for (const auto &p: L)
    trace.info() << p.first << " = " << p.second << std::endl;
}

/**
 * Get the main axis of the digital figure
 * @return
 */
int getLargeAxis() {
  auto axis = 0;
  for (auto i = 1; i < digital_dimensions.size() / 3; i++) {
    if (digital_dimensions[i] > digital_dimensions[axis]) axis = i;
  }
  return axis;
}

/**
 * Get the sizes of the digital figure
 * @return
 */
std::vector<int> getFigSizes() {
  auto d = pointels[0].dimension;
  std::vector<int> sizes = std::vector<int>(3 * d, 0);
  for (auto p: pointels) {
    for (auto i = 0; i < d; i++) {
      sizes[i + d] = std::min(sizes[i + d], p[i]);
      sizes[i + 2 * d] = std::max(sizes[i + 2 * d], p[i]);
    }
  }
  for (auto i = 0; i < d; i++) {
    sizes[i] = sizes[i + 2 * d] - sizes[i + d] + 1;
  }
  return sizes;
}

/**
 * Get all the unique integer vectors of maximum coordinate r
 * Only the vectors with gcd(x, y, z) = 1 are considered
 * @param r
 * @return
 */
Vec3is getAllVectors(int radius) {
  Vec3is vectors;
  for (auto x = radius; x >= 1; x--) {
    for (auto y = radius; y >= -radius; y--) {
      for (auto z = radius; z >= -radius; z--) {
        if (ic.gcd(ic.gcd(x, y), z) != 1) continue;
        vectors.emplace_back(x, y, z);
      }
    }
  }
  for (auto y = radius; y >= 1; y--) {
    for (auto z = radius; z >= -radius; z--) {
      if (ic.gcd(y, z) != 1) continue;
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
__device__ IntervalList checkInterval(IntervalList &buf, const Interval &toCheck, const IntervalList &figIntervals) {
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
                                    IntervalList &buf) {
  for (const auto &vInterval: vectorIntervals) {
    toCheck = intersect(buf, toCheck, checkInterval(buf, vInterval, figIntervals));
    if (toCheck.empty()) break;
  }
  return toCheck;
}

__global__ void computeVisibilityKernel(
    int axis, int *digital_dimensions, int *axises_idx,
    MyLatticeSet figLattices, FlatVisibility visibility,
    Vec3i *segmentList, int segmentSize,
    Buffer<Vec3i> *&mainsPointsBufGlobal, Buffer<Vec3i> *&keysBufGlobal, Buffer<IntervalList> *&intervalsBufGlobal,
    Buffer<IntervalList> &matchVectorBuf, Buffer<IntervalList> &eligiblesBuf
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= segmentSize) return;

  Vec3i segment = segmentList[idx];
  IntervalList &bufMatchVector = matchVectorBuf[idx];
  IntervalList &eligibles = eligiblesBuf[idx];
  MyLatticeSet latticeVector(segment, mainsPointsBufGlobal[idx], keysBufGlobal[idx], intervalsBufGlobal[idx], axis);
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
        if (res.found)
          eligibles = matchVector(eligibles, value, res.intervals, bufMatchVector);
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

__host__ void computeVisibilityGpu(int radius) {
  std::cout << "Computing visibility GPU" << std::endl;
  auto axis = getLargeAxis();
  auto tmpL = LatticeSetByIntervals<Space>(pointels.cbegin(), pointels.cend(), axis).starOfPoints();
  MyLatticeSet figLattices(tmpL);

//  const auto axises_idx = std::vector < int > {axis, axis == 0 ? 1 : 0, axis == 2 ? 1 : 2};
  int *axises_idx = new int[3]{axis, axis == 0 ? 1 : 0, axis == 2 ? 1 : 2};
  auto segmentList = getAllVectors(radius);

  Vec3i *pointelsData = new Vec3i[pointels.size()];
  for (size_t i = 0; i < pointels.size(); ++i) {
    pointelsData[i] = pointVectorToVec3i(pointels[i]);
  }
  visibility = FlatVisibility(axis, segmentList.data(), segmentList.size(), pointelsData, pointels.size());
  delete[] pointelsData;

  // Create all the buffers

  size_t N = segmentList.size();
  size_t M = 2 * radius + 1;
  size_t K = 18 * radius;
  size_t intervalCapacity = 2 * radius + 1;

  // mainsPointsBufGlobal : 2 * radius + 1 per buffer, segmentList.size() buffers
  void *raw = operator new[](N * sizeof(Buffer<Vec3i>));
  Buffer<Vec3i> *mainsPointsBufGlobal = static_cast<Buffer<Vec3i> *>(raw);

  for (size_t i = 0; i < N; ++i) {
    new(&mainsPointsBufGlobal[i]) Buffer<Vec3i>(M);
  }

  // keysBufGlobal : segmentList.size() buffers of size 9 * 2 * radius
  void *keysRaw = operator new[](N * sizeof(Buffer<Vec3i>));
  Buffer<Vec3i> *keysBufGlobal = static_cast<Buffer<Vec3i> *>(keysRaw);

  for (size_t i = 0; i < N; ++i) {
    new(&keysBufGlobal[i]) Buffer<Vec3i>(K);
  }

  // intervalsBufGlobal : segmentList.size() buffers of size 9 * 2 * radius, containing IntervalList of capacity radius * 2 + 1
  void *intervalsRaw = operator new[](N * sizeof(Buffer<IntervalList>));
  Buffer<IntervalList> *intervalsBufGlobal = static_cast<Buffer<IntervalList> *>(intervalsRaw);

  for (size_t i = 0; i < N; ++i) {
    new(&intervalsBufGlobal[i]) Buffer<IntervalList>(K);
    for (size_t j = 0; j < K; ++j) {
      new(&intervalsBufGlobal[i][j]) IntervalList(intervalCapacity);
    }
  }

  // matchVectorBuf : buffer of size segmentList.size() with IntervalList of capacity 2 * axis size + 1
  Buffer<IntervalList> matchVectorBuf(segmentList.size());
  for (size_t i = 0; i < segmentList.size(); ++i) {
    new(&matchVectorBuf[i]) IntervalList(2 * digital_dimensions[axis] + 1);
  }

  // eligiblesBuf : buffer of size segmentList.size() with IntervalList of capacity 2 * axis size + 1
  Buffer<IntervalList> eligiblesBuf(segmentList.size());
  for (size_t i = 0; i < segmentList.size(); ++i) {
    new(&eligiblesBuf[i]) IntervalList(2 * digital_dimensions[axis] + 1);
  }

  computeVisibilityKernel<<<(segmentList.size() + 255) / 256, 256>>>(
      axis,
      digital_dimensions.data(),
      axises_idx,
      figLattices, visibility, segmentList.data(), segmentList.size(),
      mainsPointsBufGlobal, keysBufGlobal, intervalsBufGlobal, matchVectorBuf, eligiblesBuf
  );
  cudaDeviceSynchronize();

  // Destruction
  for (size_t i = 0; i < N; ++i) {
    mainsPointsBufGlobal[i].~Buffer<Vec3i>();
  }
  operator delete[](raw);
  delete[] axises_idx;
  if (cudaGetLastError() != cudaSuccess) {
    std::cerr << "Error in computeVisibilityKernel: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Visibility computed" << std::endl;
}

/*

void computeVisibilityWithPointShow(std::size_t idx) {
  auto segmentRadius = VisibilityRadius;
  auto sphereRadius = VisibilityRadius;
  if (visibility.empty()) computeVisibilityGpu(segmentRadius);
  std::vector<Point> points;
  auto kdTree = LinearKDTree<Point, 3>(pointels);
  for (auto point_idx: kdTree.pointsInBall(pointels[idx], sphereRadius)) {
    auto tmp = kdTree.position(point_idx);
    if (visibility.isVisible(pointels[idx], tmp) && tmp != pointels[idx]) {
      points.push_back(tmp);
    }
  }
  std::vector<RealPoint> rpoints;
  embedPointels(points, rpoints);

  std::vector<RealPoint> rpointStart;
  embedPointels({pointels[idx]}, rpointStart);
  psVisibility = polyscope::registerPointCloud("Visibility", rpoints);
  psVisibility->setPointRadius(0.25 * gridstep, false);
  psVisibilityStart = polyscope::registerPointCloud("Start", rpointStart);
  psVisibilityStart->setPointRadius(0.3 * gridstep, false);
}

void checkParallelism() {
  // Check parallelism
  std::array<int, 512> counter{};
  std::array<int, 512> cnt_max{};
  counter.fill(0);
  cnt_max.fill(0);
#pragma omp parallel
  {
    auto n = omp_get_thread_num();
    auto m = omp_get_num_threads();
    counter[n] = n;
    cnt_max[n] = m;
  }
  bool ok = true;
  for (auto n = 0; n < cnt_max[0]; n++)
    if (counter[n] != n || cnt_max[n] != cnt_max[0])
      ok = false;
  trace.info() << "OMP parallelism " << (ok ? "(  OK  )" : "(ERROR)")
               << " #threads=" << cnt_max[0] << std::endl;
  OMP_max_nb_threads = cnt_max[0];
}

std::vector<size_t> pickSet(size_t N, size_t k, std::mt19937 &gen) {
  std::uniform_int_distribution<size_t> dis(1, N);
  std::vector<size_t> elems;

  while (elems.size() < k) {
    elems.push_back(dis(gen));
  }

  return elems;
}

void computeMeanDistanceVisibility() {
  // Print the gridstep
  std::cout << "Gridstep = " << gridstep << std::endl;

  // Print measure radius
  std::cout << "Visibility radius = " << VisibilityRadius << std::endl;

  // Print the mean distance between any two visible points
  double meanDistance = 0;
  int nbVisiblePairs = 0;

  // pick 100 random pointels

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 gen(seed);
  auto pointelsSample = pickSet(pointels.size(), pointels.size(), gen);
  auto kdTree = LinearKDTree<Point, 3>(pointels);

  for (auto i: pointelsSample) {
    for (auto point_idx: kdTree.pointsInBall(pointels[i], VisibilityRadius)) {
      auto tmp = kdTree.position(point_idx);
      if (isPointLowerThan(pointels[i], tmp) && visibility.isVisible(pointels[i], tmp) && tmp != pointels[i]) {
        meanDistance += (tmp - pointels[i]).norm();
        nbVisiblePairs++;
      }
    }
  }
  std::cout << "Mean distance between visible points = " << meanDistance / nbVisiblePairs << std::endl;
}

void computeVisibilityDirectionToSharpFeatures() {
  visibility_sharps.resize(pointels.size());
  auto kdTree = LinearKDTree<Point, 3>(pointels);
  for (int i = 0; i < pointels.size(); ++i) {
    RealVector tmpSum(0, 0, 0);
    size_t count = 0;
    for (auto point_idx: kdTree.pointsInBall(pointels[i], VisibilityRadius)) {
      auto tmp = kdTree.position(point_idx);
      if (visibility.isVisible(pointels[i], tmp) && tmp != pointels[i]) {
        tmpSum += tmp - pointels[i];
        count++;
      }
    }
    visibility_sharps[i] = -tmpSum / count;
  }
  if (!noInterface) {
    psPrimalMesh->addVertexVectorQuantity("Pointel visibility sharp direction", visibility_sharps);
  }
}
*/
double sigma = -1;
double minus2SigmaSquare = -2 * sigma * sigma;

double wSig(double d2) {
  return exp(d2 / minus2SigmaSquare);
}
/*

void computeVisibilityNormals() {
  visibility_normals.clear();
  visibility_normals.reserve(pointels.size());
  auto kdTree = LinearKDTree<Point, 3>(pointels);
  for (const auto &pointel: pointels) {
    std::vector<Point> visibles;
    RealPoint centroid(0, 0, 0);
    double total_w = 0.0;
    for (auto point_idx: kdTree.pointsInBall(pointel, 2 * sigma)) {
      auto tmp = kdTree.position(point_idx);
      if (visibility.isVisible(pointel, tmp)) { //  && tmp != pointel) {
        visibles.push_back(tmp);
//        centroid += tmp;
        const double w = wSig((pointel - tmp).squaredNorm());
        centroid += w * tmp;
        total_w += w;
      }
    }
    centroid /= total_w; // (double) visibles.size();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto &pt: visibles) {
      auto diff = pt - centroid;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
//          cov(i, j) += diff[i] * diff[j];
          cov(i, j) += diff[i] * diff[j] * wSig((pt - pointel).squaredNorm());
        }
      }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov / (double) visibles.size());
    auto normalE = solver.eigenvectors().col(0);
    visibility_normals.emplace_back(normalE.x(), normalE.y(), normalE.z());
  }
  if (!noInterface) {
    psPrimalMesh->addVertexVectorQuantity("Pointel visibility normals", visibility_normals);
    //psPrimalMesh->addVertexVectorQuantity("Pointel visibility normals", visibility_normals);
  }
}

RealVector getTrivNormal(size_t idx) {
  return primal_surface->vertexNormal(idx);
}

void reorientVisibilityNormals() {
  if (visibility_normals.empty()) {
    computeVisibilityNormals();
  }
//  if (visibility_sharps.empty()) {
//    computeVisibilityDirectionToSharpFeatures();
//  }
  for (int i = 0; i < visibility_normals.size(); ++i) {
    const auto triv_normal = getTrivNormal(i);
    if (visibility_normals[i].dot(triv_normal) < 0)
      visibility_normals[i] = -visibility_normals[i];
  }
  if (!noInterface) {
    psPrimalMesh->addVertexVectorQuantity("Pointel visibility normals", visibility_normals);
    psPrimalMesh->addVertexVectorQuantity("Pointel trivial normal", primal_surface->vertexNormals());
    psPrimalMesh->addFaceVectorQuantity("Face trivial normal", primal_surface->faceNormals());
  }
}

void computeCurvatures() {
  primal_surface->setVertexNormals(visibility_normals.cbegin(), visibility_normals.cend());
  primal_surface->computeFaceNormalsFromVertexNormals();

  mu0 = pCNC->computeMu0();
  mu1 = pCNC->computeMu1();
  mu2 = pCNC->computeMu2();
  muXY = pCNC->computeMuXY();

  H.resize(primal_surface->nbFaces());
  G.resize(primal_surface->nbFaces());
  K1.resize(primal_surface->nbFaces());
  K2.resize(primal_surface->nbFaces());
  D1.resize(primal_surface->nbFaces());
  D2.resize(primal_surface->nbFaces());
  // estimates mean (H) and Gaussian (G) curvatures by measure normalization.
  for (auto f = 0; f < primal_surface->nbFaces(); ++f) {
    const auto b = primal_surface->faceCentroid(f);
    const auto wfaces = primal_surface->computeFacesInclusionsInBall(VisibilityRadius, f, b);
    const auto area = mu0.faceMeasure(wfaces);
    H[f] = pCNC->meanCurvature(area, mu1.faceMeasure(wfaces));
    G[f] = pCNC->GaussianCurvature(area, mu2.faceMeasure(wfaces));
    const auto N = primal_surface->faceNormals()[f];
    const auto M = muXY.faceMeasure(wfaces);
    std::tie(K1[f], K2[f], D1[f], D2[f])
        = pCNC->principalCurvatures(area, M, N);
  }
}

void doRedisplayCurvatures() {
  psPrimalMesh->addFaceScalarQuantity("H (mean curvature)", H)
      ->setMapRange({-MaxCurv, MaxCurv})->setColorMap("coolwarm");
  psPrimalMesh->addFaceScalarQuantity("G (Gaussian curvature)", G)
      ->setMapRange({-0.5 * MaxCurv * MaxCurv, 0.5 * MaxCurv * MaxCurv})->setColorMap("coolwarm");
  psPrimalMesh->addFaceScalarQuantity("K1 (1st princ. curvature)", K1)
      ->setMapRange({-MaxCurv, MaxCurv})->setColorMap("coolwarm");
  psPrimalMesh->addFaceScalarQuantity("K2 (2nd princ. curvature)", K2)
      ->setMapRange({-MaxCurv, MaxCurv})->setColorMap("coolwarm");
  psPrimalMesh->addFaceVectorQuantity("D1 (1st princ. direction)", D1);
  psPrimalMesh->addFaceVectorQuantity("D2 (2nd princ. direction)", D2);
}

void doRedisplayNormalAsColors() {
  normalVisibilityColors.clear();
  normalIIColors.clear();
  normalVisibilityColors.reserve(visibility_normals.size());
  normalIIColors.reserve(visibility_normals.size());
  for (const auto &n: visibility_normals) {
    normalVisibilityColors.push_back(0.5f * (n + 1.0f));
  }
  for (const auto &n: ii_normals) {
    normalIIColors.push_back(0.5f * (n + 1.0f));
  }

  psPrimalMesh->addVertexColorQuantity("Normals visibility as colors", normalVisibilityColors);
  psPrimalMesh->addVertexColorQuantity("Normals II as colors", normalIIColors);
}

void computeL2looErrors() {
//  std::cout << "Computing L2 and Loo errors" << std::endl;
  std::vector<double> angle_dev(visibility_normals.size());
  std::vector<double> dummy(visibility_normals.size(), 0.0);
  for (int i = 0; i < visibility_normals.size(); ++i) {
    const auto sp = visibility_normals[i].dot(true_normals[i]);
    const auto fxp = std::min(1.0, std::max(-1.0, sp));
    angle_dev[i] = acos(fxp);
  }
  if (!noInterface)
    psPrimalMesh->addVertexScalarQuantity("Angle deviation", angle_dev)->setMapRange({0.0, 0.1})->setColorMap("reds");
  std::cout << "L2 error: " << SHG3::getScalarsNormL2(angle_dev, dummy) << std::endl;
  std::cout << "Loo error: " << SHG3::getScalarsNormLoo(angle_dev, dummy) << std::endl;
}
*/

/*void myCallback() {
  // Select a vertex with the mouse
  if (polyscope::pick::haveSelection()) {
    auto selection = polyscope::pick::getSelection();
    auto selectedSurface = static_cast<polyscope::SurfaceMesh *>(selection.first);
    auto idx = selection.second;

    // Only authorize selection on the input surface and the reconstruction
    auto surf = polyscope::getSurfaceMesh("Primal surface");
    const auto nv = selectedSurface->nVertices();
    // Validate that it its a face index
    if (selectedSurface == surf && idx < nv) {
      pointel_idx = idx;
      selected_point = pointels[pointel_idx];
      std::ostringstream otext;
      otext << "Selected pointel = " << pointel_idx
            << " pos=" << selected_point;
      ImGui::Text("%s", otext.str().c_str());
    }
  }
  ImGui::SliderInt("Visibility radius", &VisibilityRadius, 1, 20);
  *//*if (ImGui::Button("Visibility")) {
    computeVisibilityWithPointShow(pointel_idx);
  }
  ImGui::SameLine();*//*
  if (ImGui::Button("Visibilities OMP")) {
    trace.beginBlock("Compute visibilities OMP");
    computeVisibilityGpu(VisibilityRadius);
    Time = trace.endBlock();
  }
  *//*if (ImGui::Button("Measure Mean Distance Visibility")) {
    computeMeanDistanceVisibility();
  }
  if (ImGui::Button("Check OMP"))
    checkParallelism();
  if (ImGui::Button("Compute direction to sharp features")) {
    trace.beginBlock("Compute visibilities Direction To Sharp Features");
    computeVisibilityDirectionToSharpFeatures();
    Time = trace.endBlock();
  }
  if (ImGui::Button("Compute Normals")) {
    trace.beginBlock("Compute visibilities Normals");
    computeVisibilityNormals();
    reorientVisibilityNormals();
    Time = trace.endBlock();
    doRedisplayNormalAsColors();
  }
  ImGui::SameLine();
  if (ImGui::Button("Compute normal errors")) {
    trace.beginBlock("Compute visibilities Normals Errors");
    computeL2looErrors();
    Time = trace.endBlock();
  }
  if (ImGui::Button("Compute Curvatures")) {
    trace.beginBlock("Compute visibilities Curvatures");
    computeCurvatures();
    Time = trace.endBlock();
    doRedisplayCurvatures();
  }
  ImGui::SameLine();*//*
  ImGui::Text("nb threads = %d", OMP_max_nb_threads);
}*/

/*void testIntersection() {
  // 1|2 4|5 7|9
  for (auto v: intersect(Intervals({Interval(0, 9)}), Intervals({Interval(1, 2), Interval(4, 5), Interval(7, 10)}))) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  // 2|3 5|6 8|10
  for (auto v: intersect(Intervals({Interval(0, 3), Interval(5, 10)}), Intervals({Interval(2, 6), Interval(8, 12)}))) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  // 1|3 6|7 9|10 12|12 14|15
  for (auto v: intersect(Intervals({Interval(0, 3), Interval(5, 10), Interval(12, 15)}),
                         Intervals({Interval(1, 3), Interval(6, 7), Interval(9, 12), Interval(14, 15)}))) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
}*/


int main(int argc, char *argv[]) {
  // command line inteface options
  CLI::App app{"A tool to check visibility using lattices."};
  std::string filename = "../volumes/bunny34.vol";
  int thresholdMin = 0;
  int thresholdMax = 255;
  std::string polynomial;
  int minAABB = -10;
  int maxAABB = 10;
  bool listP = false;
  app.add_option("-i,--input", filename, "an input 3D vol file")->check(CLI::ExistingFile);
  // app.add_option("-o,--output", outputfilename, "the output OBJ filename");
  app.add_option("-p,--polynomial", polynomial,
                 "a polynomial like \"x^2+y^2+2*z^2-x*y*z+z^3-100\" or a named polynomial (see -l flag)");
  app.add_option("-g,--gridstep", gridstep, "the digitization gridstep");
  app.add_option("-m,--thresholdMin", thresholdMin,
                 "the minimal threshold m (excluded) for a voxel to belong to the digital shape.");
  app.add_option("-M,--thresholdMax", thresholdMax,
                 "the maximal threshold M (included) for a voxel to belong to the digital shape.");
  app.add_option("--minAABB", minAABB, "the lowest coordinate for the domain.");
  app.add_option("--maxAABB", maxAABB, "the highest coordinate for the domain.");
  app.add_option("-r,--radius", VisibilityRadius, "the radius of the visibility sphere");
  app.add_flag("-l", listP, "lists the known named polynomials.");
  app.add_flag("--noInterface", noInterface, "desactivate the interface and use the visibility OMP algorithm");
  app.add_option("--IIradius", iiRadius, "radius used for ii normal computation");
  double sigmaTmp = -1.0;
  app.add_option("-s,--sigma", sigmaTmp, "sigma used for visib normal computation");
  // -p "x^2+y^2+2*z^2-x*y*z+z^3-100" -g 0.5
  // Parse command line options. Exit on error.
  CLI11_PARSE(app, argc, argv)

  // React to some options.
  if (listP) {
    listPolynomials();
    return 0;
  }
  // Use options
  auto params = SH3::defaultParameters()
                | SHG3::defaultParameters()
                | SHG3::parametersGeometryEstimation();
  params("surfaceComponents", "All")("surfelAdjacency", 0); //exterior adjacency
  params("surfaceTraversal", "default");
  bool is_polynomial = !polynomial.empty();
  if (is_polynomial) {
    trace.beginBlock("Build polynomial surface");
    params("polynomial", polynomial);
    params("gridstep", gridstep);
    params("minAABB", minAABB);
    params("maxAABB", maxAABB);
    params("offset", 1.0);
    params("closed", 1);
    implicit_shape = SH3::makeImplicitShape3D(params);
    auto digitized_shape = SH3::makeDigitizedImplicitShape3D(implicit_shape, params);
    K = SH3::getKSpace(params);
    binary_image = SH3::makeBinaryImage(digitized_shape,
                                        SH3::Domain(K.lowerBound(),
                                                    K.upperBound()),
                                        params);
    trace.endBlock();
  } else {
    trace.beginBlock("Reading image vol file");
    params("thresholdMin", thresholdMin);
    params("thresholdMax", thresholdMax);
    binary_image = SH3::makeBinaryImage(filename, params);
    K = SH3::getKSpace(binary_image, params);
    trace.endBlock();
  }

  std::vector <std::vector<std::size_t>> primal_faces;
  std::vector <RealPoint> primal_positions;


  trace.beginBlock("Computing digital points and primal surface");
  // Build digital surface
  digital_surface = SH3::makeDigitalSurface(binary_image, K, params);
  primal_surface = SH3::makePrimalSurfaceMesh(digital_surface);
  surfels = SH3::getSurfelRange(digital_surface, params);
  if (is_polynomial) {
    surfel_true_normals = SHG3::getNormalVectors(implicit_shape, K, surfels, params);
  }
  // Need to convert the faces
  for (auto face = 0; face < primal_surface->nbFaces(); ++face)
    primal_faces.push_back(primal_surface->incidentVertices(face));
  // Embed with gridstep.
  for (auto v = 0; v < primal_surface->nbVertices(); v++)
    primal_surface->position(v) *= gridstep;
  primal_positions = primal_surface->positions();
  digitizePointels(primal_positions, pointels);
  digital_dimensions = getFigSizes();
  trace.info() << "Surface has " << pointels.size() << " pointels." << std::endl;
  trace.endBlock();

  // Compute trivial normals

  auto pTC = new TangencyComputer<KSpace>(K);
  pTC->init(pointels.cbegin(), pointels.cend(), true);
  int t_ring = int(round(params["t-ring"].as<double>()));
  auto surfel_trivial_normals = SHG3::getTrivialNormalVectors(K, surfels);
  primal_surface->faceNormals() = surfel_trivial_normals;
  params("r-radius", iiRadius);
  surfel_ii_normals = SHG3::getIINormalVectors(binary_image, surfels, params);
  for (auto i = 1; i < t_ring + 3; i++) {
    primal_surface->computeVertexNormalsFromFaceNormals();
    primal_surface->computeFaceNormalsFromVertexNormals();
    surfel_trivial_normals = primal_surface->faceNormals();
  }
  trivial_normals = primal_surface->vertexNormals();
  trivial_normals.resize(pointels.size());
  ii_normals.resize(pointels.size());
  true_normals.resize(pointels.size());
  for (auto &n: trivial_normals) n = RealVector::zero;
  for (auto &n: ii_normals) n = RealVector::zero;
  for (auto &n: true_normals) n = RealVector::zero;
  for (auto k = 0; k < surfels.size(); k++) {
    const auto &surf = surfels[k];
    const auto cells0 = SH3::getPrimalVertices(K, surf);
    for (const auto &c0: cells0) {
      const auto p = K.uCoords(c0);
      const auto idx = pTC->index(p);
      trivial_normals[idx] += surfel_trivial_normals[k];
      ii_normals[idx] += surfel_ii_normals[k];
      if (is_polynomial) {
        true_normals[idx] += surfel_true_normals[k];
      }
    }
  }
  for (auto &n: trivial_normals) n /= n.norm();
  for (auto &n: ii_normals) n /= n.norm();
  for (auto &n: true_normals) n /= n.norm();

  primal_surface->vertexNormals() = trivial_normals;

  if (sigmaTmp != -1.0) {
    sigma = sigmaTmp;
  } else {
    sigma = 5 * pow(gridstep, -0.5);
  }
  minus2SigmaSquare = -2 * sigma * sigma;

  std::cout << "sigma = " << sigma << std::endl;


  pCNC = CountedPtr<CNC>(new CNC(*primal_surface));
  // Initialize polyscope
  if (noInterface) {
    std::cout << "sigma = " << sigma << std::endl;
//    checkParallelism();
    trace.beginBlock("Compute visibilities");
    computeVisibilityGpu(VisibilityRadius);
    Time = trace.endBlock();
//    trace.beginBlock("Compute mean distance visibility");
//    computeMeanDistanceVisibility();
//    Time = trace.endBlock();
  }

  delete pTC;
  return EXIT_SUCCESS;

}
