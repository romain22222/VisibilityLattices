#ifndef MAIN_GPU_CUH
#define MAIN_GPU_CUH

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__
#define CUDA_GLOBAL __global__
#else
#define CUDA_HOSTDEV
#define CUDA_GLOBAL
#define CUDA_HOST
#endif

#include "Vec3i.cu"

struct IntervalGpu {
	int start;
	int end;
};

CUDA_HOSTDEV int myMax(int a, int b);

CUDA_HOSTDEV int myMin(int a, int b);

int gcd3(int a, int b, int c);

bool isPointLowerThan(const Vec3i &p1, const Vec3i &p2);

// Add declarations of GPU structs and kernels

struct IntervalList {
	IntervalGpu *data;
	int capacity;
	int size;

	CUDA_HOSTDEV IntervalList() : data(nullptr), capacity(0), size(0) {}

	CUDA_HOSTDEV ~IntervalList() = default;

#ifdef __CUDACC__

	__host__ __device__ IntervalList(int maxCapacity) : capacity(maxCapacity),
	                                                    size(0) {
//    cudaMalloc(&data, sizeof(IntervalGpu) * capacity);
		data = new IntervalGpu[maxCapacity];
	}

#endif

	CUDA_HOSTDEV IntervalGpu *begin() const {
		return data;
	}

	CUDA_HOSTDEV IntervalGpu *end() const {
		return data + size;
	}

	CUDA_HOSTDEV bool empty() const {
		return size == 0;
	}

	CUDA_HOSTDEV IntervalGpu &operator[](int index) {
		return data[index];
	}

	CUDA_HOSTDEV const IntervalGpu &operator[](int index) const {
		return data[index];
	}

	CUDA_HOSTDEV int push_back(const IntervalGpu &interval) {
		if (size >= capacity) {
			data[size++] = interval;
			return 1;
		}
		data[size++] = interval;
		return 0;
	}
};

struct LatticeFoundResult {
	int keyIndex{};
	IntervalList *intervals;
};

struct MyLatticeSet {
	Vec3i *d_keys{};
	IntervalList **d_intervals{};
	int myAxis;
	int myOtherAxis1;
	int myOtherAxis2;

	size_t numKeys = 0;

	CUDA_HOST MyLatticeSet() = default;

	CUDA_HOST MyLatticeSet(int axis, std::vector<Vec3i> &keys, std::vector<IntervalList *> &allIntervals);

#ifdef __CUDACC__

	__device__ MyLatticeSet(Vec3i segment, int axis);

//  __device__ ~MyLatticeSet();

	__device__ LatticeFoundResult find(const Vec3i &p) const;

	__device__ int findWithoutAxis(const Vec3i &p) const;

#endif
};

struct GpuVisibility {
	int mainAxis;
	int vectorsSize;
	int pointsSize;
	bool *visibles;

	Vec3i *vectorList;  // flattened IntegerVectors
	Vec3i *pointList;

	GpuVisibility() = default;

#ifdef __CUDACC__

	GpuVisibility(int mainAxis_, Vec3i *vectors, int vectorsSz, Vec3i *points, int pointsSz)
		: mainAxis(mainAxis_), vectorList(vectors), vectorsSize(vectorsSz),
		  pointList(points), pointsSize(pointsSz) {
		size_t totalSize = pointsSize * vectorsSize;
		cudaMalloc(&visibles, totalSize * sizeof(bool));
		cudaMemset(visibles, false, totalSize * sizeof(bool));
	}

	__device__
	void set(const Vec3i &offset, const IntervalList &value, size_t vectorIdx) const {
		auto p = offset;
		for (const auto &interval: value) {
			for (int i = interval.start / 2; i <= interval.end / 2; ++i) {
				p[mainAxis] = i;
				visibles[vectorIdx * pointsSize + this->getPointIdx(p)] = true;
			}
		}
	}

#endif

	CUDA_HOSTDEV
	int getPointIdx(const Vec3i &p) const {
		// Search pointList for matching point (or use a flat hash map if performance needed)
		for (int i = 0; i < pointsSize; ++i) {
			if (pointList[i] == p) return i;
		}
		return pointsSize; // Not found
	}

	CUDA_HOSTDEV
	int getVectorIdx(const Vec3i &v) const {
		for (int i = 0; i < vectorsSize; ++i) {
			if (vectorList[i] == v) return i;
		}
		return vectorsSize;
	}
};

struct HostVisibility {
	int mainAxis;
	int vectorsSize;
	int pointsSize;
	bool *visibles;
	Vec3i *vectorList;
	Vec3i *pointList;

	HostVisibility() = default;

#ifdef __CUDACC__

	HostVisibility(GpuVisibility flatVisibility) {
		mainAxis = flatVisibility.mainAxis;
		vectorsSize = flatVisibility.vectorsSize;
		pointsSize = flatVisibility.pointsSize;
		size_t totalSize = pointsSize * vectorsSize;

		visibles = new bool[totalSize];
		cudaMemcpy(visibles, flatVisibility.visibles, totalSize * sizeof(bool), cudaMemcpyDeviceToHost);
		vectorList = new Vec3i[vectorsSize];
		cudaMemcpy(vectorList, flatVisibility.vectorList, vectorsSize * sizeof(Vec3i), cudaMemcpyDeviceToHost);
		pointList = new Vec3i[pointsSize];
		cudaMemcpy(pointList, flatVisibility.pointList, pointsSize * sizeof(Vec3i), cudaMemcpyDeviceToHost);

		cudaFree(flatVisibility.visibles);
		cudaFree(flatVisibility.vectorList);
		cudaFree(flatVisibility.pointList);
	}

#else
	HostVisibility(GpuVisibility flatVisibility) {
	  std::cout << "HostVisibility constructor called without CUDA support!" << std::endl;
	  mainAxis = 0;
	  vectorsSize = 0;
	  pointsSize = 0;
	  visibles = nullptr;
	  vectorList = nullptr;
	  pointList = nullptr;
	}
#endif

	int getPointIdx(const Vec3i &p) const {
		// Search pointList for matching point (or use a flat hash map if performance needed)
		for (int i = 0; i < pointsSize; ++i) {
			if (pointList[i] == p) return i;
		}
		return pointsSize; // Not found
	}

	int getVectorIdx(const Vec3i &v) const {
		for (int i = 0; i < vectorsSize; ++i) {
			if (vectorList[i] == v) return i;
		}
		return vectorsSize;
	}

	bool isVisible(const Vec3i &p1, const Vec3i &p2) const {
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
			if (!visibles[vIdx * pointsSize + pIdx]) return false;
		}
		return true;
	}
};

CUDA_GLOBAL void computeVisibilityKernel(
	int axis, const int *digital_dimensions, const int *axises_idx,
	const MyLatticeSet &figLattices, GpuVisibility visibility,
	Vec3i *segmentList, int segmentSize
);


HostVisibility computeVisibility(
	int chunkAmount, int chunkSize,
	int axis, const int *digital_dimensions, const int *axises_idx,
	const MyLatticeSet &figLattices,
	Vec3i *segmentList, int segmentSize,
	Vec3i *pointels, int pointelsSize
);

#endif //MAIN_GPU_CUH