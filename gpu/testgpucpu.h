//
// Created by romro on 29/09/2025.
//

#ifndef VISIBILITYLATTICES_TESTGPUCPU_H
#define VISIBILITYLATTICES_TESTGPUCPU_H

#include "Vec3i.cu"

int myMaxCPU(int a, int b);

int myMinCPU(int a, int b);

bool isPointLowerThanCPU(const Vec3i &p1, const Vec3i &p2);

inline int gcdCPU(int a, int b);

inline int gcd3CPU(int a, int b, int c);

struct IntervalGpuCPU {
	int start;
	int end;
};

struct IntervalListCPU {
	IntervalGpuCPU *data;
	int capacity;
	int size;

	IntervalListCPU();

	IntervalListCPU(int maxCapacity);

	~IntervalListCPU();

	IntervalGpuCPU *begin() const;

	IntervalGpuCPU *end() const;

	bool empty() const;

	IntervalGpuCPU &operator[](int index);

	const IntervalGpuCPU &operator[](int index) const;

	int push_back(const IntervalGpuCPU &interval);
};

struct GpuVisibilityCPU {
	int mainAxis;
	int vectorsSize;
	int pointsSize;
	bool *visibles;

	Vec3i *vectorList;  // flattened IntegerVectors
	Vec3i *pointList;

	GpuVisibilityCPU() = default;

	GpuVisibilityCPU(int mainAxis_, Vec3i *vectors, int vectorsSz, Vec3i *points, int pointsSz);

	void set(const Vec3i &offset, const IntervalListCPU &value, size_t vectorIdx) const;

	int getPointIdx(const Vec3i &p) const;

	int getVectorIdx(const Vec3i &v) const;
};

struct LatticeFoundResultCPU {
	int keyIndex{};
	IntervalListCPU *intervals;
};

struct MyLatticeSetCPU {
	Vec3i *d_keys{};
	IntervalListCPU **d_intervals{};
	int myAxis;
	int myOtherAxis1;
	int myOtherAxis2;

	size_t numKeys = 0;

	MyLatticeSetCPU() = default;

	MyLatticeSetCPU(int axis, std::vector<Vec3i> &keys, std::vector<IntervalListCPU *> &allIntervals);

	MyLatticeSetCPU(const Vec3i segment, int axis);

	LatticeFoundResultCPU find(const Vec3i &p) const;

	LatticeFoundResultCPU findWithoutAxis(const Vec3i &p) const;
};

struct HostVisibilityCPU {
	int mainAxis;
	int vectorsSize;
	int pointsSize;
	bool *visibles;
	Vec3i *vectorList;
	Vec3i *pointList;

	HostVisibilityCPU() = default;

	HostVisibilityCPU(GpuVisibilityCPU flatVisibility);

	int getPointIdx(const Vec3i &p) const;

	int getVectorIdx(const Vec3i &v) const;

	bool isVisible(const Vec3i &p1, const Vec3i &p2) const;
};

HostVisibilityCPU computeVisibilityCPU(
	int chunkAmount, int chunkSize,
	int axis, const int *digital_dimensions, const int *axises_idx,
	const MyLatticeSetCPU &figLattices,
	Vec3i *segmentList, int segmentSize,
	Vec3i *pointels, int pointelsSize
);

#endif //VISIBILITYLATTICES_TESTGPUCPU_H
