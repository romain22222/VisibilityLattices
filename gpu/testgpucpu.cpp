#include <iostream>
#include <vector>
#include <DGtal/base/Common.h>
#include "testgpucpu.h"

int myMaxCPU(int a, int b) {
	return (a > b) ? a : b;
}

int myMinCPU(int a, int b) {
	return (a < b) ? a : b;
}

/**
 * Arbitrary consistent order for points
 * @param p1
 * @param p2
 * @return
 */
bool isPointLowerThanCPU(const Vec3i &p1, const Vec3i &p2) {
	return p1.x < p2.x || (p1.x == p2.x && (p1.y < p2.y || (p1.y == p2.y && p1.z < p2.z)));
}

inline int gcdCPU(int a, int b) {
	while (b != 0) {
		int tmp = b;
		b = a % b;
		a = tmp;
	}
	return a;
}

inline int gcd3CPU(int a, int b, int c) {
	return gcdCPU(a, gcdCPU(b, c));
}

template<typename Type>
struct Buffer {
	Type *data;
	size_t capacity;
	size_t size;

	explicit Buffer(size_t cap) : capacity(cap), size(0) {
		data = new Type[capacity];
	}

	~Buffer() {
		if (data) {
			delete[] data;
		}
	}

	void push_back(const Type &v) {
		if (size >= capacity) {
			printf("Buffer overflow in push_back\n");
		}
		data[size++] = v;
	}

	Type &operator[](size_t index) {
		if (index >= size) {
			printf("Buffer index out of range in operator[]\n");
		}
		return data[index];
	}
};

IntervalListCPU::IntervalListCPU() : data(nullptr), capacity(0), size(0) {}

IntervalListCPU::IntervalListCPU(int maxCapacity) : capacity(maxCapacity), size(0) {
	data = new IntervalGpuCPU[capacity];
}

IntervalListCPU::~IntervalListCPU() = default;

IntervalGpuCPU *IntervalListCPU::begin() const {
	return data;
}

IntervalGpuCPU *IntervalListCPU::end() const {
	return data + size;
}

bool IntervalListCPU::empty() const {
	return size == 0;
}

IntervalGpuCPU &IntervalListCPU::operator[](int index) {
	return data[index];
}

const IntervalGpuCPU &IntervalListCPU::operator[](int index) const {
	return data[index];
}

int IntervalListCPU::push_back(const IntervalGpuCPU &interval) {
	if (size >= capacity) {
		data[size++] = interval;
		return 1;
	}
	data[size++] = interval;
	return 0;
}

GpuVisibilityCPU::GpuVisibilityCPU(int mainAxis_, Vec3i *vectors, int vectorsSz, Vec3i *points, int pointsSz)
	: mainAxis(mainAxis_), vectorList(vectors), vectorsSize(vectorsSz),
	  pointList(points), pointsSize(pointsSz) {
	size_t totalSize = pointsSize * vectorsSize;
	visibles = new bool[totalSize];
	for (int i = 0; i < totalSize; ++i) {
		visibles[i] = false;
	}
}

void GpuVisibilityCPU::set(const Vec3i &offset, const IntervalListCPU &value, size_t vectorIdx) const {
	auto p = offset;
	for (const auto &interval: value) {
		for (int i = interval.start / 2; i <= interval.end / 2; ++i) {
			p[mainAxis] = i;
			visibles[vectorIdx * pointsSize + this->getPointIdx(p)] = true;
		}
	}
}

int GpuVisibilityCPU::getPointIdx(const Vec3i &p) const {
	// Search pointList for matching point (or use a flat hash map if performance needed)
	for (int i = 0; i < pointsSize; ++i) {
		if (pointList[i] == p) return i;
	}
	return pointsSize; // Not found
}

int GpuVisibilityCPU::getVectorIdx(const Vec3i &v) const {
	for (int i = 0; i < vectorsSize; ++i) {
		if (vectorList[i] == v) return i;
	}
	return vectorsSize;
}

MyLatticeSetCPU::MyLatticeSetCPU(int axis, std::vector<Vec3i> &keys, std::vector<IntervalListCPU *> &allIntervals)
	: myAxis(axis), myOtherAxis1((axis + 1) % 3), myOtherAxis2((axis + 2) % 3), numKeys(keys.size()) {
	d_keys = new Vec3i[numKeys];
	d_intervals = new IntervalListCPU *[numKeys];
	for (auto i = 0; i < keys.size(); i++) {
		d_keys[i] = keys[i];
		d_intervals[i] = allIntervals[i];
	}
}

MyLatticeSetCPU::MyLatticeSetCPU(const Vec3i segment, int axis)
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
	d_keys = new Vec3i[allocateAmount];
	d_intervals = new IntervalListCPU *[allocateAmount];

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
					d_intervals[numKeys] = new IntervalListCPU();
					d_intervals[numKeys]->data = new IntervalGpuCPU[1];
					d_intervals[numKeys]->size = 1;
					d_intervals[numKeys]->capacity = 1;
					d_intervals[numKeys]->data[0] = {key[axis] - 1, key[axis] + 1};
					numKeys++;
				} else {
					// If the key already exists, merge intervals
					alreadyExists.intervals->data[0].start = myMinCPU(alreadyExists.intervals->data[0].start,
					                                                  key[axis] - 1);
					alreadyExists.intervals->data[0].end = myMaxCPU(alreadyExists.intervals->data[0].end,
					                                                key[axis] + 1);
				}
			}
		}
	}
}

LatticeFoundResultCPU MyLatticeSetCPU::find(const Vec3i &p) const {
	// Search for the point p in the lattice set
	for (size_t i = 0; i < numKeys; ++i) {
		if (d_keys[i] == p) {
			return {static_cast<int>(i), d_intervals[i]};
		}
	}
	return {-1, nullptr};
}

LatticeFoundResultCPU MyLatticeSetCPU::findWithoutAxis(const Vec3i &p) const {
	// Search for the point p in the lattice set
	for (size_t i = 0; i < numKeys; ++i) {
		if (d_keys[i][myOtherAxis1] == p[myOtherAxis1] && d_keys[i][myOtherAxis2] == p[myOtherAxis2]) {
			return {static_cast<int>(i), d_intervals[i]};
		}
	}
	return {-1, nullptr};
}

HostVisibilityCPU::HostVisibilityCPU(GpuVisibilityCPU flatVisibility) {
	mainAxis = flatVisibility.mainAxis;
	vectorsSize = flatVisibility.vectorsSize;
	pointsSize = flatVisibility.pointsSize;
	size_t totalSize = pointsSize * vectorsSize;

	visibles = new bool[totalSize];
	for (size_t i = 0; i < totalSize; ++i) {
		visibles[i] = flatVisibility.visibles[i];
	}
	vectorList = new Vec3i[vectorsSize];
	for (size_t i = 0; i < vectorsSize; ++i) {
		vectorList[i] = flatVisibility.vectorList[i];
	}
	pointList = new Vec3i[pointsSize];
	for (size_t i = 0; i < pointsSize; ++i) {
		pointList[i] = flatVisibility.pointList[i];
	}
}

int HostVisibilityCPU::getPointIdx(const Vec3i &p) const {
	// Search pointList for matching point (or use a flat hash map if performance needed)
	for (int i = 0; i < pointsSize; ++i) {
		if (pointList[i] == p) return i;
	}
	return pointsSize; // Not found
}

int HostVisibilityCPU::getVectorIdx(const Vec3i &v) const {
	for (int i = 0; i < vectorsSize; ++i) {
		if (vectorList[i] == v) return i;
	}
	return vectorsSize;
}

bool HostVisibilityCPU::isVisible(const Vec3i &p1, const Vec3i &p2) const {
	if (p1 == p2) return true;

	if (!isPointLowerThanCPU(p1, p2)) return isVisible(p2, p1);

	Vec3i v = p2 - p1;
	int d = gcd3CPU(v.x, v.y, v.z);
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

/**
 * Check if the interval toCheck is contained in figIntervals
 * @param toCheck
 * @param figIntervals
 * @return
 */
IntervalListCPU
checkIntervalCPU(IntervalListCPU &buf, const IntervalGpuCPU &toCheck, const IntervalListCPU &figIntervals) {
//  IntervalListCPU result(figIntervals.size);
	buf.size = 0;
	int err = 0;
	const auto toCheckSize = toCheck.end - toCheck.start;
	for (const auto &interval: figIntervals) {
		if (interval.end - interval.start >= toCheckSize) {
			err = buf.push_back({interval.start - toCheck.start, interval.end - toCheck.end});
			if (err) {
				printf("Error pushing back in checkIntervalCPU\n");
			}
		}
	}
	return buf;
}

void intersectCPU(IntervalListCPU &buf, IntervalListCPU &l1, const IntervalListCPU &l2) {
	buf.size = 0;
	int err = 0;
	int k1 = 0, k2 = 0;
	while (k1 < l1.size && k2 < l2.size) {
		const auto interval1 = l1[k1];
		const auto interval2 = l2[k2];
		const auto i = myMaxCPU(interval1.start, interval2.start);
		const auto j = myMinCPU(interval1.end, interval2.end);
		if (i <= j) {
			err = buf.push_back({i, j});
			if (err) std::cout << "Error pushing back in intersectCPU" << std::endl;
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
IntervalListCPU matchVectorCPU(
	IntervalListCPU &buf,
	IntervalListCPU &buf2,
	IntervalListCPU &toCheck,
	const IntervalListCPU &vectorIntervals,
	const IntervalListCPU &figIntervals) {
	for (const auto &vInterval: vectorIntervals) {
		intersectCPU(buf2, toCheck, checkIntervalCPU(buf, vInterval, figIntervals));
		if (toCheck.empty()) break;
	}
	return toCheck;
}

void computeVisibilityKernelCPU(
	int axis, const int *digital_dimensions, const int *axises_idx,
	MyLatticeSetCPU *figLattices, GpuVisibilityCPU visibility,
	Vec3i *segmentList, int idx
) {
	Vec3i segment = segmentList[idx];
	IntervalListCPU buf(2 * digital_dimensions[axis] + 1);
	IntervalListCPU buf2(2 * digital_dimensions[axis] + 1);
	IntervalListCPU eligibles(2 * digital_dimensions[axis] + 1);

	MyLatticeSetCPU latticeVector(segment, axis);
	int minTx = digital_dimensions[axises_idx[1] + 3] - myMinCPU(0, segment[axises_idx[1]]);
	int maxTx = digital_dimensions[axises_idx[1] + 6] + 1 - myMaxCPU(0, segment[axises_idx[1]]);
	int minTy = digital_dimensions[axises_idx[2] + 3] - myMinCPU(0, segment[axises_idx[2]]);
	int maxTy = digital_dimensions[axises_idx[2] + 6] + 1 - myMaxCPU(0, segment[axises_idx[2]]);
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
				eligibles = matchVectorCPU(buf, buf2, eligibles, *value, *res.intervals);
				if (eligibles.empty()) break;
			}
			if (!eligibles.empty()) visibility.set(pInterest / 2, eligibles, idx);
		}
	}
	delete[] buf.data;
	delete[] buf2.data;
	delete[] eligibles.data;
	delete[] latticeVector.d_keys;
	for (size_t i = 0; i < latticeVector.numKeys; ++i) {
		delete[] latticeVector.d_intervals[i]->data;
	}
	delete[] latticeVector.d_intervals;
}

HostVisibilityCPU computeVisibilityCPU(
	int chunkAmount, int chunkSize,
	int axis, const int *digital_dimensions, const int *axises_idx,
	const MyLatticeSetCPU &figLattices,
	Vec3i *segmentList, int segmentSize,
	Vec3i *pointels, int pointelsSize
) {
	if (segmentSize == 0 || pointelsSize == 0 || figLattices.numKeys == 0) {
		std::cout << "Empty input, returning empty visibility" << std::endl;
		return {};
	}
	using namespace DGtal;
	trace.beginBlock("GPU memory allocation and copy");
	// Malloc every list on GPU
	int *d_digital_dimensions = new int[9];
	int *d_axises_idx = new int[3];
	auto *d_segmentList = new Vec3i[segmentSize];
	auto *d_pointels = new Vec3i[pointelsSize];
	for (int i = 0; i < 9; i++) d_digital_dimensions[i] = digital_dimensions[i];
	for (int i = 0; i < 3; i++) d_axises_idx[i] = axises_idx[i];
	for (int i = 0; i < segmentSize; i++) d_segmentList[i] = segmentList[i];
	for (int i = 0; i < pointelsSize; i++) d_pointels[i] = pointels[i];
	trace.endBlock();

	trace.beginBlock("FigLattices copy to GPU");
	// Malloc figLattices on GPU
	// 1. Copy d_keys
	auto *d_keys = new Vec3i[figLattices.numKeys];
	for (size_t i = 0; i < figLattices.numKeys; ++i) {
		d_keys[i] = figLattices.d_keys[i];
	}
	std::vector<IntervalListCPU> hostIntervals(figLattices.numKeys);

	for (size_t i = 0; i < figLattices.numKeys; ++i) {
		hostIntervals[i].capacity = figLattices.d_intervals[i]->capacity;
		hostIntervals[i].size = figLattices.d_intervals[i]->size;

		auto *d_data = new IntervalGpuCPU[figLattices.d_intervals[i]->size];
		for (size_t j = 0; j < figLattices.d_intervals[i]->size; ++j) {
			d_data[j] = figLattices.d_intervals[i]->data[j];
		}
		hostIntervals[i].data = d_data; // now points to device memory
	}

	auto *d_intervals = new IntervalListCPU *[figLattices.numKeys];
	for (size_t i = 0; i < figLattices.numKeys; ++i) {
		d_intervals[i] = &hostIntervals.data()[i];
	}

	// 3. Build a host mirror of MyLatticeSetCPU with patched device pointers
	MyLatticeSetCPU hostFig;
	hostFig.d_keys = d_keys;
	hostFig.d_intervals = d_intervals;
	hostFig.myAxis = figLattices.myAxis;
	hostFig.myOtherAxis1 = figLattices.myOtherAxis1;
	hostFig.myOtherAxis2 = figLattices.myOtherAxis2;
	hostFig.numKeys = figLattices.numKeys;

// 4. Copy it to device
	auto *d_figLattices = new MyLatticeSetCPU();
	*d_figLattices = hostFig;
	trace.endBlock();

	GpuVisibilityCPU tmpVisibility(axis, d_segmentList, segmentSize, d_pointels, pointelsSize);

	std::cout << "Visibility initialized" << std::endl;
	std::cout << "Launching kernel with " << chunkAmount << " blocks and " << chunkSize << " threads per block"
	          << std::endl;

	for (int i = 0; i < segmentSize; i++) {
		computeVisibilityKernelCPU(
			axis, d_digital_dimensions, d_axises_idx,
			d_figLattices, tmpVisibility,
			d_segmentList, i
		);
	}

	std::cout << "Kernel finished" << std::endl;

	// Check for kernel launch errors

	HostVisibilityCPU visibility(tmpVisibility);

	delete[] d_digital_dimensions;
	delete[] d_axises_idx;
	delete[] d_segmentList;
	delete[] d_pointels;
	delete[] d_keys;
	for (size_t i = 0; i < figLattices.numKeys; ++i) {
		delete[] hostIntervals[i].data;
	}
	delete[] d_intervals;
	delete d_figLattices;

	return visibility;
}