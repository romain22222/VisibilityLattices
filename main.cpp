#include <iostream>
#include <random>
#include <string>
#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>
#include <polyscope/pick.h>
#include <polyscope/surface_mesh.h>

// For dgtal libraries, store NDEBUG in another macro, then undef it during inclusion
#ifndef NDEBUG
#define NDEBUG
#define DGtal_NDEBUG
#endif

#include <DGtal/arithmetic/IntegerComputer.h>
#include <DGtal/base/Common.h>
#include <DGtal/geometry/meshes/CorrectedNormalCurrentComputer.h>
#include <DGtal/geometry/volumes/DigitalConvexity.h>
#include <DGtal/geometry/volumes/TangencyComputer.h>
#include <DGtal/helpers/StdDefs.h>
#include <DGtal/helpers/Shortcuts.h>
#include <DGtal/helpers/ShortcutsGeometry.h>
#include "DGtal/io/viewers/PolyscopeViewer.h"
#include "additionnalClasses/LinearKDTree.h"

#ifdef DGtal_NDEBUG
#undef NDEBUG
#undef DGtal_NDEBUG
#endif

#include "CLI11.hpp"
#include "omp.h"
#include "gpu/Vec3i.cu"
#include "src/polyhedra.h"

#ifdef USE_CUDA_VISIBILITY

#include "./gpu/middleware.h"

#endif

using namespace DGtal;
using namespace Z3i;

// Typedefs
typedef PointVector<3, Integer> IntegerVector;
typedef std::vector<IntegerVector> IntegerVectors;
typedef Shortcuts<KSpace> SH3;
typedef ShortcutsGeometry<KSpace> SHG3;
typedef DigitalConvexity<KSpace> Convexity;
typedef typename Convexity::LatticeSet LatticeSet;
typedef CorrectedNormalCurrentComputer<RealPoint, RealVector> CNC;

typedef std::pair<Integer, Integer> Interval;
typedef std::vector<Interval> Intervals;

// Polyscope
polyscope::PointCloud *psVisibility = nullptr;
polyscope::PointCloud *psVisibilityStart = nullptr;
polyscope::SurfaceMesh *psPrimalMesh = nullptr;

// Global variables
KSpace K;
IntegerComputer<Integer> ic;
std::vector<int> digital_dimensions;
SH3::SurfelRange surfels;
std::vector<Point> pointels;
std::size_t pointel_idx = 0;
Point selected_point;
CountedPtr<SH3::ImplicitShape3D> implicit_shape(nullptr);
CountedPtr<Polyhedra::PolyhedronShape> polyhedra(nullptr);
CountedPtr<SH3::BinaryImage> binary_image(nullptr);
CountedPtr<SH3::DigitalSurface> digital_surface(nullptr);
CountedPtr<SH3::SurfaceMesh> primal_surface(nullptr);

std::vector<RealPoint> visibility_normals;
std::vector<RealPoint> normalVisibilityColors;
std::vector<RealPoint> normalIIColors;
std::vector<RealPoint> normalTrueColors;
std::vector<RealPoint> visibility_sharps;

std::vector<RealPoint> mitra_normals;
std::vector<RealPoint> normalMitraColors;

std::vector<RealPoint> true_normals;
std::vector<RealPoint> trivial_normals;
std::vector<RealPoint> surfel_true_normals;
std::vector<RealPoint> ii_normals;
std::vector<RealPoint> surfel_ii_normals;

// Curvature variables
CountedPtr<CNC> pCNC;
CNC::ScalarMeasure mu0;
CNC::ScalarMeasure mu1;
CNC::ScalarMeasure mu2;
CNC::TensorMeasure muXY;
std::vector<double> H;
std::vector<double> G;
std::vector<double> K1;
std::vector<double> K2;
std::vector<RealVector> D1;
std::vector<RealVector> D2;
float MaxCurv = 0.1;

// Parameters
int VisibilityRadius = 10;
double iiRadius = 3;
double gridstep = 1.0;
int OMP_max_nb_threads = 1;
double Time = 0.0;
bool noInterface = false;
std::string polynomial;
char *inputVisibilityFilename = new char[256]("visibility.vis");
int nStarTest = 1;

// Constants
const auto emptyIE = IntegerVector();

/**
 * Arbitrary consistent order for points
 * @param p1
 * @param p2
 * @return
 */
bool isPointLowerThan(const Point &p1, const Point &p2) {
	return p1[0] < p2[0] || (p1[0] == p2[0] && p1[1] < p2[1]) || (p1[0] == p2[0] && p1[1] == p2[1] && p1[2] < p2[2]);
}

Point vec3iToPointVector(const Vec3i &vector) {
	return Point(vector.x, vector.y, vector.z);
}

class ComputeVisibilityParams : public Parameters {
public:
	ComputeVisibilityParams() {
		this->operator()("nStar", 1);
	}
};

class Visibility {
public:
	Dimension mainAxis{};
	std::vector<bool> visibles;
	size_t vectorsSize{};
	size_t pointsSize{};
	std::map<Point, size_t> pointIdxs;
	std::map<IntegerVector, size_t> vectorIdxs;

	Visibility() = default;

	Visibility(Dimension mainAxis, IntegerVectors vectors, std::vector<Point> points) : mainAxis(mainAxis) {
		vectorsSize = vectors.size();
		pointsSize = points.size();
		visibles = std::vector<bool>(vectorsSize * pointsSize, false);
		for (size_t i = 0; i < pointsSize; i++) {
			pointIdxs[points[i]] = i;
		}
		for (size_t i = 0; i < vectorsSize; i++) {
			vectorIdxs[vectors[i]] = i;
		}
	}

#ifdef USE_CUDA_VISIBILITY

	Visibility(const HostVisibility hostVisibility) {
		std::cout << "Reading visibility from GPU" << std::endl;
		mainAxis = hostVisibility.mainAxis;
		vectorsSize = hostVisibility.vectorsSize;
		pointsSize = hostVisibility.pointsSize;
		visibles = std::vector<bool>(hostVisibility.visibles, hostVisibility.visibles + vectorsSize * pointsSize);
		for (size_t i = 0; i < pointsSize; i++) {
			pointIdxs[vec3iToPointVector(hostVisibility.pointList[i])] = i;
		}
		for (size_t i = 0; i < vectorsSize; i++) {
			vectorIdxs[vec3iToPointVector(hostVisibility.vectorList[i])] = i;
		}
	}

	Visibility(const HostVisibilityCPU hostVisibility) {
		std::cout << "Reading visibility from GPU" << std::endl;
		mainAxis = hostVisibility.mainAxis;
		vectorsSize = hostVisibility.vectorsSize;
		pointsSize = hostVisibility.pointsSize;
		visibles = std::vector<bool>(hostVisibility.visibles, hostVisibility.visibles + vectorsSize * pointsSize);
		for (size_t i = 0; i < pointsSize; i++) {
			pointIdxs[vec3iToPointVector(hostVisibility.pointList[i])] = i;
		}
		for (size_t i = 0; i < vectorsSize; i++) {
			vectorIdxs[vec3iToPointVector(hostVisibility.vectorList[i])] = i;
		}
	}

#endif

	size_t getVectorIdx(const IntegerVector &v) const {
		auto it = vectorIdxs.find(v);
		if (it != vectorIdxs.end()) return it->second;
		return vectorsSize;
	}

	size_t getPointIdx(const IntegerVector &p) const {
		auto it = pointIdxs.find(p);
		if (it != pointIdxs.end()) return it->second;
		return pointsSize;
	}

	bool isVisible(const Point &p1, const Point &p2) const {
		if (p1 == p2) return true;
		// Reorder points so that v is in the upper half plane
		if (!isPointLowerThan(p1, p2)) return isVisible(p2, p1);
		IntegerVector v = p2 - p1;
		v = v / ic.gcd(v[0], ic.gcd(v[1], v[2]));
		auto vIdx = this->getVectorIdx(v);
		if (vIdx == vectorsSize) return false; // vector not computed
		for (IntegerVector p = p1; p != p2; p += v) {
			auto pointIdx = this->getPointIdx(p);
			if (pointIdx == pointsSize) return false;
			if (!visibles[vIdx * pointsSize + pointIdx]) return false;
		}
		return true;
	}

	void set(const Point &offset, const Intervals &value, const size_t vectorIdx) {
		auto p = offset / 2;
		for (auto &interval: value) {
			for (int i = interval.first / 2; i <= interval.second / 2; i++) {
				p[mainAxis] = i;
				auto pointIdx = this->getPointIdx(p);
				if (pointIdx == pointsSize) continue;
				visibles[vectorIdx * pointsSize + pointIdx] = true;
			}
		}
	}

	bool empty() const {
		return std::all_of(visibles.cbegin(), visibles.cend(), [](bool v) { return !v; });
	}

	void reset(Dimension mainAxis, IntegerVectors vectors, std::vector<Point> points) {
		this->mainAxis = mainAxis;
		vectorsSize = vectors.size();
		pointsSize = points.size();
		visibles = std::vector<bool>(vectorsSize * pointsSize, false);
		pointIdxs.clear();
		vectorIdxs.clear();
		for (size_t i = 0; i < pointsSize; i++) {
			pointIdxs[points[i]] = i;
		}
		for (size_t i = 0; i < vectorsSize; i++) {
			vectorIdxs[vectors[i]] = i;
		}
	}
};

Visibility visibility = Visibility();

void embedPointels(const std::vector<Point> &vq, std::vector<RealPoint> &vp) {
	vp.clear();
	vp.reserve(vq.size());
	for (const auto &i: vq)
		vp.emplace_back(gridstep * (i[0] - 0.5),
		                gridstep * (i[1] - 0.5),
		                gridstep * (i[2] - 0.5));
}

void digitizePointels(const std::vector<RealPoint> &vp, std::vector<Point> &vq) {
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
Dimension getLargeAxis() {
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
IntegerVectors getAllVectors(int radius) {
	IntegerVectors vectors;
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
 * From a given segment, get its star lattice
 * @param segment
 * @param axis
 * @return
 */
LatticeSet getLatticeVector(const IntegerVector &segment, Dimension axis) {
	LatticeSet L_ab(axis);
	L_ab.insert(IntegerVector());
	for (Dimension k = 0; k < 3; k++) {
		const Integer n = (segment[k] >= 0) ? segment[k] : -segment[k];
		const Integer d = (segment[k] >= 0) ? 1 : -1;
		if (n == 0) continue;
		Point kc;
		for (Integer i = 1; i < n; i++) {
			for (Dimension j = 0; j < 3; j++) {
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
			L_ab.insert(kc);
		}
	}
	if (segment != IntegerVector()) L_ab.insert(2 * segment);
	return L_ab.starOfCells();
}

/**
 * Check if the interval toCheck is contained in figIntervals
 * @param toCheck
 * @param figIntervals
 * @return
 */
Intervals
checkInterval(const Interval toCheck, const Intervals &figIntervals, bool debug = false) {
	Intervals result;
	result.reserve(figIntervals.size());
	const auto toCheckSize = toCheck.second - toCheck.first;
	int count = 0;
	for (auto interval: figIntervals) {
		if (debug) count++;
		if (interval.second - interval.first >= toCheckSize) {
			result.emplace_back(interval.first - toCheck.first, interval.second - toCheck.second);
		}
	}
	if (debug) printf("checkInterval: %d loops\n", count);
	return result;
}

Intervals intersect(const Intervals &l1, const Intervals &l2, bool debug = false) {
	Intervals result;
	// check 3rd example of testIntersection to see that you need at most this amount
	result.reserve(l1.size() + l2.size() - 1);
	//	result.reserve( std::max( l1.size(), l2.size() ) );
	int k1 = 0, k2 = 0;
	int count = 0;
	while (k1 < l1.size() && k2 < l2.size()) {
		if (debug) count++;
		const auto interval1 = l1[k1];
		const auto interval2 = l2[k2];
		const auto i = std::max(interval1.first, interval2.first);
		const auto j = std::min(interval1.second, interval2.second);
		if (i <= j) result.emplace_back(i, j);
		if (interval1.second <= interval2.second) k1++;
		if (interval1.second >= interval2.second) k2++;
	}
	if (debug) printf("intersect: %d loops\n", count);
	return result;
}

/**
 * From the current interval toCheck, find the shifts of latticeVector that makes latticeVector contained in figLattices
 * @param toCheck
 * @param latticeVector
 * @param figLattices
 * @return
 */
Intervals matchVector(Intervals &toCheck,
                      const Intervals &vectorIntervals,
                      const Intervals &figIntervals,
                      bool debug = false) {
	auto count = 0;
	for (const auto &vInterval: vectorIntervals) {
		toCheck = intersect(toCheck, checkInterval(vInterval, figIntervals, debug), debug);
		if (debug) count++;
		if (toCheck.empty()) break;
	}
	if (debug) printf("matchVector: %d loops\n", count);
	return toCheck;
}

LatticeSetByIntervals<Space> getCoverOfShape(LatticeSetByIntervals<Space> &shapeLattice, int dim) {
	LatticeSetByIntervals<Space> result = shapeLattice;

	for (auto &pV: shapeLattice.data()) {
		// Each pv is a point entry and all its shifts which are in the star. So we also need to add all its upper and lower neighbors and dilate the intervals too
		// 1. Get the current point and its intervals
		const Point &p = pV.first;
		LatticeSetByIntervals<Space>::Intervals::Container intervals = pV.second.data();

		// 2. On received intervals, dilate them by 2 in both directions
		LatticeSetByIntervals<Space>::Intervals::Container dilatedIntervals;
		for (const auto &interval: intervals) {
			dilatedIntervals.emplace_back(interval.first - 2, interval.second + 2);
		}
		LatticeSetByIntervals<Space>::Intervals::Container mergedIntervals;
		for (const auto &interval: dilatedIntervals) {
			if (mergedIntervals.empty() || mergedIntervals.back().second + 1 < interval.first) {
				mergedIntervals.push_back(interval);
			} else {
				mergedIntervals.back().second = interval.second;
			}
		}
		auto dilatedLatticeSet = LatticeSetByIntervals<Space>::Intervals();
		dilatedLatticeSet.data() = mergedIntervals;
		// 3. For each delta in {-2, ..., 2}^(d-1), add the intervals to point p + delta
		for (int i = 0; i < pow(5, dim - 1); i++) {
			Point delta;
			int idx = i;
			for (Dimension d = 0; d < dim; d++) {
				if (d == shapeLattice.axis()) {
					delta[d] = 0;
				} else {
					delta[d] = (idx % 5) - 2;
					idx /= 5;
				}
			}
			result.at(p + delta) = result.at(p + delta).set_union(dilatedLatticeSet);
		}
	}

	return result;
}

LatticeSetByIntervals<Space> getNStarOfShape(const int n, const int d) {
	auto shapeLattice = LatticeSetByIntervals<Space>(pointels.cbegin(), pointels.cend(), d).starOfPoints();
	for (auto i = 1; i < n; i++) {
		shapeLattice = getCoverOfShape(shapeLattice, shapeLattice.dimension);
	}
	return shapeLattice;
}

void computeVisibilityOmp(int radius, const Parameters &params = ComputeVisibilityParams()) {
	std::cout << "Computing visibility OMP" << std::endl;
	Dimension axis = getLargeAxis();
	auto latticeStart = getNStarOfShape(params["nStar"].as<int>(), axis);
	std::cout << "Lattice size: " << latticeStart.size() << std::endl;
	auto tmpFigLattices = latticeStart.data();
	std::map<Point, Intervals> figLattices;
	for (auto p: tmpFigLattices) {
		figLattices[p.first] = p.second.data();
	}
	const auto axises_idx = std::vector<Dimension>{axis, axis == 0 ? 1U : 0U, axis == 2 ? 1U : 2U};
	auto segmentList = getAllVectors(radius);

	// avoid thread imbalance
	std::shuffle(segmentList.begin(), segmentList.end(), std::mt19937(std::random_device()()));

	std::vector<Point> figLatticePoints;
	figLatticePoints.reserve(pointels.size() * 4);
	for (const auto &p: figLattices) {
		if (p.first[axises_idx[1]] % 2 == 0 && p.first[axises_idx[2]] % 2 == 0) {
			for (const auto &interval: p.second) {
				for (auto i = interval.first + 1; i <= interval.second; i += 2) {
					Point pt = p.first / 2;
					pt[axis] = i / 2;
					figLatticePoints.emplace_back(pt);
				}
			}
		}
	}
	visibility.reset(axis, segmentList, figLatticePoints);
	size_t chunkSize = 64;
	auto chunkAmount = segmentList.size() / chunkSize + 1;
	auto isSegmentListMultChunkSize = segmentList.size() % chunkSize == 0;
	std::cout << "Starting // OMP" << std::endl;
#pragma omp parallel for schedule(dynamic)
	for (auto chunkIdx = 0; chunkIdx < chunkAmount - isSegmentListMultChunkSize; chunkIdx++) {
		IntegerVector segment;
		Intervals eligibles;
		int minTx, maxTx, minTy, maxTy;
		for (auto segmentIdx = chunkIdx * chunkSize;
		     segmentIdx < std::min((chunkIdx + 1) * chunkSize, segmentList.size()); segmentIdx++) {
			segment = segmentList[segmentIdx];
			auto tmp = getLatticeVector(segment, axis).data();
			std::map<Point, Intervals> latticeVector;
			for (auto p: tmp) {
				latticeVector[p.first] = p.second.data();
			}
			minTx = digital_dimensions[axises_idx[1] + 3] - std::min(0, segment[axises_idx[1]]);
			maxTx = digital_dimensions[axises_idx[1] + 6] + 1 - std::max(0, segment[axises_idx[1]]);
			minTy = digital_dimensions[axises_idx[2] + 3] - std::min(0, segment[axises_idx[2]]);
			maxTy = digital_dimensions[axises_idx[2] + 6] + 1 - std::max(0, segment[axises_idx[2]]);
			for (auto tx = minTx; tx < maxTx; tx++) {
				for (auto ty = minTy; ty < maxTy; ty++) {
					eligibles.clear();
					eligibles.emplace_back(2 * digital_dimensions[axis + 3] - 1,
					                       2 * digital_dimensions[axis + 6] + 1);
					const Point pInterest(axis == 0 ? 0 : 2 * tx,
					                      axis == 1 ? 0 : 2 * (axis == 0 ? tx : ty),
					                      axis == 2 ? 0 : 2 * ty);
					for (const auto &cInfo: latticeVector) {
						const auto it = figLattices.find(pInterest + cInfo.first);
						if (it == figLattices.end()) {
							eligibles.clear();
							break;
						}
						eligibles = matchVector(eligibles, cInfo.second, it->second);
						if (eligibles.empty()) break;
					}
					if (!eligibles.empty()) {
						visibility.set(pInterest, eligibles, segmentIdx);
					}
				}
			}
		}
	}
	std::cout << "Visibility computed" << std::endl;
}


void computeVisibilityOmpGPU(int radius, const Parameters &params = ComputeVisibilityParams()) {
	std::cout << "Computing visibility OMP" << std::endl;
	Dimension axis = getLargeAxis();
	auto nStar = params["nStar"].as<int>();
	auto latticeStart = getNStarOfShape(nStar, axis);
	std::cout << "Lattice size: " << latticeStart.size() << std::endl;
	auto tmpFigLattices = latticeStart.data();
	std::map<Point, Intervals> figLattices;
	for (auto p: tmpFigLattices) {
		figLattices[p.first] = p.second.data();
	}
	const auto axises_idx = std::vector<Dimension>{axis, axis == 0 ? 1U : 0U, axis == 2 ? 1U : 2U};
	auto segmentList = getAllVectors(radius);

	// avoid thread imbalance
	std::shuffle(segmentList.begin(), segmentList.end(), std::mt19937(std::random_device()()));

	std::vector<Point> figLatticePoints;
	figLatticePoints.reserve(pointels.size() * 26);
	for (const auto &p: figLattices) {
		if (p.first[axises_idx[1]] % 2 == 0 && p.first[axises_idx[2]] % 2 == 0) {
			for (const auto &interval: p.second) {
				for (auto i = interval.first + 1; i <= interval.second; i += 2) {
					Point pt = p.first / 2;
					pt[axis] = i / 2;
					figLatticePoints.emplace_back(pt);
				}
			}
		}
	}
	visibility.reset(axis, segmentList, figLatticePoints);
	size_t chunkSize = 64;
	auto chunkAmount = segmentList.size() / chunkSize + 1;
	auto isSegmentListMultChunkSize = segmentList.size() % chunkSize == 0;
	std::cout << "Starting // OMP" << std::endl;
#pragma omp target teams distribute parallel for schedule(dynamic)
	for (auto chunkIdx = 0; chunkIdx < chunkAmount - isSegmentListMultChunkSize; chunkIdx++) {
		IntegerVector segment;
		Intervals eligibles;
		int minTx, maxTx, minTy, maxTy;
		for (auto segmentIdx = chunkIdx * chunkSize;
		     segmentIdx < std::min((chunkIdx + 1) * chunkSize, segmentList.size()); segmentIdx++) {
			segment = segmentList[segmentIdx];
			auto tmp = getLatticeVector(segment, axis).data();
			std::map<Point, Intervals> latticeVector;
			for (auto p: tmp) {
				latticeVector[p.first] = p.second.data();
			}
			minTx = digital_dimensions[axises_idx[1] + 3] - std::min(0, segment[axises_idx[1]]);
			maxTx = digital_dimensions[axises_idx[1] + 6] + 1 - std::max(0, segment[axises_idx[1]]);
			minTy = digital_dimensions[axises_idx[2] + 3] - std::min(0, segment[axises_idx[2]]);
			maxTy = digital_dimensions[axises_idx[2] + 6] + 1 - std::max(0, segment[axises_idx[2]]);
#pragma omp collapse(2)
			for (auto tx = minTx; tx < maxTx; tx++) {
				for (auto ty = minTy; ty < maxTy; ty++) {
					eligibles.clear();
					eligibles.emplace_back(2 * digital_dimensions[axis + 3] - 1,
					                       2 * digital_dimensions[axis + 6] + 1);
					const Point pInterest(axis == 0 ? 0 : 2 * tx,
					                      axis == 1 ? 0 : 2 * (axis == 0 ? tx : ty),
					                      axis == 2 ? 0 : 2 * ty);
					for (const auto &cInfo: latticeVector) {
						const auto it = figLattices.find(pInterest + cInfo.first);
						if (it == figLattices.end()) {
							eligibles.clear();
							break;
						}
						eligibles = matchVector(eligibles, cInfo.second, it->second);
						if (eligibles.empty()) break;
					}
					if (!eligibles.empty()) {
						visibility.set(pInterest, eligibles, segmentIdx);
					}
				}
			}
		}
	}
	std::cout << "Visibility computed" << std::endl;
}

void testNStarOfShape(int n) {
	auto shapeLattice = getNStarOfShape(n, getLargeAxis());
	std::cout << "Lattice size after " << n << " stars: " << shapeLattice.size() << std::endl;
	// show in polyscope
	auto points = shapeLattice.toPointRange(); /*
	std::vector<RealPoint> vp;
	embedPointels(points, vp);
	for (auto &p: vp) {
		p /= 2;
	}*/
	polyscope::registerPointCloud("NStar Shape Lattice", points);
}

/**
 * Compute the figure visibility using lattices
 * @param idx
 * @return
 */
void computeVisibility(int radius, const Parameters &params = ComputeVisibilityParams()) {
	std::cout << "Computing visibility" << std::endl;
	Dimension axis = getLargeAxis();
	auto nStar = params["nStar"].as<int>();
	auto latticeStart = getNStarOfShape(nStar, axis);
	std::cout << "Lattice size: " << latticeStart.size() << std::endl;
	auto tmpFigLattices = latticeStart.data();
	std::map<Point, Intervals> figLattices;
	for (auto p: tmpFigLattices) {
		figLattices[p.first] = p.second.data();
	}
	const auto axises_idx = std::vector<Dimension>{axis, axis == 0 ? 1U : 0U, axis == 2 ? 1U : 2U};
	IntegerVector segment;
	Intervals eligibles;
	auto segmentList = getAllVectors(radius);
	// I can't reset the visibility with pointels directly because i need to have the same order of points as in figLattices's closed points
	// visibility.reset(axis, segmentList, pointels);
	// So I create a vector of points from figLattices, which are all the points in the figure's n-star which have only even coordinates
	std::vector<Point> figLatticePoints;
	figLatticePoints.reserve(pointels.size() * 26);
	for (const auto &p: figLattices) {
		if (p.first[axises_idx[1]] % 2 == 0 && p.first[axises_idx[2]] % 2 == 0) {
			for (const auto &interval: p.second) {
				for (auto i = interval.first + 1; i <= interval.second; i += 2) {
					Point pt = p.first / 2;
					pt[axis] = i / 2;
					figLatticePoints.emplace_back(pt);
				}
			}
		}
	}
	visibility.reset(axis, segmentList, figLatticePoints);
	int minTx, maxTx, minTy, maxTy;
	std::vector<long> durations = std::vector<long>(segmentList.size() / 100, 0);
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	// Enumerate all pairs of vectorIdx, vector
	for (auto segmentIdx = 0; segmentIdx < segmentList.size(); segmentIdx++) {
		if (segmentIdx % 100 == 0) start = std::chrono::high_resolution_clock::now();
		segment = segmentList[segmentIdx];
		auto tmp = getLatticeVector(segment, axis).data();
		std::map<Point, Intervals> latticeVector;
		for (auto p: tmp) {
			latticeVector[p.first] = p.second.data();
		}
		minTx = digital_dimensions[axises_idx[1] + 3] - std::min(0, segment[axises_idx[1]]);
		maxTx = digital_dimensions[axises_idx[1] + 6] + 1 - std::max(0, segment[axises_idx[1]]);
		minTy = digital_dimensions[axises_idx[2] + 3] - std::min(0, segment[axises_idx[2]]);
		maxTy = digital_dimensions[axises_idx[2] + 6] + 1 - std::max(0, segment[axises_idx[2]]);
		for (auto tx = minTx; tx < maxTx; tx++) {
			for (auto ty = minTy; ty < maxTy; ty++) {
				eligibles.clear();
				eligibles.emplace_back(2 * digital_dimensions[axis + 3] - 1,
				                       2 * digital_dimensions[axis + 6] + 1);
				const Point pInterest(axis == 0 ? 0 : 2 * tx,
				                      axis == 1 ? 0 : 2 * (axis == 0 ? tx : ty),
				                      axis == 2 ? 0 : 2 * ty);
				for (const auto &cInfo: latticeVector) {
					// time this call
					eligibles = matchVector(eligibles, cInfo.second, figLattices[pInterest + cInfo.first]);
					if (eligibles.empty()) break;
				}
				if (!eligibles.empty()) {
					visibility.set(pInterest, eligibles, segmentIdx);
				}
			}
		}
		if (segmentIdx % 100 == 99) {
			end = std::chrono::high_resolution_clock::now();
			durations[segmentIdx / 100] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		}
	}
	std::ofstream durationsFile;
	durationsFile.open("durations.csv");
	durationsFile << "idx,duration" << std::endl;
	for (auto idx = 0; idx < durations.size(); idx++) {
		durationsFile << idx << "," << durations[idx] << std::endl;
	}
	durationsFile.close();
	std::cout << "Visibility computed" << std::endl;
}

// define << operator on file for visibility
std::ostream &operator<<(std::ostream &os, const Visibility &visib) {
	os << visib.mainAxis << std::endl;
	os << visib.vectorsSize << std::endl;
	os << visib.pointsSize << std::endl;

	for (const auto &v: visib.vectorIdxs) {
		os << v.first[0] << " " << v.first[1] << " " << v.first[2] << " " << v.second << " ";
	}
	os << std::endl;

	for (const auto &v: visib.pointIdxs) {
		os << v.first[0] << " " << v.first[1] << " " << v.first[2] << " " << v.second << " ";
	}
	os << std::endl;

	for (const auto &v: visib.visibles) {
		os << v << " ";
	}
	os << std::endl;
	return os;
}

void saveVisibility(const std::string &filename) {
	std::ofstream file;
	/*// print first infos of each variable to test
	for (int i = 0; i < 5 && i < visibility.vectorsSize; i++) {
		auto it = visibility.vectorIdxs.begin();
		std::advance(it, i);
		trace.info() << "Vector (" << it->first[0] << "," << it->first[1] << "," << it->first[2]
								 << ") has idx " << it->second << std::endl;
	}
	for (int i = 0; i < 5 && i < visibility.pointsSize; i++) {
			auto it = visibility.pointIdxs.begin();
			std::advance(it, i);
			trace.info() << "Point (" << it->first[0] << "," << it->first[1] << "," << it->first[2]
									 << ") has idx " << it->second << std::endl;
	}
	int start = 0;
	while (start < visibility.visibles.size() && !visibility.visibles[start]) start++;
	trace.info() << "First visible at index " << start << std::endl;
	for (int i = start; i < start+20 && i < visibility.visibles.size(); i++) {
			trace.info() << "Visible[" << i << "] = " << (visibility.visibles[i] ? "true" : "false") << std::endl;
	}
	trace.info() << "visibility size: " << visibility.visibles.size() << std::endl;
  */
	file.open(filename);
	file << visibility;
	file.close();
}

// define >> operator on file for visibility
std::istream &operator>>(std::istream &is, Visibility &visib) {
	Dimension mainAxis;
	size_t vectorsSize;
	size_t pointsSize;
	is >> mainAxis;
	is >> vectorsSize;
	is >> pointsSize;
	visib.vectorsSize = vectorsSize;
	visib.pointsSize = pointsSize;

	IntegerVectors vs(vectorsSize);
	std::vector<size_t> vectorIdxs(vectorsSize);
	std::vector<Point> ps(pointsSize);
	std::vector<size_t> pointIdxs(pointsSize);

	for (size_t i = 0; i < vectorsSize; i++) {
		IntegerVector v;
		int x, y, z;
		size_t idx;
		is >> x >> y >> z >> idx;
		vs[i] = IntegerVector(x, y, z);
		vectorIdxs[i] = idx;
	}
	for (size_t i = 0; i < pointsSize; i++) {
		Point p;
		int x, y, z;
		size_t idx;
		is >> x >> y >> z >> idx;
		ps[i] = Point(x, y, z);
		pointIdxs[i] = idx;
	}

	visib.reset(mainAxis, vs, ps);
	for (size_t i = 0; i < vectorsSize; i++) {
		visib.vectorIdxs[vs[i]] = vectorIdxs[i];
	}
	for (size_t i = 0; i < pointsSize; i++) {
		visib.pointIdxs[ps[i]] = pointIdxs[i];
	}

	visib.visibles = std::vector<bool>(vectorsSize * pointsSize, false);
	for (size_t i = 0; i < vectorsSize * pointsSize; i++) {
		bool v;
		is >> v;
		visib.visibles[i] = v;
	}
	return is;
}

void loadVisibility(const std::string &filename) {
	std::ifstream file;
	file.open(filename);
	// Print full path
	trace.info() << "Loading visibility from file: " << std::filesystem::absolute(filename) << std::endl;
	file >> visibility;
	/*
		// print first infos of each variable to test
		for (int i = 0; i < 5 && i < visibility.vectorsSize; i++) {
			auto it = visibility.vectorIdxs.begin();
			std::advance(it, i);
			trace.info() << "Vector (" << it->first[0] << "," << it->first[1] << "," << it->first[2]
									 << ") has idx " << it->second << std::endl;
		}
		for (int i = 0; i < 5 && i < visibility.pointsSize; i++) {
			auto it = visibility.pointIdxs.begin();
			std::advance(it, i);
			trace.info() << "Point (" << it->first[0] << "," << it->first[1] << "," << it->first[2]
									 << ") has idx " << it->second << std::endl;
		}
		int start = 0;
		while (start < visibility.visibles.size() && !visibility.visibles[start]) start++;
		trace.info() << "First visible at index " << start << std::endl;
		for (int i = start; i < start+20 && i < visibility.visibles.size(); i++) {
			trace.info() << "Visible[" << i << "] = " << (visibility.visibles[i] ? "true" : "false") << std::endl;
		}
		trace.info() << "visibility size: " << visibility.visibles.size() << std::endl;
		*/
	file.close();
}

void computeVisibilityWithPointShow(std::size_t idx) {
	auto segmentRadius = VisibilityRadius;
	auto sphereRadius = VisibilityRadius;
	if (visibility.empty()) computeVisibility(segmentRadius);
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
	trace.info() << "OMP parallelism " << (ok ? "(	OK	)" : "(ERROR)")
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

	unsigned seed = 0;
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

double sigma = -1;
int mitra_radius = (int) 4. / gridstep;
double minus2SigmaSquare = -2 * sigma * sigma;

double wSig(double d2) {
	return exp(d2 / minus2SigmaSquare);
}

double noWeight(double _) {
	return 1.0;
}

auto weighter = wSig;

void computeMitraNormals() {
	mitra_normals.clear();
	mitra_normals.reserve(pointels.size());
	auto kdTree = LinearKDTree<Point, 3>(pointels);
	for (const auto &pointel: pointels) {
		// 1. Get all the points in the ball of mitra_radius
		// 2. Compute the centroid of these points
		// 3. Compute the covariance matrix of these points
		std::vector<Point> neighbors;
		RealPoint centroid(0, 0, 0);
		for (auto point_idx: kdTree.pointsInBall(pointel, mitra_radius)) {
			auto tmp = kdTree.position(point_idx);
			neighbors.push_back(tmp);
			centroid += tmp;
		}
		centroid /= (double) neighbors.size();
		Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
		for (const auto &pt: neighbors) {
			auto diff = pt - centroid;
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					cov(i, j) += diff[i] * diff[j];
				}
			}
		}
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov / (double) neighbors.size());
		auto normalE = solver.eigenvectors().col(0);
		mitra_normals.emplace_back(normalE.x(), normalE.y(), normalE.z());
	}
	if (!noInterface) {
		psPrimalMesh->addVertexVectorQuantity("Pointel Mitra normals", mitra_normals);
	}
}

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
			if (visibility.isVisible(pointel, tmp)) {
				//	&& tmp != pointel) {
				visibles.push_back(tmp);
				//				centroid += tmp;
				const double w = weighter((pointel - tmp).squaredNorm());
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
					//					cov(i, j) += diff[i] * diff[j];
					cov(i, j) += diff[i] * diff[j] * weighter((pt - pointel).squaredNorm());
				}
			}
		}
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov / (double) visibles.size());
		auto normalE = solver.eigenvectors().col(0);
		visibility_normals.emplace_back(normalE.x(), normalE.y(), normalE.z());
	}
	if (!noInterface) {
		psPrimalMesh->addVertexVectorQuantity("Pointel visibility normals", visibility_normals);
	}
}

RealVector getTrivNormal(size_t idx) {
	return primal_surface->vertexNormal(idx);
}

void reorientVisibilityNormals() {
	if (visibility_normals.empty()) {
		computeVisibilityNormals();
	}
	//	if (visibility_sharps.empty()) {
	//		computeVisibilityDirectionToSharpFeatures();
	//	}
	for (int i = 0; i < visibility_normals.size(); ++i) {
		const auto triv_normal = getTrivNormal(i);
		if (visibility_normals[i].dot(triv_normal) < 0)
			visibility_normals[i] = -visibility_normals[i];
	}
	if (!mitra_normals.empty()) {
		for (int i = 0; i < mitra_normals.size(); ++i) {
			const auto triv_normal = getTrivNormal(i);
			if (mitra_normals[i].dot(triv_normal) < 0)
				mitra_normals[i] = -mitra_normals[i];
		}
	}
	if (!noInterface) {
		psPrimalMesh->addVertexVectorQuantity("Pointel visibility normals", visibility_normals);
		psPrimalMesh->addVertexVectorQuantity("Pointel trivial normal", primal_surface->vertexNormals());
		psPrimalMesh->addFaceVectorQuantity("Face trivial normal", primal_surface->faceNormals());
	}
}


void computeCurvaturesFor(std::vector<RealVector> &normals) {
	primal_surface->setVertexNormals(normals.cbegin(), normals.cend());
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

void doRedisplayNormalAsColorsAbsolute() {
	normalVisibilityColors.clear();
	normalIIColors.clear();
	normalVisibilityColors.reserve(visibility_normals.size());
	normalIIColors.reserve(visibility_normals.size());
	normalMitraColors.clear();
	normalMitraColors.reserve(mitra_normals.size());
	for (const auto &n: visibility_normals) {
		normalVisibilityColors.push_back(0.5f * (n + 1.0f));
	}
	for (const auto &n: ii_normals) {
		normalIIColors.push_back(0.5f * (n + 1.0f));
	}
	for (const auto &n: mitra_normals) {
		normalMitraColors.push_back(0.5f * (n + 1.0f));
	}
	psPrimalMesh->addVertexColorQuantity("Normals visibility as colors (absolute)", normalVisibilityColors);
	psPrimalMesh->addVertexColorQuantity("Normals II as colors (absolute)", normalIIColors);
	psPrimalMesh->addVertexColorQuantity("Normals Mitra as colors (absolute)", normalMitraColors);

	if (!true_normals.empty()) {
		normalTrueColors.clear();
		normalTrueColors.reserve(true_normals.size());
		for (const auto &n: true_normals) {
			normalTrueColors.push_back(0.5f * (n + 1.0f));
		}
		psPrimalMesh->addVertexColorQuantity("Normals true as colors (absolute)", normalTrueColors);
		psPrimalMesh->addVertexVectorQuantity("Normals true", true_normals);
	}
}

void doRedisplayNormalAsColorsRelativeFor(const std::vector<RealVector> &normals, std::string method) {
	glm::mat4 M = polyscope::view::getCameraViewMatrix();
	std::vector<glm::vec3> colors(normals.size());
	for (auto i = 0; i < normals.size(); i++) {
		const auto &n = normals[i];
		glm::vec4 wn(n[0], n[1], n[2], 0.0);
		glm::vec4 vn = M * wn;
		glm::vec3 c(vn[0] * 0.5 + 0.5, vn[1] * 0.5 + 0.5, fabs(vn[2]) * 0.5 + 0.5);
		colors[i] = c;
	}
	psPrimalMesh->addVertexColorQuantity("Normals " + method + " as colors (relative)", colors);
}

void doRedisplayNormalAsColorsRelative() {
	doRedisplayNormalAsColorsRelativeFor(visibility_normals, "visibility");
	doRedisplayNormalAsColorsRelativeFor(ii_normals, "II");
	doRedisplayNormalAsColorsRelativeFor(mitra_normals, "Mitra");
	if (!true_normals.empty()) {
		doRedisplayNormalAsColorsRelativeFor(true_normals, "true");
	}
}


auto olddgd = Polyhedra::digitization_gridstep_distance;

void computeTrueNormalsPolyhedra() {
	true_normals.clear();
	for (const auto &pointel: pointels) {
		true_normals.push_back(polyhedra->gradient((pointel - 0.5) * gridstep));
	}
	for (auto &n: true_normals) if (n.norm() != 0) n /= n.norm();
}

void computeL2looErrorsNormals() {
	if (olddgd != Polyhedra::digitization_gridstep_distance && Polyhedra::isPolyhedron(polynomial)) {
		std::cout << "Recomputing true normals with dgd " << Polyhedra::digitization_gridstep_distance << " (old was "
			<< olddgd << ")" << std::endl;
		computeTrueNormalsPolyhedra();
		olddgd = gridstep;
	}
	//	std::cout << "Computing L2 and Loo errors" << std::endl;
	std::vector<double> angle_dev(visibility_normals.size());
	std::vector<double> dummy(visibility_normals.size(), 0.0);
	for (int i = 0; i < visibility_normals.size(); ++i) {
		if (true_normals[i].norm() == 0) {
			angle_dev[i] = 0.0;
			continue;
		}
		const auto sp = visibility_normals[i].dot(true_normals[i]);
		const auto fxp = std::min(1.0, std::max(-1.0, sp));
		angle_dev[i] = acos(fxp);
	}
	if (!noInterface)
		psPrimalMesh->addVertexScalarQuantity("Angle deviation visib", angle_dev)->setMapRange({0.0, 0.1})->setColorMap(
			"reds");
	std::cout << "L2 error normals visib: " << SHG3::getScalarsNormL2(angle_dev, dummy) << std::endl;
	std::cout << "Loo error normals visib: " << SHG3::getScalarsNormLoo(angle_dev, dummy) << std::endl;

	angle_dev.clear();
	angle_dev.resize(ii_normals.size());
	for (int i = 0; i < ii_normals.size(); ++i) {
		if (true_normals[i].norm() == 0) {
			angle_dev[i] = 0.0;
			continue;
		}
		const auto sp = ii_normals[i].dot(true_normals[i]);
		const auto fxp = std::min(1.0, std::max(-1.0, sp));
		angle_dev[i] = acos(fxp);
	}
	if (!noInterface)
		psPrimalMesh->addVertexScalarQuantity("Angle deviation II", angle_dev)->setMapRange({0.0, 0.1})->setColorMap(
			"reds");
	std::cout << "L2 error normals II: " << SHG3::getScalarsNormL2(angle_dev, dummy) << std::endl;
	std::cout << "Loo error normals II: " << SHG3::getScalarsNormLoo(angle_dev, dummy) << std::endl;

	angle_dev.clear();
	angle_dev.resize(mitra_normals.size());
	for (int i = 0; i < mitra_normals.size(); ++i) {
		if (true_normals[i].norm() == 0) {
			angle_dev[i] = 0.0;
			continue;
		}
		const auto sp = mitra_normals[i].dot(true_normals[i]);
		const auto fxp = std::min(1.0, std::max(-1.0, sp));
		angle_dev[i] = acos(fxp);
	}
	if (!noInterface)
		psPrimalMesh->addVertexScalarQuantity("Angle deviation Mitra", angle_dev)->setMapRange({0.0, 0.1})->setColorMap(
			"reds");
	std::cout << "L2 error normals Mitra: " << SHG3::getScalarsNormL2(angle_dev, dummy) << std::endl;
	std::cout << "Loo error normals Mitra: " << SHG3::getScalarsNormLoo(angle_dev, dummy) << std::endl;
}

std::vector<double> operator-(const std::vector<double> &lhs, const std::vector<double> &rhs) {
	std::vector<double> res(lhs.size());
	for (size_t i = 0; i < lhs.size(); ++i) {
		res[i] = lhs[i] - rhs[i];
	}
	return res;
}

void computeL2looErrorsCurvaturesPolyhedron() {
	auto true_H = Polyhedra::getMeanCurvatureEstimation(polyhedra, K, surfels, SHG3::parametersShapeGeometry());
	auto true_G = Polyhedra::getGaussianCurvatureEstimation(polyhedra, K, surfels, SHG3::parametersShapeGeometry());
	auto true_K1 = Polyhedra::getFirstPrincipalCurvatureEstimation(polyhedra, K, surfels,
	                                                               SHG3::parametersShapeGeometry());
	auto true_K2 = Polyhedra::getSecondPrincipalCurvatureEstimation(polyhedra, K, surfels,
	                                                                SHG3::parametersShapeGeometry());

	// L2 and Loo errors
	for (const auto &[computedC, trueC]: {
		     std::pair(H, true_H), std::pair(G, true_G), std::pair(K1, true_K1), std::pair(K2, true_K2)
	     }) {
		// For each value pair in each pair, use it if true is not infinite
		std::vector<double> filteredComputedC;
		std::vector<double> filteredTrueC;
		for (size_t i = 0; i < computedC.size(); ++i) {
			if (std::isfinite(trueC[i])) {
				filteredComputedC.push_back(computedC[i]);
				filteredTrueC.push_back(trueC[i]);
			} else {
				filteredComputedC.push_back(0);
				filteredTrueC.push_back(0);
			}
		}
		for (auto f: {SHG3::getScalarsNormL2, SHG3::getScalarsNormLoo})
			std::cout << "L" << (f == SHG3::getScalarsNormL2 ? "2" : "oo")
				<< " error for " << (computedC == H ? "H" : computedC == G ? "G" : computedC == K1 ? "K1" : "K2")
				<< " curvature: "
				<< f(filteredComputedC, filteredTrueC) << std::endl;

		if (!noInterface) {
			psPrimalMesh->addFaceScalarQuantity(
					std::string("Error ") +
					(computedC == H ? "H" : computedC == G ? "G" : computedC == K1 ? "K1" : "K2") + " curvature",
					filteredComputedC - filteredTrueC)
				->setMapRange({0.0, MaxCurv})->setColorMap("reds");
		}
	}
}

void computeL2looErrorsCurvatures() {
	if (Polyhedra::isPolyhedron(polynomial)) {
		computeL2looErrorsCurvaturesPolyhedron();
		return;
	}
	auto true_H = SHG3::getMeanCurvatures(implicit_shape, K, surfels, SHG3::parametersShapeGeometry());
	auto true_G = SHG3::getGaussianCurvatures(implicit_shape, K, surfels, SHG3::parametersShapeGeometry());
	auto true_K1 = SHG3::getFirstPrincipalCurvatures(implicit_shape, K, surfels, SHG3::parametersShapeGeometry());
	auto true_K2 = SHG3::getSecondPrincipalCurvatures(implicit_shape, K, surfels, SHG3::parametersShapeGeometry());

	if (!noInterface) {
		psPrimalMesh->addFaceScalarQuantity("H true curvature", true_H)
			->setMapRange({-MaxCurv, MaxCurv})->setColorMap("coolwarm");
		psPrimalMesh->addFaceScalarQuantity("G true curvature", true_G)
			->setMapRange({-0.5 * MaxCurv * MaxCurv, 0.5 * MaxCurv * MaxCurv})->setColorMap("coolwarm");
		psPrimalMesh->addFaceScalarQuantity("K1 true curvature", true_K1)
			->setMapRange({-MaxCurv, MaxCurv})->setColorMap("coolwarm");
		psPrimalMesh->addFaceScalarQuantity("K2 true curvature", true_K2)
			->setMapRange({-MaxCurv, MaxCurv})->setColorMap("coolwarm");
	}

	for (const auto &[computedC, trueC]: {
		     std::pair(H, true_H), std::pair(G, true_G), std::pair(K1, true_K1), std::pair(K2, true_K2)
	     }) {
		for (auto f: {SHG3::getScalarsNormL2, SHG3::getScalarsNormLoo})
			std::cout << "L" << (f == SHG3::getScalarsNormL2 ? "2" : "oo")
				<< " error for " << (computedC == H ? "H" : computedC == G ? "G" : computedC == K1 ? "K1" : "K2")
				<< " curvature: "
				<< f(computedC, trueC) << std::endl;
		if (!noInterface) {
			psPrimalMesh->addFaceScalarQuantity(
					std::string("Error ") +
					(computedC == H ? "H" : computedC == G ? "G" : computedC == K1 ? "K1" : "K2") + " curvature",
					computedC - trueC)
				->setMapRange({0.0, MaxCurv})->setColorMap("reds");
		}
	}
}

#ifdef USE_CUDA_VISIBILITY

void computeVisibilityCuda(int radius) {
	std::cout << "Computing visibility CUDA" << std::endl;
	visibility = Visibility(computeVisibilityGpu(radius, digital_dimensions, pointels));
	std::cout << "Visibility computed" << std::endl;
}

void computeVisibilityCudaCPU(int radius) {
	std::cout << "Computing visibility CUDA" << std::endl;
	visibility = Visibility(computeVisibilityGpuCPU(radius, digital_dimensions, pointels));
	std::cout << "Visibility computed" << std::endl;
}

#endif

void testDistancePlan() {
	// For each plan of the implicit shape, compute using planeDistance the distance of each pointel to the plan and put the values in polyscope
	auto plans = polyhedra->getPlanes();
	std::vector<double> distances(pointels.size());
	for (size_t i = 0; i < plans.size(); ++i) {
		for (size_t j = 0; j < pointels.size(); ++j) {
			distances[j] = Polyhedra::planeDistance(plans[i], pointels[j], gridstep);
		}
		psPrimalMesh->addVertexScalarQuantity("Distances to plan " + std::to_string(i), distances)->setMapRange({
			-0.1, 0.1
		})->setColorMap("coolwarm");
	}
}

void testNearestPoint() {
	Parameters params = SH3::defaultParameters()
	                    | SHG3::defaultParameters()
	                    | SHG3::parametersGeometryEstimation();
	std::vector<RealPoint> points;
	// Compute the nearest point on polyhedra for each pointel of the shape
	int maxIter = params["projectionMaxIter"].as<int>();
	double accuracy = params["projectionAccuracy"].as<double>();
	double gamma = params["projectionGamma"].as<double>();
	/*for (const auto &pointel: pointels) {
		points.push_back(polyhedra->nearestPoint((pointel - 0.5) * gridstep, accuracy, maxIter, gamma));
	}*/
	// Display the points in polyscope
	// polyscope::registerPointCloud("Nearest points", points);
	// for all points, compute the gradient
	std::vector<RealVector> gradients;
	for (const auto &pointel: pointels) {
		gradients.push_back(polyhedra->gradient((pointel - 0.5) * gridstep));
	}
	// Display the gradients as colors
	std::vector<glm::vec3> colors;
	for (const auto &grad: gradients) {
		colors.push_back(glm::vec3(0.5f * (grad[0] + 1.0f), 0.5f * (grad[1] + 1.0f), 0.5f * (grad[2] + 1.0f)));
	}
	psPrimalMesh->addVertexColorQuantity("Gradients nearest points", colors)->setEnabled(true);
	psPrimalMesh->addVertexVectorQuantity("Gradients nearest points vectors", gradients);
}

void myCallback() {
	std::string visibilityFilename;
	// Select a vertex with the mouse
	if (polyscope::haveSelection()) {
		auto selection = polyscope::getSelection();
		auto selectedSurface = static_cast<polyscope::SurfaceMesh *>(selection.structure);
		auto idx = selection.localIndex;

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
	ImGui::SliderInt("mitra radius", &mitra_radius, 1, 20);
	if (ImGui::Button("Visibility")) {
		computeVisibilityWithPointShow(pointel_idx);
	}
	if (ImGui::Button("Visibilities")) {
		trace.beginBlock("Compute visibilities");
		computeVisibility(VisibilityRadius, ComputeVisibilityParams()("nStar", nStarTest));
		Time = trace.endBlock();
	}
	ImGui::SameLine();
	if (ImGui::Button("Visibilities OMP")) {
		trace.beginBlock("Compute visibilities OMP (nStar not taken into account and may fail)");
		computeVisibilityOmp(VisibilityRadius);
		Time = trace.endBlock();
	}
	ImGui::SameLine();
	if (ImGui::Button("Visibilities OMP GPU")) {
		trace.beginBlock("Compute visibilities OMP GPU");
		computeVisibilityOmpGPU(VisibilityRadius, ComputeVisibilityParams()("nStar", nStarTest));
		Time = trace.endBlock();
	}
	if (ImGui::Button("Measure Mean Distance Visibility")) {
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
		Time = trace.endBlock();
		computeMitraNormals();
		reorientVisibilityNormals();
		doRedisplayNormalAsColorsAbsolute();
		doRedisplayNormalAsColorsRelative();
	}
	ImGui::SameLine();
	if (ImGui::Button("Compute normal errors")) {
		trace.beginBlock("Compute visibilities Normals Errors");
		computeL2looErrorsNormals();
		Time = trace.endBlock();
	}
	const char *items[] = {"Visibility normals", "Mitra normals", "II normals", "CTriv"};
	static int item_current = 0;
	ImGui::Combo("Normals to use for curv.", &item_current, items, IM_ARRAYSIZE(items));
	if (ImGui::Button("Compute Curvatures")) {
		trace.beginBlock("Compute visibilities Curvatures");
		if (item_current == 0) {
			computeCurvaturesFor(visibility_normals);
		} else if (item_current == 1) {
			computeCurvaturesFor(mitra_normals);
		} else if (item_current == 2) {
			computeCurvaturesFor(ii_normals);
		} else {
			computeCurvaturesFor(trivial_normals);
		}
		Time = trace.endBlock();
		doRedisplayCurvatures();
	}
	ImGui::SameLine();
	if (ImGui::Button("Compute curvature errors")) {
		trace.beginBlock("Compute visibilities Curvature Errors");
		computeL2looErrorsCurvatures();
		Time = trace.endBlock();
	}
#ifdef USE_CUDA_VISIBILITY
	if (ImGui::Button("Compute CUDA visibilities")) {
		trace.beginBlock("Compute visibilities CUDA");
		computeVisibilityCuda(VisibilityRadius);
		Time = trace.endBlock();
	}
	ImGui::SameLine();
	if (ImGui::Button("Compute CUDA CPU vers. visibilities")) {
		trace.beginBlock("Compute visibilities CUDA CPU Vers.");
		computeVisibilityCudaCPU(VisibilityRadius);
		Time = trace.endBlock();
	}
#endif
	ImGui::InputText("Visibility file", inputVisibilityFilename, 256);
	if (ImGui::Button("Save Visibility")) {
		visibilityFilename = std::string(inputVisibilityFilename);
		saveVisibility(visibilityFilename);
	}
	ImGui::SameLine();
	if (ImGui::Button("Load Visibility")) {
		visibilityFilename = std::string(inputVisibilityFilename);
		loadVisibility(visibilityFilename);
	}
	ImGui::SameLine();
	ImGui::Text("nb threads = %d", OMP_max_nb_threads);
	if (ImGui::Button("TEST DISTANCE PLAN")) {
		testDistancePlan();
	}
	ImGui::InputInt("nstar", &nStarTest, 1);
	ImGui::SameLine();
	if (ImGui::Button("testNStarOfShape")) {
		testNStarOfShape(nStarTest);
	}
	// Add a selector for the weighting function used for normal estimation
	const char *weightSelectorItems[] = {"Gaussian", "No Weight"};
	static int weightCurrentItem = 0;
	ImGui::Combo("Weighter", &weightCurrentItem, weightSelectorItems, IM_ARRAYSIZE(weightSelectorItems));
	if (weightCurrentItem == 0) {
		weighter = wSig;
	} else {
		weighter = noWeight;
	}
	// Add a slider for Polyhedra::digitization_gridstep_distance (default 0.2, can vary from 0.05 to 1)
	ImGui::SliderFloat("Gridstep distance for plane distance", &Polyhedra::digitization_gridstep_distance, 0.05f, 1.0f);

	if (ImGui::Button("TEST NEAREST POINT")) {
		testNearestPoint();
	}

	ImGui::InputFloat("MinMaxCurv", &MaxCurv, 0.01f, 1.0f);
	// Shortcuts
	if (ImGui::IsKeyPressed(ImGuiKey_N)) {
		doRedisplayNormalAsColorsRelative();
	}
}

void testIntersection() {
	// 1|2 4|5 7|9
	for (auto v: intersect(Intervals({Interval(0, 9)}), Intervals({Interval(1, 2), Interval(4, 5), Interval(7, 10)}))) {
		std::cout << v << " ";
	}
	std::cout << std::endl;

	// 2|3 5|6 8|10
	for (auto v: intersect(Intervals({Interval(0, 3), Interval(5, 10)}),
	                       Intervals({Interval(2, 6), Interval(8, 12)}))) {
		std::cout << v << " ";
	}
	std::cout << std::endl;

	// 1|3 6|7 9|10 12|12 14|15
	for (auto v: intersect(Intervals({Interval(0, 3), Interval(5, 10), Interval(12, 15)}),
	                       Intervals({Interval(1, 3), Interval(6, 7), Interval(9, 12), Interval(14, 15)}))) {
		std::cout << v << " ";
	}
	std::cout << std::endl;
}

void TMP(std::string filename) {
	auto params = SH3::defaultParameters()
	              | SHG3::defaultParameters()
	              | SHG3::parametersGeometryEstimation();
	const double h = 0.25;
	CountedPtr<SH3::BinaryImage> binary_image;
	SH3::KSpace K;
	//	if (filename == "") {
	//		params("polynomial", "goursat")("gridstep", h);
	//		auto implicit_shape = SH3::makeImplicitShape3D(params);
	//		auto digitized_shape = SH3::makeDigitizedImplicitShape3D(implicit_shape, params);
	//		binary_image = SH3::makeBinaryImage(digitized_shape, params);
	//		K = SH3::getKSpace(params);
	//	} else {
	int thresholdMin = 0;
	int thresholdMax = 255;
	params("thresholdMin", thresholdMin);
	params("thresholdMax", thresholdMax);

	binary_image = SH3::makeBinaryImage(filename, params);

	K = SH3::getKSpace(binary_image, params);
	//	}
	auto embedder = SH3::getCellEmbedder(K);
	auto surface = SH3::makeLightDigitalSurface(binary_image, K, params);
	SH3::Cell2Index c2i;
	auto pointels = SH3::getPointelRange(c2i, surface);
	auto primal_surface = SH3::makePrimalSurfaceMesh(surface);
	SH3::RealPoints pos(pointels.size());
	std::transform(pointels.cbegin(), pointels.cend(), pos.begin(),
	               [&](const SH3::Cell &c) { return h * embedder(c); });
	//	auto ppos		 = SHG3::getPositions( implicit_shape, pos, params );
	// init primal_faces
	std::vector<std::vector<std::size_t> > primal_faces;
	for (auto face = 0; face < primal_surface->nbFaces(); ++face)
		primal_faces.push_back(primal_surface->incidentVertices(face));

	polyscope::init();
	psPrimalMesh = polyscope::registerSurfaceMesh("Primal surface",
	                                              pos,
	                                              primal_faces);
	psPrimalMesh->setSurfaceColor(glm::vec3(0.50, 0.50, 0.75));
	psPrimalMesh->setEdgeWidth(1.5);
	polyscope::show();
}

int gpuRun(int argc, char *argv[]) {
	noInterface = false;

	// command line inteface options
	CLI::App app{"A tool to check visibility using lattices."};
	std::string filename = "../volumes/bunny34.vol";
	int thresholdMin = 0;
	int thresholdMax = 255;
	int minAABB = -10;
	int maxAABB = 10;
	bool listP = false;
	bool computeCurvaturesFlag = false;
	bool computeNormalsFlag = false;
	bool noFurtherComputation = false;
	double sigmaTmp = -1.0;
	int VisibilityRadiusTmp = -1;
	int nstar = 1;
	std::string saveVisibilityFilename;
	std::string saveShapeFilename = "shape.vol";
	std::string visibComputeMethod = "OMP_GPU";
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
	app.add_option("-r,--radius", VisibilityRadiusTmp, "the radius of the visibility sphere");
	app.add_flag("-l", listP, "lists the known named polynomials.");
	app.add_option("-s,--sigma", sigmaTmp, "sigma used for visib normal computation");
	app.add_flag("--gpuRun", noInterface, "only work as the job asked for gpuRun function");

	app.add_flag("--computeNormals", computeNormalsFlag, "compute visibility normals after visibility computation");
	app.add_flag("--computeCurvatures", computeCurvaturesFlag,
	             "compute curvatures after visibility normals computation");
	app.add_option("--saveShapeFilename", saveShapeFilename,
	               "a flag to save the primal surface shape in an OBJ file after computation");
	app.add_option("--saveVisibility", saveVisibilityFilename, "a filename to save the computed visibility");
	app.add_option("--visibComputeMethod", visibComputeMethod,
	               "method to compute visibility: 'CPU', 'OMP', 'OMP_GPU', 'GPU' (default 'OMP_GPU')");
	app.add_flag("--noFurtherComputation", noFurtherComputation,
	             "if set, no computation after generating the shape");
	app.add_option("--nstar", nstar,
	               "number of stars to use in visibility computation (default 1, will do every star from 1 to nstar)");
	app.add_option("--mitraRadius", mitra_radius, "radius used for Mitra normal computation (default 4/gridstep)");

	CLI11_PARSE(app, argc, argv)
	if (listP) {
		listPolynomials();
		return 0;
	}

	if (computeCurvaturesFlag && !computeNormalsFlag) {
		std::cerr << "Error: to compute curvatures, normals must be computed first." << std::endl;
		return 1;
	}

	auto params = SH3::defaultParameters()
	              | SHG3::defaultParameters()
	              | SHG3::parametersGeometryEstimation();
	params("surfaceComponents", "All")("surfelAdjacency", 0); //exterior adjacency
	params("surfaceTraversal", "default");
	bool is_polynomial = !polynomial.empty();
	if (is_polynomial) {
		trace.beginBlock("Build polynomial surface");
		if (!Polyhedra::isPolyhedron(polynomial)) {
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
		} else {
			std::cout << "Using polyhedron: " << polynomial << std::endl;
			polyhedra = Polyhedra::makeImplicitPolyhedron(polynomial, gridstep);
			binary_image = Polyhedra::makeBinaryPolyhedron(polyhedra, gridstep, minAABB, maxAABB);
			std::cout << "Digitization done." << std::endl;
			K = SH3::getKSpace(binary_image, params);
			std::cout << "KSpace done." << std::endl;
		}
		trace.endBlock();
	} else {
		trace.beginBlock("Reading image vol file");
		params("thresholdMin", thresholdMin);
		params("thresholdMax", thresholdMax);
		binary_image = SH3::makeBinaryImage(filename, params);
		K = SH3::getKSpace(binary_image, params);
		trace.endBlock();
	}

	if (!saveShapeFilename.empty()) {
		SH3::saveBinaryImage(binary_image, saveShapeFilename);
	}
	if (noFurtherComputation) {
		return 0;
	}

	std::vector<std::vector<std::size_t> > primal_faces;
	std::vector<RealPoint> primal_positions;


	trace.beginBlock("Computing digital points and primal surface");
	// Build digital surface
	digital_surface = SH3::makeDigitalSurface(binary_image, K, params);
	primal_surface = SH3::makePrimalSurfaceMesh(digital_surface);
	surfels = SH3::getSurfelRange(digital_surface, params);
	if (Polyhedra::isPolyhedron(polynomial)) {
		surfel_true_normals = Polyhedra::getNormalVectors(polyhedra, K, surfels, params);
	} else if (is_polynomial) {
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

	if (computeNormalsFlag) {
		// Initialize normals with trivial normals
		trace.beginBlock("Compute trivial normals");
		auto pTC = new TangencyComputer<KSpace>(K);
		pTC->init(pointels.cbegin(), pointels.cend(), true);
		int t_ring = int(round(params["t-ring"].as<double>()));
		auto surfel_trivial_normals = SHG3::getTrivialNormalVectors(K, surfels);
		primal_surface->faceNormals() = surfel_trivial_normals;
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
				if (is_polynomial && !Polyhedra::isPolyhedron(polynomial)) {
					true_normals[idx] += surfel_true_normals[k];
				}
			}
		}
		for (auto &n: trivial_normals) n /= n.norm();
		for (auto &n: ii_normals) n /= n.norm();
		if (is_polynomial && !Polyhedra::isPolyhedron(polynomial)) {
			for (auto &n: true_normals)
				if (n.norm() != 0) n /= n.norm();
		} else {
			computeTrueNormalsPolyhedra();
		}
		computeMitraNormals();
		primal_surface->vertexNormals() = trivial_normals;
		trace.endBlock();

		if (computeCurvaturesFlag) {
			trace.beginBlock("Initialize CNC for curvature computation");
			pCNC = CountedPtr<CNC>(new CNC(*primal_surface));
			trace.endBlock();
		}
	}


	if (sigmaTmp != -1.0) {
		sigma = sigmaTmp;
	} else {
		sigma = 5 * pow(gridstep, -0.5);
	}
	minus2SigmaSquare = -2 * sigma * sigma;
	if (VisibilityRadiusTmp > 0) {
		VisibilityRadius = VisibilityRadiusTmp;
	} else {
		VisibilityRadius = 2 * static_cast<int>(sigma);
	}
	std::cout << "sigma = " << sigma << std::endl;


	// Ready to choose program
	for (int star = 1; star <= nstar; ++star) {
		std::cout << "GPU run for nStar = " << star << std::endl;
		auto params = ComputeVisibilityParams()("nStar", star);
		trace.beginBlock("Compute visibilities for nStar=" + std::to_string(star));
		if (visibComputeMethod == "CPU") {
			computeVisibility(VisibilityRadius, params);
		} else if (visibComputeMethod == "OMP") {
			computeVisibilityOmp(VisibilityRadius);
		} else if (visibComputeMethod == "OMP_GPU") {
			computeVisibilityOmpGPU(VisibilityRadius, params);
		} else if (visibComputeMethod == "GPU") {
#ifdef USE_CUDA_VISIBILITY
			computeVisibilityCuda(VisibilityRadius);
#else
			std::cerr << "Error: CUDA visibility computation not available. Recompile with CUDA support." << std::endl;
			return 1;
#endif
		} else {
			std::cerr << "Error: unknown visibility computation method '" << visibComputeMethod << "'." << std::endl;
			return 1;
		}
		Time = trace.endBlock();
		computeMeanDistanceVisibility();
		if (computeNormalsFlag) {
			trace.beginBlock("Compute visibilities Normals");
			computeVisibilityNormals();
			computeMitraNormals();
			reorientVisibilityNormals();
			Time = trace.endBlock();
			if (is_polynomial) {
				computeL2looErrorsNormals();
			}
		}
		if (computeCurvaturesFlag) {
			trace.beginBlock("Compute visibilities Curvatures");
			computeCurvaturesFor(visibility_normals);
			Time = trace.endBlock();
			if (is_polynomial) {
				computeL2looErrorsCurvatures();
			}
			trace.beginBlock("Compute mitra curvatures");
			computeCurvaturesFor(mitra_normals);
			Time = trace.endBlock();
			if (is_polynomial) {
				computeL2looErrorsCurvatures();
			}
			trace.beginBlock("Compute ii curvatures");
			computeCurvaturesFor(ii_normals);
			Time = trace.endBlock();
			if (is_polynomial) {
				computeL2looErrorsCurvatures();
			}
		}
		/*if (!saveVisibilityFilename.empty()) {
			trace.beginBlock("Save visibility");
			saveVisibility(saveVisibilityFilename);
			trace.endBlock();
		}*/
		std::cout << "GPU run completed successfully." << std::endl;
	}
	return 0;
}

int main(int argc, char *argv[]) {
	// command line inteface options
	CLI::App app{"A tool to check visibility using lattices."};
	std::string filename = "../volumes/bunny34.vol";
	int thresholdMin = 0;
	int thresholdMax = 255;
	int minAABB = -10;
	int maxAABB = 10;
	int VisibilityRadiusTmp = -1;
	bool listP = false;
	bool gpuRunFlag = false;
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
	app.add_option("-r,--radius", VisibilityRadiusTmp, "the radius of the visibility sphere");
	app.add_flag("-l", listP, "lists the known named polynomials.");
	app.add_flag("--noInterface", noInterface, "desactivate the interface and use the visibility OMP algorithm");
	app.add_option("--IIradius", iiRadius, "radius used for ii normal computation");
	double sigmaTmp = -1.0;
	app.add_option("-s,--sigma", sigmaTmp, "sigma used for visib normal computation");
	app.add_flag("--gpuRun", gpuRunFlag, "only work as the job asked for gpuRun function");

	bool ignore = false;
	std::string _;

	// gpuRun only
	app.add_flag("--computeNormals", ignore, "gpuRun only : compute visibility normals after visibility computation");
	app.add_flag("--computeCurvatures", ignore,
	             "gpuRun only : compute curvatures after visibility normals computation");
	app.add_option("--saveShapeFilename", _,
	               "gpuRun only : a flag to save the primal surface shape in an OBJ file after computation");
	app.add_option("--saveVisibility", _, "gpuRun only : a filename to save the computed visibility");
	app.add_option("--visibComputeMethod", _,
	               "gpuRun only : method to compute visibility: 'CPU', 'OMP', 'OMP_GPU', 'GPU' (default 'OMP_GPU')");
	app.add_flag("--noFurtherComputation", ignore,
	             "gpuRun only : if set, no computation after generating the shape");
	app.add_option("--nstar", ignore, "number of stars to use in visibility computation (default 1)");
	app.add_option("--mitraRadius", mitra_radius, "radius used for Mitra normal computation (default 4/gridstep)");

	// -p "x^2+y^2+2*z^2-x*y*z+z^3-100" -g 0.5
	// Parse command line options. Exit on error.
	CLI11_PARSE(app, argc, argv)

	if (gpuRunFlag) {
		return gpuRun(argc, argv);
	}

	// React to some options.
	if (listP) {
		listPolynomials();
		return 0;
	}
	//	if (polynomial != "") {
	//		TMP("");
	//	} else {
	//		TMP(filename);
	//	}
	//	return 0;
	// Use options
	auto params = SH3::defaultParameters()
	              | SHG3::defaultParameters()
	              | SHG3::parametersGeometryEstimation();
	params("surfaceComponents", "All")("surfelAdjacency", 0); //exterior adjacency
	params("surfaceTraversal", "default");
	bool is_polynomial = !polynomial.empty();
	if (is_polynomial) {
		trace.beginBlock("Build polynomial surface");
		if (!Polyhedra::isPolyhedron(polynomial)) {
			params("gridstep", gridstep);
			params("minAABB", minAABB);
			params("maxAABB", maxAABB);
			params("polynomial", polynomial);
			params("offset", 1.0);
			params("closed", 1);
			implicit_shape = SH3::makeImplicitShape3D(params);
			auto digitized_shape = SH3::makeDigitizedImplicitShape3D(implicit_shape, params);
			K = SH3::getKSpace(params);
			binary_image = SH3::makeBinaryImage(digitized_shape,
			                                    SH3::Domain(K.lowerBound(),
			                                                K.upperBound()),
			                                    params);
		} else {
			std::cout << "Using polyhedron: " << polynomial << std::endl;
			// polyhedra = CountedPtr(new Polyhedra::PolyhedronShape(std::vector<Polyhedra::Plane>(), {
			// 	                                                      Polyhedra::Ellipsoid{
			// 		                                                      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, 1,
			// 		                                                      5, 1
			// 	                                                      }
			//                                                       }, gridstep));
			polyhedra = Polyhedra::makeImplicitPolyhedron(polynomial, gridstep);
			binary_image = Polyhedra::makeBinaryPolyhedron(polyhedra, gridstep, minAABB, maxAABB);
			std::cout << "Digitization done." << std::endl;
			K = SH3::getKSpace(binary_image, params);
			std::cout << "KSpace done." << std::endl;
		}
		trace.endBlock();
	} else {
		trace.beginBlock("Reading image vol file");
		params("thresholdMin", thresholdMin);
		params("thresholdMax", thresholdMax);
		binary_image = SH3::makeBinaryImage(filename, params);
		K = SH3::getKSpace(binary_image, params);
		trace.endBlock();
	}

	std::vector<std::vector<std::size_t> > primal_faces;
	std::vector<RealPoint> primal_positions;


	trace.beginBlock("Computing digital points and primal surface");
	// Build digital surface
	digital_surface = SH3::makeDigitalSurface(binary_image, K, params);
	primal_surface = SH3::makePrimalSurfaceMesh(digital_surface);
	surfels = SH3::getSurfelRange(digital_surface, params);
	if (Polyhedra::isPolyhedron(polynomial)) {
		surfel_true_normals = Polyhedra::getNormalVectors(polyhedra, K, surfels, params);
	} else if (is_polynomial) {
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
			if (is_polynomial && !Polyhedra::isPolyhedron(polynomial)) {
				true_normals[idx] += surfel_true_normals[k];
			}
		}
	}
	for (auto &n: trivial_normals) n /= n.norm();
	for (auto &n: ii_normals) n /= n.norm();
	if (is_polynomial && !Polyhedra::isPolyhedron(polynomial)) {
		for (auto &n: true_normals)
			if (n.norm() != 0) n /= n.norm();
	} else if (Polyhedra::isPolyhedron(polynomial)) {
		computeTrueNormalsPolyhedra();
	}
	primal_surface->vertexNormals() = trivial_normals;

	if (sigmaTmp != -1.0) {
		sigma = sigmaTmp;
	} else {
		sigma = 5 * pow(gridstep, -0.5);
	}
	if (VisibilityRadiusTmp > 0) {
		VisibilityRadius = VisibilityRadiusTmp;
	} else {
		VisibilityRadius = 2 * static_cast<int>(sigma);
	}
	minus2SigmaSquare = -2 * sigma * sigma;

	std::cout << "sigma = " << sigma << std::endl;


	pCNC = CountedPtr<CNC>(new CNC(*primal_surface));

	// Initialize polyscope
	if (noInterface) {
		std::cout << "sigma = " << sigma << std::endl;
		checkParallelism();
		trace.beginBlock("Compute visibilities");
		computeVisibilityOmp(VisibilityRadius);
		Time = trace.endBlock();
		//		trace.beginBlock("Compute mean distance visibility");
		//		computeMeanDistanceVisibility();
		//		Time = trace.endBlock();
	} else {
		polyscope::init();
		psPrimalMesh = polyscope::registerSurfaceMesh("Primal surface",
		                                              primal_positions,
		                                              primal_faces);
		psPrimalMesh->setSurfaceColor(glm::vec3(0.50, 0.50, 0.75));
		psPrimalMesh->setEdgeWidth(1.5);
		polyscope::state::userCallback = myCallback;
		polyscope::show();
	}
	return EXIT_SUCCESS;
}
