#include "polyhedra.h"
#include <cmath>
#include <iostream>

namespace Polyhedra {
	using namespace DGtal;
	using namespace Z3i;
	using Scalar = RealVector::Component;

	bool isPolyhedron(const std::string &shape) {
		static const std::vector<std::string> names =
			{"cube", "tetrahedron", "triangular_pyramid", "dodecahedron", "icosahedron"};
		return std::ranges::any_of(names, [&](const std::string &n) { return n == shape; });
	}

	void listPolyhedron() {
		std::cout << "Available polyhedra:\n";
		std::cout << " - cube\n";
		std::cout << " - tetrahedron (alias: triangular_pyramid)\n";
		std::cout << " - dodecahedron\n";
		std::cout << " - icosahedron\n";
	}

	static RealPoint normalize(const RealPoint &v) {
		double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
		return (n == 0.0) ? v : v / n;
	}

	class PolyhedronShape {
	public:
		using Space = Z3i::Space;
		using RealPoint = Space::RealPoint;
		using RealVector = Space::RealVector;
	private:
		std::vector<std::pair<RealPoint, double>> myPlanes;

	public:
		PolyhedronShape(const std::vector<std::pair<RealPoint, double>> &planes)
			: myPlanes(planes) {}

		DGtal::Orientation orientation(const RealPoint &p) const {
			std::vector<DGtal::Orientation> orientations(myPlanes.size(), DGtal::INSIDE);
			int i = 0;
			for (auto &pl: myPlanes) {
				double dot = pl.first.dot(p);
				if (dot >= pl.second)
					return DGtal::OUTSIDE;
				else if (dot == pl.second)
					orientations[i] = DGtal::ON;
				i++;

			}
			if (std::ranges::any_of(orientations, [](DGtal::Orientation o) { return o == DGtal::ON; }))
				return DGtal::ON;
			else
				return DGtal::INSIDE;
		}

		RealPoint nearestPoint(RealPoint x,
		                       double accuracy,
		                       int maxIter,
		                       double gamma) const {
			for (int iter = 0; iter < maxIter; ++iter) {
				RealPoint totalCorrection = RealPoint(0, 0, 0);

				for (const auto &pl: myPlanes) {
					const RealPoint &n = pl.first;
					double d = pl.second;

					double dist = n.dot(x) - d*d;
					if (dist > 0) // outside half-space → project
						totalCorrection -= gamma * dist * n;
				}

				if (totalCorrection.norm() < accuracy)
					break; // converged

				x += totalCorrection;
			}
			return x;
		}

		RealVector gradient(const RealPoint &p) const {
			auto np = nearestPoint(p, 1e-10, 100, 0.5);
			// First compute the nearest plane(s) to p
			double maxDist = -std::numeric_limits<double>::infinity();
			for (const auto &pl: myPlanes) {
				double dist = pl.first.dot(np) - pl.second*pl.second;
				if (dist > maxDist)
					maxDist = dist;
			}
			// Then sum the normals of all planes at this distance
			RealVector grad(0, 0, 0);
			for (const auto &pl: myPlanes) {
				double dist = pl.first.dot(np) - pl.second*pl.second;
				if (std::abs(dist - maxDist) < 1e-10) // consider numerical precision
					grad += pl.first;
			}
			return normalize(grad);
		}

	};

	CountedPtr<PolyhedronShape> makeImplicitPolyhedron(const std::string &shape) {
		std::vector<std::pair<RealPoint, double>> planes;

		if (shape == "cube") {
			double d = 5.0;
			planes = {
				{{1,  0,  0},  d},
				{{-1, 0,  0},  d},
				{{0,  1,  0},  d},
				{{0,  -1, 0},  d},
				{{0,  0,  1},  d},
				{{0,  0,  -1}, d}
			};
		} else if (shape == "tetrahedron" || shape == "triangular_pyramid") {
			// Regular tetrahedron with base normal (0,-1,0)
			double d = 3;

			std::vector<RealPoint> normals = {
				{0.0,     -1.0,   0.0},               // base
				{0.9428,  0.3333, 0.0000},      // side 1
				{-0.4714, 0.3333, 0.8165},      // side 2
				{-0.4714, 0.3333, -0.8165}       // side 3
			};

			for (const auto &n: normals)
				planes.emplace_back(n, d);
		} else if (shape == "dodecahedron") {
			double phi = (1.0 + std::sqrt(5.0)) / 2.0;
			std::vector<RealPoint> normals = {
				{0,    1,    phi},
				{0,    -1,   phi},
				{0,    1,    -phi},
				{0,    -1,   -phi},
				{1,    phi,  0},
				{-1,   phi,  0},
				{1,    -phi, 0},
				{-1,   -phi, 0},
				{phi,  0,    1},
				{-phi, 0,    1},
				{phi,  0,    -1},
				{-phi, 0,    -1}
			};
			double d = 3;
			for (const auto &n: normals) planes.emplace_back(normalize(n), d);
		} else if (shape == "icosahedron") {
			double phi = (1.0 + std::sqrt(5.0)) / 2.0;
			std::vector<RealPoint> normals = {
				{0,    1,    phi},
				{0,    -1,   phi},
				{0,    1,    -phi},
				{0,    -1,   -phi},
				{1,    phi,  0},
				{-1,   phi,  0},
				{1,    -phi, 0},
				{-1,   -phi, 0},
				{phi,  0,    1},
				{-phi, 0,    1},
				{phi,  0,    -1},
				{-phi, 0,    -1},
				{1,    1,    phi},
				{-1,   1,    phi},
				{1,    -1,   phi},
				{-1,   -1,   phi},
				{1,    1,    -phi},
				{-1,   1,    -phi},
				{1,    -1,   -phi},
				{-1,   -1,   -phi}
			};
			double d = 3;
			for (const auto &n: normals) planes.emplace_back(normalize(n), d);
		} else {
			trace.error() << "[makeImplicitPolyhedron] Unknown shape: " << shape << std::endl;
		}

		return CountedPtr<PolyhedronShape>(new PolyhedronShape(planes));
	}

	CountedPtr<SH3::BinaryImage> makeBinaryPolyhedron(const std::string &shape,
	                                                  double gridstep,
	                                                  double minAABB,
	                                                  double maxAABB) {
		auto implicitShape = makeImplicitPolyhedron(shape);
		return makeBinaryPolyhedron(implicitShape, gridstep, minAABB, maxAABB);
	}

	CountedPtr<SH3::BinaryImage> makeBinaryPolyhedron(const CountedPtr<PolyhedronShape> &implicitShape,
	                                                  double gridstep,
	                                                  double minAABB,
	                                                  double maxAABB) {
		RealPoint p1(minAABB - gridstep, minAABB - gridstep, minAABB - gridstep);
		RealPoint p2(maxAABB + gridstep, maxAABB + gridstep, maxAABB + gridstep);

		GaussDigitizer<Z3i::Space, PolyhedronShape> digitizer;
		digitizer.attach(implicitShape);
		digitizer.init(p1, p2, gridstep);

		auto domain = digitizer.getDomain();
		auto binary_image = SH3::makeBinaryImage(domain);
		for (const auto &p: domain)
			binary_image->setValue(p, digitizer(p));

		return binary_image;
	}

	std::vector<RealVector>
	getNormalVectors(const CountedPtr<PolyhedronShape> &shape,
	                 const KSpace &K,
	                 const SH3::SurfelRange &surfels,
	                 const Parameters &params) {
		using NormalEstimator =
			DGtal::TrueDigitalSurfaceLocalEstimator<
				KSpace,
				PolyhedronShape,
				NormalFunctor>;

		std::vector<RealVector> n_estimations;
		NormalEstimator estimator;

		int maxIter = params["projectionMaxIter"].as<int>();
		auto accuracy = params["projectionAccuracy"].as<double>();
		auto gamma = params["projectionGamma"].as<double>();
		auto h = params["gridstep"].as<Scalar>();

		estimator.attach(*shape);
		estimator.setParams(K, NormalFunctor(), maxIter, accuracy, gamma);
		estimator.init(h, surfels.begin(), surfels.end());
		estimator.eval(surfels.begin(), surfels.end(),
		               std::back_inserter(n_estimations));

		return n_estimations;
	}


}
