#include "polyhedra.h"
#include <cmath>
#include <iostream>

namespace Polyhedra {
	using namespace DGtal;
	using namespace Z3i;
	using Scalar = RealVector::Component;
	using Scalars = std::vector<Scalar>;

	typedef std::pair<RealPoint, double> Plane;
	constexpr auto eps = 1e-6;

	double planeDistance(const Plane &plane, const RealPoint &p, const double gridstep) {
		return plane.first.dot(gridstep * p) - plane.second;
	}

	bool isPolyhedron(const std::string &shape) {
		static const std::vector<std::string> names =
		{
			"cube", "tetrahedron", "triangular_pyramid", "dodecahedron", "icosahedron", "cubesphere", "double_ellipsoid"
		};
		return std::ranges::any_of(names, [&](const std::string &n) { return n == shape; });
	}

	void listPolyhedron() {
		std::cout << "Available polyhedra:\n";
		std::cout << " - cube\n";
		std::cout << " - tetrahedron (alias: triangular_pyramid)\n";
		std::cout << " - dodecahedron\n";
		std::cout << " - icosahedron\n";
		std::cout << " - cubesphere\n";
		std::cout << " - double_ellipsoid\n";
	}

	static RealPoint normalize(const RealPoint &v) {
		double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
		return (n == 0.0) ? v : v / n;
	}

	PolyhedronShape::PolyhedronShape(const std::vector<Plane> &planes, double gridstep)
		: myPlanes(planes), digitization_gridstep(gridstep) {
	}

	PolyhedronShape::PolyhedronShape(const std::vector<Plane> &planes,
	                                 const std::vector<Ellipsoid> &ellipsoids,
	                                 double gridstep)
		: myPlanes(planes),
		  myEllipsoids(ellipsoids),
		  digitization_gridstep(gridstep) {
	}

	std::vector<Plane> PolyhedronShape::getPlanes() const {
		return myPlanes;
	}

	double ellipsoidDistance(const Ellipsoid &e,
	                         const RealPoint &p,
	                         double gridstep) {
		RealVector d = gridstep * p - e.center;
		double x = d.dot(e.u) / e.a;
		double y = d.dot(e.v) / e.b;
		double z = d.dot(e.w) / e.c;
		return x * x + y * y + z * z - 1.0;
	}

	RealVector ellipsoidGradient(const Ellipsoid &e,
	                             const RealPoint &p,
	                             double gridstep) {
		RealPoint d = gridstep * p - e.center;
		return 2.0 * RealVector(
			       d[0] / (e.a * e.a),
			       d[1] / (e.b * e.b),
			       d[2] / (e.c * e.c)
		       );
	}

	Orientation PolyhedronShape::planeOrientation(const RealPoint &p) const {
		Orientation o = DGtal::INSIDE;

		// 1. Plans → toujours intersection
		for (const auto &pl: myPlanes) {
			double d = planeDistance(pl, p, digitization_gridstep);
			if (d > eps) {
				o = DGtal::OUTSIDE;
				break;
			}
			if (std::abs(d) <= eps) o = DGtal::ON;
		}
		if (myPlanes.empty())
			o = DGtal::OUTSIDE;
		return o;
	}

	Orientation PolyhedronShape::orientation(const RealPoint &p) const {
		Orientation o = planeOrientation(p);
		if (o == DGtal::INSIDE || myEllipsoids.empty())
			return o;
		for (const auto &e: myEllipsoids) {
			double d = ellipsoidDistance(e, p, digitization_gridstep);
			if (d < -eps)
				return DGtal::INSIDE;
			if (d < eps)
				o = DGtal::ON;
		}
		return o;
	}

	RealPoint PolyhedronShape::totalCorrectionPlane(double gamma, const RealPoint &x) const {
		RealPoint totalCorrection(0, 0, 0);
		for (const auto &pl: myPlanes) {
			auto dist = pl.first.dot(x) - pl.second * pl.second;
			if (dist > 0) // outside half-space -> project
				totalCorrection -= gamma * dist * pl.first;
		}
		return totalCorrection;
	}

	RealPoint PolyhedronShape::totalCorrectionEllipsoid(double gamma, const RealPoint &x) const {
		RealPoint totalCorrection(0, 0, 0);
		for (const auto &el: myEllipsoids) {
			auto dist = ellipsoidDistance(el, x, digitization_gridstep);
			if (dist > 0) // outside ellipsoid -> project
				totalCorrection -= gamma * dist * ellipsoidGradient(el, x, digitization_gridstep);
		}
		return totalCorrection;
	}


	RealPoint PolyhedronShape::nearestPoint(RealPoint x,
	                                        double accuracy,
	                                        int maxIter,
	                                        double gamma) const {
		for (int iter = 0; iter < maxIter; ++iter) {
			RealPoint correctionP = totalCorrectionPlane(gamma, x);
			RealPoint correctionE = totalCorrectionEllipsoid(gamma, x);
			if ((correctionP.norm() < accuracy && !myPlanes.empty()) || (
				    correctionE.norm() < accuracy && !myEllipsoids.empty()))
				break;

			// Apply the smallest correction
			if (!myPlanes.empty() && correctionP.norm() < correctionE.norm()) {
				x += correctionP;
			} else {
				x += correctionE;
			}
		}
		return x;
	}

	RealVector PolyhedronShape::gradient(const RealPoint &p) const {
		// First check for ellipsoids
		// Then check plane orientations, if outside -> keep gradient of ellipsoid, else if any gradient of ellipsoid : ambiguous, else keep plane gradient
		RealVector grad(0, 0, 0);
		int count = 0;
		for (const auto &el: myEllipsoids) {
			if (std::abs(ellipsoidDistance(el, p, digitization_gridstep)) <= digitization_gridstep_distance + eps) {
				grad = ellipsoidGradient(el, p, digitization_gridstep);
				count++;
			}
			if (count >= 2)
				break;
		}
		if (count > 1) {
			return RealVector(0, 0, 0); // ambiguous gradient
		}
		if (count == 1) {
			// for each plane, check if outside
			for (const auto &pl: myPlanes)
				if (planeDistance(pl, p, digitization_gridstep) > digitization_gridstep_distance + eps)
					return grad; // keep ellipsoid gradient
		}
		for (const auto &pl: myPlanes) {
			if (std::abs(planeDistance(pl, p, digitization_gridstep)) <= digitization_gridstep_distance + eps) {
				grad = pl.first;
				count++;
			}
			if (count >= 2)
				break;
		}
		if (count > 1) {
			return RealVector(0, 0, 0); // ambiguous gradient
		}
		return grad;
	}


	int PolyhedronShape::countIntersections(const RealPoint &p) const {
		auto intersectingPlanes = 0;
		for (const auto &pl: myPlanes) {
			if (std::abs(planeDistance(pl, p, digitization_gridstep)) <= digitization_gridstep_distance + eps) {
				intersectingPlanes++;
			}
		}
		return intersectingPlanes;
	}

	void PolyhedronShape::principalCurvatures(const RealPoint &p, double &k1, double &k2) const {
		if (countIntersections(p) < 2) {
			k1 = 0.0;
			k2 = 0.0;
		}
		k1 = std::numeric_limits<double>::infinity();
		k2 = std::numeric_limits<double>::infinity();
	}

	double PolyhedronShape::meanCurvature(const RealPoint &p) const {
		if (countIntersections(p) < 2)
			return 0.0;
		return std::numeric_limits<double>::infinity();
	}

	double PolyhedronShape::gaussianCurvature(const RealPoint &p) const {
		if (countIntersections(p) < 2)
			return 0.0;
		return std::numeric_limits<double>::infinity();
	}

	CountedPtr<PolyhedronShape> makeImplicitPolyhedron(const std::string &shape, double gridstep, double d) {
		std::vector<Plane> planes;
		std::vector<Ellipsoid> ellipsoids;

		if (shape == "cube") {
			planes = {
				{{1, 0, 0}, d},
				{{-1, 0, 0}, d},
				{{0, 1, 0}, d},
				{{0, -1, 0}, d},
				{{0, 0, 1}, d},
				{{0, 0, -1}, d}
			};
		} else if (shape == "tetrahedron" || shape == "triangular_pyramid") {
			// Regular tetrahedron with base normal (0,-1,0)
			std::vector<RealPoint> normals = {
				{0.0, -1.0, 0.0}, // base
				{0.9428, 0.3333, 0.0000}, // side 1
				{-0.4714, 0.3333, 0.8165}, // side 2
				{-0.4714, 0.3333, -0.8165} // side 3
			};

			for (const auto &n: normals)
				planes.emplace_back(n, d);
		} else if (shape == "dodecahedron") {
			double phi = (1.0 + std::sqrt(5.0)) / 2.0;
			std::vector<RealPoint> normals = {
				{0, 1, phi},
				{0, -1, phi},
				{0, 1, -phi},
				{0, -1, -phi},
				{1, phi, 0},
				{-1, phi, 0},
				{1, -phi, 0},
				{-1, -phi, 0},
				{phi, 0, 1},
				{-phi, 0, 1},
				{phi, 0, -1},
				{-phi, 0, -1}
			};
			for (const auto &n: normals) planes.emplace_back(normalize(n), d);
		} else if (shape == "icosahedron") {
			double phi = (1.0 + std::sqrt(5.0)) / 2.0;
			std::vector<RealPoint> positions = {
				{phi, 1, 0},
				{-phi, 1, 0},
				{phi, -1, 0},
				{-phi, -1, 0},
				{1, 0, phi},
				{1, 0, -phi},
				{-1, 0, phi},
				{-1, 0, -phi},
				{0, phi, 1},
				{0, -phi, 1},
				{0, phi, -1},
				{0, -phi, -1}
			};
			std::vector<RealPoint> normals;
			for (size_t i = 0; i < positions.size(); ++i) {
				for (size_t j = i + 1; j < positions.size(); ++j) {
					for (size_t k = j + 1; k < positions.size(); ++k) {
						// First check if they form a face (ie distance between points is correct)
						double d1 = (positions[i] - positions[j]).norm();
						double d2 = (positions[j] - positions[k]).norm();
						double d3 = (positions[k] - positions[i]).norm();
						double expected = 2;
						if (std::abs(d1 - expected) < 1e-5 &&
						    std::abs(d2 - expected) < 1e-5 &&
						    std::abs(d3 - expected) < 1e-5) {
							// They form a face → compute normal
							RealVector v1 = positions[j] - positions[i];
							RealVector v2 = positions[k] - positions[i];
							RealVector n = v1.crossProduct(v2);
							// Ensure normal points outward
							RealPoint center = (positions[i] + positions[j] + positions[k]) /
							                   RealPoint(3, 3, 3);
							if (n.dot(center) < 0)
								n = -n;
							normals.push_back(n);
						}
					}
				}
			}
			for (const auto &n: normals) planes.emplace_back(normalize(n), d);
			auto normalToPutDown = planes[0].first;
			auto normalDown = RealPoint(0, -1, 0);
			auto c = normalToPutDown.dot(normalDown);
			auto v = normalToPutDown.crossProduct(normalDown);
			auto vx = SimpleMatrix<double, 3, 3>();
			vx(0, 0) = 0;
			vx(0, 1) = -v[2];
			vx(0, 2) = v[1];
			vx(1, 0) = v[2];
			vx(1, 1) = 0;
			vx(1, 2) = -v[0];
			vx(2, 0) = -v[1];
			vx(2, 1) = v[0];
			vx(2, 2) = 0;
			SimpleMatrix<double, 3, 3> R;
			R.identity();
			R += vx;
			R += (vx * vx) * (1 / (1 + c));
			for (auto &p: planes) {
				p.first = R * p.first;
			}
		} else if (shape == "cubesphere") {
			planes = {
				{{1, 0, 0}, d},
				{{-1, 0, 0}, d},
				{{0, 1, 0}, d},
				{{0, -1, 0}, d},
				{{0, 0, 1}, d},
				{{0, 0, -1}, d}
			};
			ellipsoids = {
				Ellipsoid{{1, 1, 1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, 1.5 * d, 1.5 * d, 1.5 * d}
			};
		} else if (shape == "double_ellipsoid") {
			ellipsoids = {
				Ellipsoid{{-d / 1.5, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, d / 1.2, d, d},
				Ellipsoid{{d / 1.5, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, d / 1.2, d, d}
			};
		} else {
			trace.error() << "[makeImplicitPolyhedron] Unknown shape: " << shape << std::endl;
		}

		return CountedPtr<PolyhedronShape>(new PolyhedronShape(planes, ellipsoids,
		                                                       gridstep));
	}

	CountedPtr<SH3::BinaryImage> makeBinaryPolyhedron(const std::string &shape,
	                                                  double gridstep,
	                                                  double minAABB,
	                                                  double maxAABB) {
		auto implicitShape = makeImplicitPolyhedron(shape, gridstep);
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

	template<typename ReturnType, template<typename> class Functor>
	static std::vector<ReturnType> getMeasureEstimation(
		const CountedPtr<PolyhedronShape> &shape,
		const KSpace &K,
		const SH3::SurfelRange &surfels,
		const Parameters &params) {
		using MeasureFunctor = Functor<PolyhedronShape>;
		using MeasureEstimator = TrueDigitalSurfaceLocalEstimator
			<KSpace, PolyhedronShape, MeasureFunctor>;
		std::vector<ReturnType> n_true_estimations;
		MeasureEstimator true_estimator;
		int maxIter = params["projectionMaxIter"].as<int>();
		double accuracy = params["projectionAccuracy"].as<double>();
		double gamma = params["projectionGamma"].as<double>();
		Scalar gridstep = params["gridstep"].as<Scalar>();
		true_estimator.attach(*shape);
		true_estimator.setParams(K, MeasureFunctor(), maxIter, accuracy, gamma);
		true_estimator.init(gridstep, surfels.begin(), surfels.end());
		true_estimator.eval(surfels.begin(), surfels.end(),
		                    std::back_inserter(n_true_estimations));
		return n_true_estimations;
	}

	std::vector<RealVector>
	getNormalVectors(const CountedPtr<PolyhedronShape> &shape,
	                 const KSpace &K,
	                 const SH3::SurfelRange &surfels,
	                 const Parameters &params) {
		return getMeasureEstimation<RealVector, sgf::ShapeNormalVectorFunctor>(
			shape, K, surfels, params);
	}

	// Mean curvature
	Scalars getMeanCurvatureEstimation(const CountedPtr<PolyhedronShape> &shape,
	                                   const KSpace &K,
	                                   const SH3::SurfelRange &surfels,
	                                   const Parameters &params) {
		return getMeasureEstimation<Scalar, sgf::ShapeMeanCurvatureFunctor>(
			shape, K, surfels, params);
	}

	// Gaussian curvature
	Scalars getGaussianCurvatureEstimation(const CountedPtr<PolyhedronShape> &shape,
	                                       const KSpace &K,
	                                       const SH3::SurfelRange &surfels,
	                                       const Parameters &params) {
		return getMeasureEstimation<Scalar, sgf::ShapeGaussianCurvatureFunctor>(
			shape, K, surfels, params);
	}

	// Principal curvatures
	Scalars getFirstPrincipalCurvatureEstimation(const CountedPtr<PolyhedronShape> &shape,
	                                             const KSpace &K,
	                                             const SH3::SurfelRange &surfels,
	                                             const Parameters &params) {
		return getMeasureEstimation<Scalar,
			sgf::ShapeFirstPrincipalCurvatureFunctor>(
			shape, K, surfels, params);
	}

	Scalars getSecondPrincipalCurvatureEstimation(const CountedPtr<PolyhedronShape> &shape,
	                                              const KSpace &K,
	                                              const SH3::SurfelRange &surfels,
	                                              const Parameters &params) {
		return getMeasureEstimation<Scalar,
			sgf::ShapeSecondPrincipalCurvatureFunctor>(
			shape, K, surfels, params);
	}
}
