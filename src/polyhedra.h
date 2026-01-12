#ifndef POLYHEDRA__H
#define POLYHEDRA__H

#include <DGtal/helpers/StdDefs.h>
#include <DGtal/base/CountedPtr.h>
#include <DGtal/helpers/Shortcuts.h>
#include <DGtal/helpers/ShortcutsGeometry.h>
#include <DGtal/shapes/GaussDigitizer.h>

#include <string>
#include <vector>
#include <utility>

namespace Polyhedra {
	using namespace DGtal;
	using namespace Z3i;
	typedef Shortcuts<KSpace> SH3;
	typedef ShortcutsGeometry<KSpace> SHG3;
	typedef std::pair<RealPoint, double> Plane;

	inline float digitization_gridstep_distance = 1.0f;

	double planeDistance(const Plane &plane, const RealPoint &p, double gridstep);

	bool isPolyhedron(const std::string &shape);

	class PolyhedronShape {
	public:
		using Space = Z3i::Space;
		using RealPoint = Space::RealPoint;
		using RealVector = Space::RealVector;

	private:
		std::vector<Plane> myPlanes;
		double digitization_gridstep;

	public:
		PolyhedronShape(const std::vector<Plane> &planes, double gridstep);

		std::vector<Plane> getPlanes() const;

		Orientation orientation(const RealPoint &p) const;

		RealPoint nearestPoint(RealPoint x,
		                       double accuracy,
		                       int maxIter,
		                       double gamma) const;

		RealVector gradient(const RealPoint &p) const;

		int countIntersections(const RealPoint &p) const;

		void principalCurvatures(const RealPoint &p, double &k1, double &k2) const;

		double meanCurvature(const RealPoint &p) const;

		double gaussianCurvature(const RealPoint &p) const;
	};

	CountedPtr<PolyhedronShape> makeImplicitPolyhedron(const std::string &shape, double gridstep, int d = 1.0);

	typedef sgf::ShapeNormalVectorFunctor<PolyhedronShape> NormalFunctor;

	CountedPtr<SH3::BinaryImage> makeBinaryPolyhedron(const std::string &shape,
	                                                  double gridstep,
	                                                  double minAABB,
	                                                  double maxAABB);

	CountedPtr<SH3::BinaryImage> makeBinaryPolyhedron(const CountedPtr<PolyhedronShape> &implicitShape,
	                                                  double gridstep,
	                                                  double minAABB,
	                                                  double maxAABB);

	std::vector<RealVector> getNormalVectors(const CountedPtr<PolyhedronShape> &shape,
	                                         const KSpace &K,
	                                         const SH3::SurfelRange &surfels,
	                                         const Parameters &params);

	SH3::Scalars getMeanCurvatureEstimation(const CountedPtr<PolyhedronShape> &shape,
	                                        const KSpace &K,
	                                        const SH3::SurfelRange &surfels,
	                                        const Parameters &params);

	SH3::Scalars getGaussianCurvatureEstimation(const CountedPtr<PolyhedronShape> &shape,
	                                            const KSpace &K,
	                                            const SH3::SurfelRange &surfels,
	                                            const Parameters &params);

	SH3::Scalars getFirstPrincipalCurvatureEstimation(const CountedPtr<PolyhedronShape> &shape,
	                                                  const KSpace &K,
	                                                  const SH3::SurfelRange &surfels,
	                                                  const Parameters &params);

	SH3::Scalars getSecondPrincipalCurvatureEstimation(const CountedPtr<PolyhedronShape> &shape,
	                                                   const KSpace &K,
	                                                   const SH3::SurfelRange &surfels,
	                                                   const Parameters &params);


	void listPolyhedron();
}

#endif
