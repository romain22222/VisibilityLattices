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

	bool isPolyhedron(const std::string &shape);

	class PolyhedronShape;

	CountedPtr<PolyhedronShape> makeImplicitPolyhedron(const std::string &shape, int d = 1.0);

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
