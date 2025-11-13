#ifndef POLYHEDRA__H
#define POLYHEDRA__H

#include <DGtal/helpers/StdDefs.h>
#include <DGtal/base/CountedPtr.h>
#include <DGtal/helpers/Shortcuts.h>
#include <DGtal/shapes/GaussDigitizer.h>

#include <string>
#include <vector>
#include <utility>

namespace Polyhedra
{
	using namespace DGtal;
	using namespace Z3i;
	typedef Shortcuts<KSpace> SH3;

	bool isPolyhedron(const std::string& shape);

	std::vector<std::pair<RealPoint,double>> makeImplicitPolyhedron(const std::string& shape);

	class PolyhedronShape;

	CountedPtr<SH3::BinaryImage> makeBinaryPolyhedron(const std::string& shape,
	                                                  double gridstep,
	                                                  double minAABB,
	                                                  double maxAABB);

	void listPolyhedron();
}

#endif
