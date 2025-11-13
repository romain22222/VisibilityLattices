#include "polyhedra.h"
#include <cmath>
#include <iostream>

namespace Polyhedra
{
	using namespace DGtal;
	using namespace Z3i;

	bool isPolyhedron(const std::string& shape)
	{
		static const std::vector<std::string> names =
			{"cube", "tetrahedron", "triangular_pyramid", "dodecahedron", "icosahedron"};
		for (auto& n : names)
			if (n == shape) return true;
		return false;
	}

	void listPolyhedron()
	{
		std::cout << "Available polyhedra:\n";
		std::cout << " - cube\n";
		std::cout << " - tetrahedron (alias: triangular_pyramid)\n";
		std::cout << " - dodecahedron\n";
		std::cout << " - icosahedron\n";
	}

	static RealPoint normalize(const RealPoint& v)
	{
		double n = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
		return (n == 0.0) ? v : v / n;
	}

	std::vector<std::pair<RealPoint,double>> makeImplicitPolyhedron(const std::string& shape)
	{
		std::vector<std::pair<RealPoint,double>> planes;

		if (shape == "cube")
		{
			double d = 5.0;
			planes = {
				{{ 1, 0, 0}, d}, {{-1, 0, 0}, d},
				{{ 0, 1, 0}, d}, {{ 0,-1, 0}, d},
				{{ 0, 0, 1}, d}, {{ 0, 0,-1}, d}
			};
		}
		else if (shape == "tetrahedron" || shape == "triangular_pyramid")
		{
			// Regular tetrahedron with base normal (0,-1,0)
			double d = 4.0;

			std::vector<RealPoint> normals = {
				{  0.0, -1.0,  0.0 },               // base
				{  0.9428,  0.3333,  0.0000 },      // side 1
				{ -0.4714,  0.3333,  0.8165 },      // side 2
				{ -0.4714,  0.3333, -0.8165 }       // side 3
			};

			for (auto n : normals)
				planes.push_back({normalize(n), d-0.5});
		}
		else if (shape == "dodecahedron")
		{
			double phi = (1.0 + std::sqrt(5.0)) / 2.0;
			std::vector<RealPoint> normals = {
				{ 0,  1,  phi}, { 0, -1,  phi}, { 0,  1, -phi}, { 0, -1, -phi},
				{ 1,  phi, 0}, {-1,  phi, 0}, { 1, -phi, 0}, {-1, -phi, 0},
				{ phi, 0,  1}, {-phi, 0,  1}, { phi, 0, -1}, {-phi, 0, -1}
			};
			double d = 4.0;
			for (auto n : normals) planes.push_back({normalize(n), d});
		}
		else if (shape == "icosahedron")
		{
			double phi = (1.0 + std::sqrt(5.0)) / 2.0;
			std::vector<RealPoint> normals = {
				{ 0,  1,  phi}, { 0, -1,  phi}, { 0,  1, -phi}, { 0, -1, -phi},
				{ 1,  phi, 0}, {-1,  phi, 0}, { 1, -phi, 0}, {-1, -phi, 0},
				{ phi, 0,  1}, {-phi, 0,  1}, { phi, 0, -1}, {-phi, 0, -1},
				{ 1,  1,  phi}, {-1,  1,  phi}, { 1, -1,  phi}, {-1, -1,  phi},
				{ 1,  1, -phi}, {-1,  1, -phi}, { 1, -1, -phi}, {-1, -1, -phi}
			};
			double d = 3.5;
			for (auto n : normals) planes.push_back({normalize(n), d});
		}
		else
		{
			trace.error() << "[makeImplicitPolyhedron] Unknown shape: " << shape << std::endl;
		}

		return planes;
	}

	class PolyhedronShape
	{
	public:
		using Space = Z3i::Space;
		using RealPoint = Space::RealPoint;

	private:
		std::vector<std::pair<RealPoint,double>> myPlanes;

	public:
		PolyhedronShape(const std::vector<std::pair<RealPoint,double>>& planes)
			: myPlanes(planes) {}

		DGtal::Orientation orientation(const RealPoint& p) const
		{
			for (auto& pl : myPlanes)
			{
				double dot = pl.first.dot(p);
				if (dot >= pl.second)
					return DGtal::OUTSIDE;
			}
			return DGtal::INSIDE;
		}
	};


	CountedPtr<SH3::BinaryImage> makeBinaryPolyhedron(const std::string& shape,
	                                                  double gridstep,
	                                                  double minAABB,
	                                                  double maxAABB)
	{
		auto planes = makeImplicitPolyhedron(shape);
		PolyhedronShape implicitShape(planes);

		RealPoint p1(minAABB - gridstep, minAABB - gridstep, minAABB - gridstep);
		RealPoint p2(maxAABB + gridstep, maxAABB + gridstep, maxAABB + gridstep);

		GaussDigitizer<Z3i::Space, PolyhedronShape> digitizer;
		digitizer.attach(implicitShape);
		digitizer.init(p1, p2, gridstep);

		auto domain = digitizer.getDomain();
		auto binary_image = SH3::makeBinaryImage(domain);
		for (auto p : domain)
			binary_image->setValue(p, digitizer(p));

		return binary_image;
	}

}
