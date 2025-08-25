//
// Created by romro on 25/08/2025.
//

#include <DGtal/arithmetic/IntegerComputer.h>
#include <DGtal/base/Common.h>
#include <DGtal/geometry/meshes/CorrectedNormalCurrentComputer.h>
#include <DGtal/geometry/volumes/DigitalConvexity.h>
#include <DGtal/geometry/volumes/TangencyComputer.h>
#include <DGtal/helpers/StdDefs.h>
#include <DGtal/helpers/Shortcuts.h>
#include <DGtal/helpers/ShortcutsGeometry.h>
#include "middleware.h"
#include "main_gpu.cuh"

using namespace DGtal;
using namespace Z3i;

typedef std::vector<Vec3i> Vec3is;
typedef DigitalConvexity<KSpace> Convexity;
typedef typename Convexity::LatticeSet LatticeSet;


// Global variables
IntegerComputer<Integer> icgpu;

Vec3i pointVectorToVec3i(const Point &vector) {
  return {vector[0], vector[1], vector[2]};
}

/**
 * Get the main axis of the digital figure
 * @return
 */
int getLargeAxis(std::vector<int> &digital_dimensions) {
  auto axis = 0;
  for (auto i = 1; i < digital_dimensions.size() / 3; i++) {
    if (digital_dimensions[i] > digital_dimensions[axis]) axis = i;
  }
  return axis;
}

/**
 * Get all the unique integer vectors of maximum coordinate r
 * Only the vectors with gcd(x, y, z) = 1 are considered
 * @param r
 * @return
 */
Vec3is getAllVectorsVec3i(int radius) {
  Vec3is vectors;
  for (auto x = radius; x >= 1; x--) {
    for (auto y = radius; y >= -radius; y--) {
      for (auto z = radius; z >= -radius; z--) {
        if (icgpu.gcd(icgpu.gcd(x, y), z) != 1) continue;
        vectors.emplace_back(x, y, z);
      }
    }
  }
  for (auto y = radius; y >= 1; y--) {
    for (auto z = radius; z >= -radius; z--) {
      if (icgpu.gcd(y, z) != 1) continue;
      vectors.emplace_back(0, y, z);
    }
  }
  vectors.emplace_back(0, 0, 1);
  return vectors;
}

MyLatticeSet makeLatticeSet(LatticeSet &latticeSet) {
  Vec3is points;
  std::vector<IntervalList> allIntervals;

  for (auto [key, intervals]: latticeSet.data()) {
    points.push_back(pointVectorToVec3i(key));
    const auto &ivs = intervals.data();
    IntervalList intervalList;
    intervalList.data = new IntervalGpu[ivs.size()];
    intervalList.capacity = ivs.size();
    intervalList.size = ivs.size();
    for (size_t i = 0; i < ivs.size(); ++i) {
      intervalList.data[i] = {ivs[i].first, ivs[i].second};
    }
    allIntervals.push_back(intervalList);
  }

  MyLatticeSet result(latticeSet.axis(), points, allIntervals);
  return result;
}

HostVisibility computeVisibilityGpu(int radius, std::vector<int> &digital_dimensions,
                                    std::vector<Point> &pointels) {
  std::cout << "Computing visibility GPU" << std::endl;
  auto axis = getLargeAxis(digital_dimensions);
  auto tmpL = LatticeSetByIntervals<Space>(pointels.cbegin(), pointels.cend(), axis).starOfPoints();
  std::cout << "Lattice set computed" << std::endl;
  auto figLattices = makeLatticeSet(tmpL);

//  const auto axises_idx = std::vector < int > {axis, axis == 0 ? 1 : 0, axis == 2 ? 1 : 2};
  int *axises_idx = new int[3]{axis, axis == 0 ? 1 : 0, axis == 2 ? 1 : 2};
  auto segmentList = getAllVectorsVec3i(radius);

  std::cout << "Segment list computed with " << segmentList.size() << " segments" << std::endl;

  auto *pointelsData = new Vec3i[pointels.size()];
  for (size_t i = 0; i < pointels.size(); ++i) {
    pointelsData[i] = pointVectorToVec3i(pointels[i]);
  }

  std::cout << "Pointels digitized" << std::endl;

  HostVisibility visibility = computeVisibility(
//        (segmentList.size() + 255) / 256, 256,
      segmentList.size(), 1,
      axis, digital_dimensions.data(), axises_idx,
      figLattices,
      segmentList.data(), segmentList.size(),
      pointelsData, pointels.size());

  delete[] axises_idx;
  std::cout << "Visibility computed" << std::endl;
  return visibility;
}