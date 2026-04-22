#pragma once

#include <string>
#include <vector>

#include <DGtal/helpers/StdDefs.h>

namespace AdaFitBridge {

using Point = DGtal::Z3i::RealPoint;

struct Config {
	std::string pythonExecutable = "python3";
	std::string runnerScript;
	std::string repoPath;
	std::string checkpointPath;
	std::string outputDirectory;
	int kNeighbors = 64;
};

struct Result {
	bool success = false;
	std::string message;
	std::string command;
	std::string pointsFile;
	std::string normalsFile;
	std::string logFile;
};

bool exportPointCloudXYZ(const std::vector<Point> &points, const std::string &filePath, std::string *errorMessage = nullptr);
bool importNormalsXYZ(const std::string &filePath, std::vector<Point> &normals, std::string *errorMessage = nullptr);
Result computeNormals(const std::vector<Point> &points, std::vector<Point> &normals, const Config &config);

}

