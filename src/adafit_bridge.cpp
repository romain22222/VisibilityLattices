#include "adafit_bridge.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace AdaFitBridge {
namespace {

std::string shellQuote(const std::string &value) {
	std::string quoted = "'";
	for (char c: value) {
		if (c == '\'') {
			quoted += "'\"'\"'";
		} else {
			quoted += c;
		}
	}
	quoted += "'";
	return quoted;
}

std::string readWholeFile(const std::filesystem::path &filePath) {
	std::ifstream input(filePath);
	if (!input) {
		return {};
	}
	std::ostringstream buffer;
	buffer << input.rdbuf();
	return buffer.str();
}

std::string normalizePath(const std::string &path) {
	if (path.empty()) {
		return {};
	}
	return std::filesystem::absolute(std::filesystem::path(path)).string();
}

} // namespace

bool exportPointCloudXYZ(const std::vector<Point> &points, const std::string &filePath, std::string *errorMessage) {
	std::ofstream output(filePath);
	if (!output) {
		if (errorMessage) {
			*errorMessage = "Unable to open point cloud export file: " + filePath;
		}
		return false;
	}
	for (const auto &point: points) {
		output << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
	}
	return true;
}

bool importNormalsXYZ(const std::string &filePath, std::vector<Point> &normals, std::string *errorMessage) {
	normals.clear();
	std::ifstream input(filePath);
	if (!input) {
		if (errorMessage) {
			*errorMessage = "Unable to open normal file: " + filePath;
		}
		return false;
	}
	Point normal;
	while (input >> normal[0] >> normal[1] >> normal[2]) {
		auto norm = normal.norm();
		if (norm > 1e-12) {
			normal /= norm;
		}
		normals.push_back(normal);
	}
	if (normals.empty()) {
		if (errorMessage) {
			*errorMessage = "No normals could be parsed from: " + filePath;
		}
		return false;
	}
	return true;
}

Result computeNormals(const std::vector<Point> &points, std::vector<Point> &normals, const Config &config) {
	Result result;
	if (points.empty()) {
		result.message = "AdaFit bridge received an empty point cloud.";
		return result;
	}

	namespace fs = std::filesystem;
	const fs::path workDir = config.outputDirectory.empty()
		                        ? fs::temp_directory_path() / "visibilityLattices_adafit"
		                        : fs::path(config.outputDirectory);
	fs::create_directories(workDir);

	const fs::path pointsFile = workDir / "point_cloud.xyz";
	const fs::path normalsFile = workDir / "normals.xyz";
	const fs::path logFile = workDir / "adafit.log";
	fs::remove(normalsFile);
	fs::remove(logFile);

	std::string errorMessage;
	if (!exportPointCloudXYZ(points, pointsFile.string(), &errorMessage)) {
		result.message = errorMessage;
		return result;
	}

	const std::string pythonExecutable = config.pythonExecutable.empty() ? "python3" : config.pythonExecutable;
	const std::string runnerScript = normalizePath(config.runnerScript);
	if (runnerScript.empty()) {
		result.message = "AdaFit runner script path is empty.";
		return result;
	}

	std::ostringstream command;
	command << shellQuote(pythonExecutable)
	        << ' ' << shellQuote(runnerScript)
	        << " --input " << shellQuote(pointsFile.string())
	        << " --output " << shellQuote(normalsFile.string())
	        << " --k " << config.kNeighbors;
	if (!config.repoPath.empty()) {
		command << " --repo " << shellQuote(normalizePath(config.repoPath));
	}
	if (!config.checkpointPath.empty()) {
		command << " --checkpoint " << shellQuote(normalizePath(config.checkpointPath));
	}
	command << " > " << shellQuote(logFile.string()) << " 2>&1";

	result.command = command.str();
	result.pointsFile = pointsFile.string();
	result.normalsFile = normalsFile.string();
	result.logFile = logFile.string();

	const int exitCode = std::system(result.command.c_str());
	if (exitCode != 0) {
		result.message = "AdaFit runner failed with exit code " + std::to_string(exitCode);
		const auto logContent = readWholeFile(logFile);
		if (!logContent.empty()) {
			result.message += "\n" + logContent;
		}
		return result;
	}

	if (!importNormalsXYZ(normalsFile.string(), normals, &errorMessage)) {
		result.message = errorMessage;
		const auto logContent = readWholeFile(logFile);
		if (!logContent.empty()) {
			result.message += "\n" + logContent;
		}
		return result;
	}
	if (normals.size() != points.size()) {
		result.message = "AdaFit returned " + std::to_string(normals.size()) +
		                 " normals for " + std::to_string(points.size()) + " input points.";
		return result;
	}

	result.success = true;
	result.message = "AdaFit normals written to " + normalsFile.string();
	const auto logContent = readWholeFile(logFile);
	if (!logContent.empty()) {
		result.message += "\n" + logContent;
	}
	return result;
}

}

