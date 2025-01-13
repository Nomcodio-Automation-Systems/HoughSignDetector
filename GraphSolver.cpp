module;
#include <array>

module hgd.GraphSolver;

namespace hgd::ImageProcessing {
	GraphSolver::GraphSolver(int rows, int cols, int maxPoints)
		: imageRows(rows), imageCols(cols), maxPoints(maxPoints) {
		intersectionMatrix.resize(maxPoints, std::vector<Intersection>(maxPoints));
		visitedNodes.resize(maxPoints, false);
	}

	void GraphSolver::setFormulasAndBounds(const std::vector<std::vector<double>>& formulas,
		const std::vector<std::array<int, 4>>& bounds,
		const std::vector<bool>& lineFlipFlags) {
		lineFormulas = formulas;
		lineBounds = bounds;
		lineFlip = lineFlipFlags;
	}

	void GraphSolver::findIntersections(int numberOfLines) {
		for (int i = 0; i < numberOfLines; ++i) {
			double slope1 = lineFormulas[i][0];
			double intercept1 = lineFormulas[i][1];
			bool flip1 = lineFlip[i];

			for (int j = 0; j < numberOfLines; ++j) {
				if (i == j) continue;

				double slope2 = lineFormulas[j][0];
				double intercept2 = lineFormulas[j][1];
				bool flip2 = lineFlip[j];

				double xIntersect = 0.0, yIntersect = 0.0;
				bool found = false;

				if (flip1 && !flip2) {
					xIntersect = intercept1;
					yIntersect = slope2 * intercept1 + intercept2;
					found = true;
				}
				else if (flip2 && !flip1) {
					xIntersect = intercept2;
					yIntersect = slope1 * intercept2 + intercept1;
					found = true;
				}
				else if (!flip1 && !flip2) {
					double deltaSlope = slope1 - slope2;
					double deltaIntercept = intercept1 - intercept2;

					if (deltaSlope != 0.0) {
						xIntersect = -deltaIntercept / deltaSlope;
						yIntersect = slope1 * xIntersect + intercept1;
						found = true;
					}
				}

				if (found) {
					const auto& bounds1 = lineBounds[i];
					const auto& bounds2 = lineBounds[j];

					if (xIntersect >= 0 && xIntersect < imageCols &&
						yIntersect >= 0 && yIntersect < imageRows &&
						xIntersect >= bounds1[0] && xIntersect <= bounds1[1] &&
						xIntersect >= bounds2[0] && xIntersect <= bounds2[1] &&
						yIntersect >= bounds1[2] && yIntersect <= bounds1[3] &&
						yIntersect >= bounds2[2] && yIntersect <= bounds2[3]) {

						intersectionMatrix[i][j] = { xIntersect, yIntersect, true };
					}
				}
			}
		}
	}
	bool GraphSolver::recursiveSolve(int remainingCorners, int lastEdgeIndex) {
		bool found = false;

		for (int i = 0; i < maxPoints && !found; ++i) {
			if (intersectionMatrix[lastEdgeIndex][i].valid) {
				if (visitedNodes[i] && remainingCorners == 0) {
					// Push an array<double, 2> to resultPoints
					resultPoints.push_back(std::array<double, 2>{ intersectionMatrix[lastEdgeIndex][i].x, intersectionMatrix[lastEdgeIndex][i].y });
					++resultCounter;
					return true;
				}
				else if (!visitedNodes[i] && remainingCorners > 0) {
					visitedNodes[i] = true;
					found = recursiveSolve(remainingCorners - 1, i);
					if (found) {
						// Push an array<double, 2> to resultPoints
						resultPoints.push_back(std::array<double, 2>{ intersectionMatrix[lastEdgeIndex][i].x, intersectionMatrix[lastEdgeIndex][i].y });
						++resultCounter;
					}
				}
			}
		}
		return found;
	}

	bool GraphSolver::solveGraph(int requiredCorners) {
		for (size_t i = 0; i < maxPoints; ++i) {
			std::fill(visitedNodes.begin(), visitedNodes.end(), false);
			visitedNodes[i] = true;

			if (recursiveSolve(requiredCorners, i)) {
				return true;
			}
		}
		return false;
	}
}