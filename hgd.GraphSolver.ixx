export module hgd.GraphSolver;

import <vector>;

export namespace hgd::ImageProcessing {
	export class GraphSolver {
	public:
		GraphSolver(int rows, int cols, int maxPoints);

		void findIntersections(int numberOfLines);
		bool solveGraph(int requiredCorners);
		void setFormulasAndBounds(const std::vector<std::vector<double>>& formulas,
			const std::vector<std::array<int, 4>>& bounds,
			const std::vector<bool>& lineFlipFlags);

	private:
		struct Intersection {
			double x = 0;
			double y = 0;
			bool valid = false;
		};

		bool recursiveSolve(int remainingCorners, int lastEdgeIndex);

		std::vector<std::vector<Intersection>> intersectionMatrix;
		std::vector<bool> visitedNodes;
		std::vector<std::array<double, 2>> resultPoints;
		std::vector<std::vector<double>> lineFormulas;
		std::vector<std::array<int, 4>> lineBounds;
		std::vector<bool> lineFlip;

		int resultCounter = 0;
		int imageRows;
		int imageCols;
		int maxPoints;
	};
}
