module;
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

export module hgd.FormulaGenerator;


import <vector>;

export namespace hgd::ImageProcessing {
	class FormulaGenerator {
	private:
		cv::Mat image; // Input image
		int offsetx, offsety; // Pixel offsets
		std::vector<std::vector<double>> globalPositions;
		std::vector<std::vector<double>> formelArray;
		std::vector<bool> formelFlip;

	public:
		FormulaGenerator(const cv::Mat& image, int offsetx, int offsety);

		void setGlobalPositions(const std::vector<std::vector<double>>& positions);
		const std::vector<std::vector<double>>& getFormelArray() const;
		const std::vector<bool>& getFormelFlip() const;

		void generateFormel(int numberOfMax);
	};
}
