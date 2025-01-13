module;
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

export module hgd.LineDetector;

import <vector>;

export namespace hgd::ImageProcessing {

	export class LineDetector {
	public:
		LineDetector(int maxPoints, int deviation, int offsetX, int offsetY);

		void processImage(const cv::Mat& inputImage, const std::vector<std::vector<double>>& houghPositions);


		const std::vector<std::array<int, 4>>& getFormelBorders() const;

		void displayLines(const std::string& windowName = "Detected Lines") const;

	private:
		int maxPoints;
		int deviation;
		int offsetX;
		int offsetY;

		std::vector<std::array<int, 4>> formelBorders;
		cv::Mat processedImage;

		bool isValidPixel(int x, int y, const cv::Mat& grayData) const;

		void updateLineEntry(int startX, int startY, int endX, int endY,
			int& entryLength, int& entryStartX, int& entryStartY,
			int& entryEndX, int& entryEndY) const;
	};
}
