module;
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

module hgd.FormulaGenerator;

import <vector>;

namespace hgd::ImageProcessing {

	FormulaGenerator::FormulaGenerator(const cv::Mat& image, int offsetx, int offsety)
		: image(image), offsetx(offsetx), offsety(offsety) {
	}

	void FormulaGenerator::setGlobalPositions(const std::vector<std::vector<double>>& positions) {
		globalPositions = positions;
		formelArray.resize(positions.size(), std::vector<double>(2, 0.0));
		formelFlip.resize(positions.size(), false);
	}

	const std::vector<std::vector<double>>& FormulaGenerator::getFormelArray() const {
		return formelArray;
	}

	const std::vector<bool>& FormulaGenerator::getFormelFlip() const {
		return formelFlip;
	}

	void FormulaGenerator::generateFormel(int numberOfMax) {
		int xPixel, yPixel, x1, x2, y1, y2, r;
		double t, m, b;
		bool weHaveFirst, weHaveSecond, flip;
		int thickness = 1;
		int lineType = 8;

		for (int i = 0; i < numberOfMax; i++) {
			t = globalPositions[i][0]; // Angle (theta)
			r = globalPositions[i][1]; // Distance (rho)

			weHaveFirst = false;
			weHaveSecond = false;

			// Search along x-axis
			for (int x = 0 + offsetx; x < image.cols - offsetx; x++) {
				yPixel = static_cast<int>((static_cast<double>(r) / std::sin(t)) -
					((static_cast<double>(x) - static_cast<double>(image.cols) / 2.0) *
						(std::cos(t) / std::sin(t))) +
					static_cast<double>(image.rows) / 2.0);

				yPixel += offsety;

				if (yPixel >= 0 && yPixel < image.rows && x >= 0 && x < image.cols) {
					if (weHaveFirst) {
						x2 = x;
						y2 = yPixel;
						weHaveSecond = true;
					}
					else {
						x1 = x;
						y1 = yPixel;
						weHaveFirst = true;
					}
				}
			}

			// If no valid points found in x-direction, search along y-axis
			if (!weHaveFirst || !weHaveSecond) {
				weHaveFirst = false;
				weHaveSecond = false;

				for (int y = 0 + offsety; y < image.rows - offsety; y++) {
					xPixel = static_cast<int>((static_cast<double>(r) / std::cos(t)) -
						((static_cast<double>(y) - static_cast<double>(image.rows) / 2.0) *
							std::tan(t)) +
						static_cast<double>(image.cols) / 2.0);

					xPixel += offsetx;

					if (xPixel >= 0 && xPixel < image.cols && y >= 0 && y < image.rows) {
						if (weHaveFirst) {
							x2 = xPixel;
							y2 = y;
							weHaveSecond = true;
						}
						else {
							x1 = xPixel;
							y1 = y;
							weHaveFirst = true;
						}
					}
				}
			}

			if (weHaveFirst && weHaveSecond) {
				cv::Point P1(x1, y1);
				cv::Point P2(x2, y2);

				// Draw the line on the image
				cv::line(image, P1, P2, cv::Scalar(0, 255, 0), thickness, lineType);

				// Compute the slope (m) and intercept (b)
				if ((x2 - x1) != 0) {
					m = static_cast<double>(y2 - y1) / (x2 - x1);
					b = static_cast<double>(y1) - (m * x1);
					flip = false;
				}
				else {
					m = static_cast<double>(x2 - x1) / (y2 - y1);
					b = static_cast<double>(x1) - (m * y1);
					flip = true;
				}

				formelArray[i][0] = m;
				formelArray[i][1] = b;
				formelFlip[i] = flip;
			}
		}
	}

} // namespace hgd::ImageProcessing
