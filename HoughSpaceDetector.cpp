module;
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


module  hgd.HoughSpaceDetector;

namespace hgd::ImageProcessing {
	HoughSpaceDetector::HoughSpaceDetector(int rangeOfTheta)
        : rMax(0), rangeOfTheta(rangeOfTheta), houghSpace(){ // Proper initialization
		// houghSpace is default-constructed using cv::Mat's default constructor
	}


	void HoughSpaceDetector::buildHoughSpace(const cv::Mat& edgeImage) {
		int sizeX = edgeImage.cols;
		int sizeY = edgeImage.rows;
		rMax = static_cast<int>(sqrt(sizeX * sizeX / 4.0 + sizeY * sizeY / 4.0));

		houghSpace.create(rMax + 1, rangeOfTheta + 1, CV_8UC1); // Explicitly created
		houghSpace = cv::Scalar::all(0);

		for (int x = 0; x < sizeX; x++) {
			for (int y = 0; y < sizeY; y++) {
				if (edgeImage.at<uchar>(y, x) > 0) { // Edge pixel detected
					for (int t = 0; t < rangeOfTheta; t++) {
						double theta = CV_PI * t / 180.0;
						double cosTheta = cos(theta);
						double sinTheta = sin(theta);
						double f1 = x - sizeX / 2.0;
						double f2 = y - sizeY / 2.0;
						int r = static_cast<int>(floor(f1 * cosTheta + f2 * sinTheta + 0.5));
						r = -(r - rMax);

						if (r >= 0 && r <= rMax) {
							if (houghSpace.at<uchar>(r, t) < 255) {
								houghSpace.at<uchar>(r, t)++;
							}
						}
					}
				}
			}
		}
	}

	void HoughSpaceDetector::findMaxima(int numberOfMax) {
		cv::Mat localHoughSpace;
		houghSpace.copyTo(localHoughSpace);
		maxPositions.resize(numberOfMax, std::vector<double>(2, 0));
		int minBrightness = 300;

		for (int i = 0; i < numberOfMax; i++) {
			cv::Point maxLoc;
			double maxVal;
			cv::minMaxLoc(localHoughSpace, nullptr, &maxVal, nullptr, &maxLoc);
			int maxX = maxLoc.x;
			int maxY = maxLoc.y;

			maxPositions[i][0] = maxX;
			maxPositions[i][1] = maxY;

			if (localHoughSpace.at<uchar>(maxY, maxX) < minBrightness) {
				minBrightness = localHoughSpace.at<uchar>(maxY, maxX);
			}

			cv::rectangle(localHoughSpace, cv::Point(maxX - 10, maxY - 10),
				cv::Point(maxX + 10, maxY + 10), cv::Scalar(0), -1);
		}
	}

	const cv::Mat& HoughSpaceDetector::getHoughSpace() const {
		return houghSpace;
	}

	const std::vector<std::vector<double>> HoughSpaceDetector::getMaxPositions() const {
		return maxPositions;
	}

};

