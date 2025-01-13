module;
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

module hgd.LineDetector;

namespace hgd::ImageProcessing {

	bool LineDetector::isValidPixel(int x, int y, const cv::Mat& grayData) const {
		for (int m = x - deviation; m <= x + deviation; ++m) {
			for (int n = y - deviation; n <= y + deviation; ++n) {
				if (m >= 0 && n >= 0 && m < grayData.cols && n < grayData.rows && grayData.at<uchar>(n, m) > 0) {
					return true;
				}
			}
		}
		return false;
	}

	void LineDetector::updateLineEntry(int startX, int startY, int endX, int endY,
		int& entryLength, int& entryStartX, int& entryStartY,
		int& entryEndX, int& entryEndY) const {
		int newLength = static_cast<int>(std::sqrt(std::pow(endX - startX, 2) + std::pow(endY - startY, 2)));
		if (newLength > entryLength) {
			entryLength = newLength;
			entryStartX = startX;
			entryStartY = startY;
			entryEndX = endX;
			entryEndY = endY;
		}
	}

	LineDetector::LineDetector(int maxPoints, int deviation, int offsetX, int offsetY)
		: maxPoints(maxPoints), deviation(deviation), offsetX(offsetX), offsetY(offsetY) {
	}

    void LineDetector::processImage(const cv::Mat& inputImage, const std::vector<std::vector<double>>& houghPositions) {
        processedImage = cv::Mat::zeros(inputImage.size(), inputImage.type());
        cv::Mat grayData;
        cv::cvtColor(inputImage, grayData, cv::COLOR_BGR2GRAY);

        formelBorders.resize(maxPoints);

        for (int i = 0; i < maxPoints; ++i) {
            double theta = houghPositions[i][0];
            int rho = static_cast<int>(houghPositions[i][1]);

            int maxEntryLength = 0;
            int entryStartX = 0, entryStartY = 0, entryEndX = 0, entryEndY = 0;

            bool weHaveFirst = false, weHaveSecond = false;

            // Check along x-axis
            for (int x = offsetX; x < inputImage.cols - offsetX; ++x) {
                int yPixel = static_cast<int>((rho / std::sin(theta)) -
                    ((x - static_cast<double>(inputImage.cols) / 2.0) * (std::cos(theta) / std::sin(theta))) +
                    static_cast<double>(inputImage.rows) / 2.0) + offsetY;

                if (yPixel >= 0 && yPixel < inputImage.rows && grayData.at<uchar>(yPixel, x) > 0) {
                    if (!weHaveFirst) {
                        entryStartX = x;
                        entryStartY = yPixel;
                        weHaveFirst = true;
                    } else {
                        entryEndX = x;
                        entryEndY = yPixel;
                        weHaveSecond = true;
                    }
                } else if (weHaveFirst && !weHaveSecond) {
                    for (int m = x - deviation; m <= x + deviation; ++m) {
                        for (int n = yPixel - deviation; n <= yPixel + deviation; ++n) {
                            if (m >= 0 && n >= 0 && m < grayData.cols && n < grayData.rows && grayData.at<uchar>(n, m) > 0) {
                                entryEndX = x;
                                entryEndY = yPixel;
                                weHaveSecond = true;
                                break;
                            }
                        }
                        if (weHaveSecond) break;
                    }
                }

                if (weHaveFirst && weHaveSecond) {
                    updateLineEntry(entryStartX, entryStartY, entryEndX, entryEndY, maxEntryLength,
                        entryStartX, entryStartY, entryEndX, entryEndY);
                    weHaveFirst = false;
                    weHaveSecond = false;
                }
            }

            // Check along y-axis if no valid line found along x-axis
            if (!weHaveFirst && !weHaveSecond) {
                for (int y = offsetY; y < inputImage.rows - offsetY; ++y) {
                    int xPixel = static_cast<int>((rho / std::cos(theta)) -
                        ((y - static_cast<double>(inputImage.rows) / 2.0) * (std::tan(theta))) +
                        static_cast<double>(inputImage.cols) / 2.0) + offsetX;

                    if (xPixel >= 0 && xPixel < inputImage.cols && grayData.at<uchar>(y, xPixel) > 0) {
                        if (!weHaveFirst) {
                            entryStartX = xPixel;
                            entryStartY = y;
                            weHaveFirst = true;
                        } else {
                            entryEndX = xPixel;
                            entryEndY = y;
                            weHaveSecond = true;
                        }
                    } else if (weHaveFirst && !weHaveSecond) {
                        for (int m = xPixel - deviation; m <= xPixel + deviation; ++m) {
                            for (int n = y - deviation; n <= y + deviation; ++n) {
                                if (m >= 0 && n >= 0 && m < grayData.cols && n < grayData.rows && grayData.at<uchar>(n, m) > 0) {
                                    entryEndX = xPixel;
                                    entryEndY = y;
                                    weHaveSecond = true;
                                    break;
                                }
                            }
                            if (weHaveSecond) break;
                        }
                    }

                    if (weHaveFirst && weHaveSecond) {
                        updateLineEntry(entryStartX, entryStartY, entryEndX, entryEndY, maxEntryLength,
                            entryStartX, entryStartY, entryEndX, entryEndY);
                        weHaveFirst = false;
                        weHaveSecond = false;
                    }
                }
            }

            // Store the detected line
            formelBorders[i] = { entryStartX, entryEndX, entryStartY, entryEndY };

            // Draw the line
            cv::line(processedImage, cv::Point(entryStartX, entryStartY),
                cv::Point(entryEndX, entryEndY), cv::Scalar(0, 255, 0), 1, cv::LINE_8);
        }
    }


	const std::vector<std::array<int, 4>>& LineDetector::getFormelBorders() const {
		return formelBorders;
	}

	void LineDetector::displayLines(const std::string& windowName) const {
		cv::imshow(windowName, processedImage);
	}
}
