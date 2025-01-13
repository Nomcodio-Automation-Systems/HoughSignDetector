module;
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

export module hgd.HoughSpaceDetector;

export import <vector>;

export namespace hgd::ImageProcessing {
	export class HoughSpaceDetector {
	private:
		cv::Mat houghSpace;
		int rMax;
		int rangeOfTheta;
		std::vector<std::vector<double>>  maxPositions;

	public:
		HoughSpaceDetector(int rangeOfTheta = 360);

		void buildHoughSpace(const cv::Mat& edgeImage);
		void findMaxima(int numberOfMax);

		const cv::Mat& getHoughSpace() const;
		const std::vector<std::vector<double>> getMaxPositions() const;
	};
};


