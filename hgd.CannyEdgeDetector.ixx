module;
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


export module hgd.CannyEdgeDetector;


export import <vector>;

export namespace hgd::ImageProcessing {

	export class CannyEdgeDetector {
	public:
		CannyEdgeDetector(); // Constructor
		~CannyEdgeDetector(); // Destructor

		void setSourceImage(const cv::Mat& image); // Set source image
		cv::Mat getEdgesImage() const; // Get the detected edges
		void process(); // Perform edge detection

		void setLowThreshold(float threshold);
		void setHighThreshold(float threshold);
		void setGaussianKernelWidth(int width);
		void setGaussianKernelRadius(float radius);
		void setContrastNormalized(bool normalized);

		float getLowThreshold() const;
		float getHighThreshold() const;
		int getGaussianKernelWidth() const;
		float getGaussianKernelRadius() const;
		bool isContrastNormalized() const;

	private:
		void initializeArrays();
		void computeGradients(float kernelRadius, int kernelWidth);
		void applyHysteresis(int low, int high);
		void followEdges(int x, int y, int offset, int threshold);
		void thresholdEdges();
		void computeLuminance();
		float gaussian(float x, float sigma) const;

		static constexpr float MAGNITUDE_LIMIT = 255.0f;
		static constexpr int MAGNITUDE_MAX = 255;
		static constexpr float GAUSSIAN_CUT_OFF = 0.005f;
		static constexpr float  MAGNITUDE_SCALE = 1.0f;

		cv::Mat sourceImage;
		cv::Mat edgesImage;
		std::vector<int> magnitude;
		std::vector<float> xConv, yConv, xGradient, yGradient;

		int width, height;
		float lowThreshold, highThreshold;
		float gaussianKernelRadius;
		int gaussianKernelWidth;
		bool contrastNormalized;
	};
}
