module;
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


module hgd.CannyEdgeDetector;
import <vector>;
  




namespace hgd::ImageProcessing {

	CannyEdgeDetector::CannyEdgeDetector()
		: lowThreshold(2.5f), highThreshold(7.5f),
		gaussianKernelRadius(2.0f), gaussianKernelWidth(16),
		contrastNormalized(false) {
	}

	CannyEdgeDetector::~CannyEdgeDetector() = default;

	void CannyEdgeDetector::setSourceImage(const cv::Mat& image) {
		if (image.empty()) throw std::invalid_argument("Source image is empty.");
		sourceImage = image.clone();
	}

	cv::Mat CannyEdgeDetector::getEdgesImage() const {
		return edgesImage.clone();
	}

	void CannyEdgeDetector::process() {
		if (sourceImage.empty()) throw std::logic_error("Source image is not set.");

		width = sourceImage.cols;
		height = sourceImage.rows;

		initializeArrays();
		computeLuminance();
		if (contrastNormalized) {
			cv::normalize(sourceImage, sourceImage, 0, 255, cv::NORM_MINMAX);
		}
		computeGradients(gaussianKernelRadius, gaussianKernelWidth);

		int low = static_cast<int>(lowThreshold);
		int high = static_cast<int>(highThreshold);

		applyHysteresis(low, high);
		thresholdEdges();
	}

	void CannyEdgeDetector::setLowThreshold(float threshold) {
		if (threshold < 0) throw std::invalid_argument("Low threshold must be non-negative.");
		lowThreshold = threshold;
	}
	void CannyEdgeDetector::setHighThreshold(float threshold) {
		if (threshold < 0) throw std::invalid_argument("High threshold must be non-negative.");
		highThreshold = threshold;
	}

	void CannyEdgeDetector::setGaussianKernelWidth(int width)
	{
		if (width < 3 || width % 2 == 0) throw std::invalid_argument("Gaussian kernel width must be an odd number greater than or equal to 3.");
		gaussianKernelWidth = width;
	}

	void CannyEdgeDetector::setGaussianKernelRadius(float radius)
	{
		if (radius <= 0) throw std::invalid_argument("Gaussian kernel radius must be positive.");
		gaussianKernelRadius = radius;
	}

	void CannyEdgeDetector::setContrastNormalized(bool normalized)
	{
		contrastNormalized = normalized;
	}

	float CannyEdgeDetector::getLowThreshold() const
	{
		return lowThreshold;
	}

	float CannyEdgeDetector::getHighThreshold() const
	{
		return highThreshold;
	}

	int CannyEdgeDetector::getGaussianKernelWidth() const
	{
		return gaussianKernelWidth;
	}

	float CannyEdgeDetector::getGaussianKernelRadius() const
	{
		return gaussianKernelRadius;
	}

	bool CannyEdgeDetector::isContrastNormalized() const
	{
		return contrastNormalized;
	}



	// Other setters and getters follow similar patterns...

	void CannyEdgeDetector::initializeArrays() {
		edgesImage = cv::Mat::zeros(height, width, CV_8UC1);
		magnitude.resize(width * height, 0);
		xConv.resize(width * height, 0.0f);
		yConv.resize(width * height, 0.0f);
		xGradient.resize(width * height, 0.0f);
		yGradient.resize(width * height, 0.0f);
	}

    void CannyEdgeDetector::computeGradients(float kernelRadius, int kernelWidth) {
        // Generate Gaussian convolution masks
        std::vector<float> gaussianKernel(kernelWidth);
        std::vector<float> derivativeKernel(kernelWidth);

        int effectiveKernelWidth = 0; // Actual width of the kernel based on Gaussian cutoff
        for (int i = 0; i < kernelWidth; ++i) {
            // Compute Gaussian values
            float g1 = gaussian(i, kernelRadius);
            if (g1 <= GAUSSIAN_CUT_OFF && i >= 2) break; // Stop when values fall below cutoff
            float g2 = gaussian(i - 0.5f, kernelRadius);
            float g3 = gaussian(i + 0.5f, kernelRadius);

            // Populate Gaussian kernel and its derivative
            gaussianKernel[i] = (g1 + g2 + g3) / 3.f / (2.f * static_cast<float>(CV_PI) * kernelRadius * kernelRadius);
            derivativeKernel[i] = g3 - g2;
            effectiveKernelWidth++;
        }

        // Compute convolution limits
        int borderOffset = effectiveKernelWidth - 1;
        int startX = borderOffset;
        int endX = width - borderOffset;
        int startY = borderOffset;
        int endY = height - borderOffset;

        // Perform convolution in X and Y directions
        for (int x = startX; x < endX; ++x) {
            for (int y = startY; y < endY; ++y) {
                int index = y * width + x;

                float xSum = sourceImage.at<uchar>(y, x) * gaussianKernel[0];
                float ySum = xSum;

                for (int offset = 1; offset < effectiveKernelWidth; ++offset) {
                    xSum += gaussianKernel[offset] * (sourceImage.at<uchar>(y, x - offset) + sourceImage.at<uchar>(y, x + offset));
                    ySum += gaussianKernel[offset] * (sourceImage.at<uchar>(y - offset, x) + sourceImage.at<uchar>(y + offset, x));
                }

                xConv[index] = xSum;
                yConv[index] = ySum;
            }
        }

        // Compute X and Y gradients using the derivative kernel
        for (int x = startX; x < endX; ++x) {
            for (int y = startY * width; y < endY * width; y += width) {
                int index = x + y;
                float gradientSum = 0.f;

                for (int offset = 1; offset < effectiveKernelWidth; ++offset) {
                    gradientSum += derivativeKernel[offset] * (yConv[index - offset] - yConv[index + offset]);
                }

                xGradient[index] = gradientSum;
            }
        }

        for (int x = effectiveKernelWidth; x < width - effectiveKernelWidth; ++x) {
            for (int y = startY * width; y < endY * width; y += width) {
                int index = x + y;
                float gradientSum = 0.f;

                for (int offset = 1, yOffset = width; offset < effectiveKernelWidth; ++offset, yOffset += width) {
                    gradientSum += derivativeKernel[offset] * (xConv[index - yOffset] - xConv[index + yOffset]);
                }

                yGradient[index] = gradientSum;
            }
        }

        // Non-Maximal Suppression
        for (int x = effectiveKernelWidth; x < width - effectiveKernelWidth; ++x) {
            for (int y = effectiveKernelWidth * width; y < (height - effectiveKernelWidth) * width; y += width) {
                int index = x + y;

                float xGrad = xGradient[index];
                float yGrad = yGradient[index];
                float magnitude = std::hypot(xGrad, yGrad);

                // Neighbor indices for non-maximal suppression
                int north = index - width;
                int south = index + width;
                int west = index - 1;
                int east = index + 1;
                int northEast = north + 1;
                int northWest = north - 1;
                int southEast = south + 1;
                int southWest = south - 1;

                float northMag = std::hypot(xGradient[north], yGradient[north]);
                float southMag = std::hypot(xGradient[south], yGradient[south]);
                float westMag = std::hypot(xGradient[west], yGradient[west]);
                float eastMag = std::hypot(xGradient[east], yGradient[east]);
                float northEastMag = std::hypot(xGradient[northEast], yGradient[northEast]);
                float northWestMag = std::hypot(xGradient[northWest], yGradient[northWest]);
                float southEastMag = std::hypot(xGradient[southEast], yGradient[southEast]);
                float southWestMag = std::hypot(xGradient[southWest], yGradient[southWest]);

                float tempComparison;

                if (xGrad * yGrad <= 0.0f
                    ? std::abs(xGrad) >= std::abs(yGrad)
                    ? (tempComparison = std::abs(xGrad * magnitude)) >= std::abs(yGrad * northEastMag - (xGrad + yGrad) * eastMag)
                    && tempComparison > std::abs(yGrad * southWestMag - (xGrad + yGrad) * westMag)
                    : (tempComparison = std::abs(yGrad * magnitude)) >= std::abs(xGrad * northEastMag - (yGrad + xGrad) * northMag)
                    && tempComparison > std::abs(xGrad * southWestMag - (yGrad + xGrad) * southMag)
                    : std::abs(xGrad) >= std::abs(yGrad)
                    ? (tempComparison = std::abs(xGrad * magnitude)) >= std::abs(yGrad * southEastMag + (xGrad - yGrad) * eastMag)
                    && tempComparison > std::abs(yGrad * northWestMag + (xGrad - yGrad) * westMag)
                    : (tempComparison = std::abs(yGrad * magnitude)) >= std::abs(xGrad * southEastMag + (yGrad - xGrad) * southMag)
                    && tempComparison > std::abs(xGrad * northWestMag + (yGrad - xGrad) * northMag)
                    ) {

                    this->magnitude[index] = magnitude >= MAGNITUDE_LIMIT ? MAGNITUDE_MAX : static_cast<int>(MAGNITUDE_SCALE * magnitude);
                } else {
                    this->magnitude[index] = 0;
                }
            }
        }
    }


	void CannyEdgeDetector::applyHysteresis(int low, int high) {
		//we reuse the magnitude array to store the edge pixels
		sourceImage = cv::Mat::zeros(height, width, CV_8UC1);
	}

	void CannyEdgeDetector::followEdges(int x, int y, int offset, int threshold)
	{
		if (x < 0 || x >= width || y < 0 || y >= height) return;
		if (edgesImage.at<uchar>(y, x) > 0 || magnitude[offset] < threshold) return;
		edgesImage.at<uchar>(y, x) = 255;
		for (int i = -1; i <= 1; ++i) {
			for (int j = -1; j <= 1; ++j) {
				followEdges(x + i, y + j, offset + i + j * width, threshold);
			}
		}
	}

	void CannyEdgeDetector::thresholdEdges() {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				edgesImage.at<uchar>(y, x) = edgesImage.at<uchar>(y, x) > 0 ? 255 : 0;
			}
		}
	}

    void CannyEdgeDetector::computeLuminance() {
        // Ensure sourceImage is valid
        if (sourceImage.empty()) {
            throw std::runtime_error("Source image is empty.");
        }

        // Get image type
        int imageType = sourceImage.type();
        cv::Mat luminanceImage;
        luminanceImage.create(sourceImage.rows, sourceImage.cols, CV_8UC1); // Default to grayscale
        luminanceImage = cv::Scalar::all(0);

        if (imageType == CV_32SC3) {
            // Handle 32-bit 3-channel images
            cv::Mat tempImage;
            tempImage.create(sourceImage.rows, sourceImage.cols, CV_32SC3);
            tempImage = cv::Scalar::all(0);
            sourceImage.copyTo(tempImage);

            for (int i = 0; i < sourceImage.total(); i++) {
                int x = i % sourceImage.cols;
                int y = i / sourceImage.cols;

                // Extract RGB values from the 32-bit integer pixel
                int pixelValue = tempImage.at<int>(y, x);
                int red = (pixelValue & 0xff0000) >> 16;
                int green = (pixelValue & 0xff00) >> 8;
                int blue = pixelValue & 0xff;

                // Compute luminance and store in the output matrix
                luminanceImage.at<uchar>(y, x) = static_cast<uchar>(0.299 * red + 0.587 * green + 0.114 * blue);
            }

        } else if (imageType == CV_8SC1) {
            // Handle 8-bit single-channel signed images
            cv::Mat tempImage;
            tempImage.create(sourceImage.rows, sourceImage.cols, CV_8SC1);
            tempImage = cv::Scalar::all(0);
            sourceImage.copyTo(tempImage);

            for (int i = 0; i < sourceImage.total(); i++) {
                int x = i % sourceImage.cols;
                int y = i / sourceImage.cols;

                // Convert signed 8-bit value to unsigned luminance
                luminanceImage.at<uchar>(y, x) = static_cast<uchar>(tempImage.at<char>(y, x) & 0xff);
            }

        } else if (imageType == CV_8UC1) {
            // Direct copy for 8-bit grayscale images
            sourceImage.copyTo(luminanceImage);

        } else {
            // Unsupported image type
            throw std::invalid_argument("Unsupported image type for luminance computation.");
        }

        // Replace the original data matrix with the computed luminance matrix
        luminanceImage.copyTo(sourceImage);
    }


	float CannyEdgeDetector::gaussian(float x, float sigma) const {
		return std::exp(-(x * x) / (2.0f * sigma * sigma));
	}

	
}
