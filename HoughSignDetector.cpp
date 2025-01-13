#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Import custom modules
import hgd.CannyEdgeDetector;
import hgd.HoughSpaceDetector;
import hgd.FormulaGenerator;
import hgd.LineDetector;
import hgd.GraphSolver;

using namespace hgd::ImageProcessing;
using namespace cv;
using namespace std;

// Globals for simplicity (consider encapsulating in a context class/struct)
Mat image;
Mat src_gray;
Mat dst;
Mat mat;
Mat detected_edges;
Mat gray_dat;
const char* window_name = "Edge Map";
int lowThreshold = 50;
int max_lowThreshold = 100;
int ratiom = 3;
int externMaximas = 0;
CannyEdgeDetector canny;
HoughSpaceDetector hough;
// Function prototypes
void CannyThreshold(int, void*);

int countMaximas(int brightness, const Mat& space);
std::tuple<int, int> nextMax(const Mat& matrix);

/**
 * Count the number of points in the given space that meet or exceed the specified brightness threshold.
 * Points are identified and processed iteratively, reducing their neighborhood's brightness in the local space.
 *
 * @param brightness The minimum brightness that a point must have to be considered.
 * @return The number of points that meet the brightness criteria.
 */
int countMaximas(int brightness, const Mat& space) {
	if (space.empty()) {
		cerr << "Input space matrix is empty." << endl;
		return 0;
	}

	Mat localSpace;
	space.copyTo(localSpace);

	int minBrightness = 300; // Arbitrary initial brightness threshold
	int count = 0;

	// Iterate until the brightness of the next maximum falls below the threshold
	while (minBrightness > brightness) {
		// Find the next maximum
		auto max = nextMax(localSpace); // Assume nextMax returns a tuple<int, int>
		int maxX = get<0>(max);
		int maxY = get<1>(max);

		// Update brightness at the maximum point
		minBrightness = localSpace.at<uchar>(maxY, maxX);

		// Exit the loop if brightness falls below threshold
		if (minBrightness < brightness) {
			break;
		}

		// Zero out the neighborhood of the maximum
		for (int xPos = maxX - 10; xPos <= maxX + 10; ++xPos) {
			for (int yPos = maxY - 10; yPos <= maxY + 10; ++yPos) {
				int wrappedX = (xPos < 0) ? (localSpace.cols + xPos) : (xPos % localSpace.cols);
				int wrappedY = (yPos < 0) ? (localSpace.rows + yPos) : (yPos % localSpace.rows);

				localSpace.at<uchar>(wrappedY, wrappedX) = 0; // Set pixel to 0
			}
		}

		// Increment the count of valid maxima
		++count;
	}

	return count;
}

/**
 * Example of the nextMax function, which identifies the next maximum in the matrix.
 * Returns the coordinates of the maximum point as a tuple.
 */
std::tuple<int, int> nextMax(const Mat& matrix) {
	Point maxLoc;
	minMaxLoc(matrix, nullptr, nullptr, nullptr, &maxLoc); // Finds the max location
	return { maxLoc.x, maxLoc.y };
}

// Main Function
int main(int argc, const char** argv) {
	// Command-line arguments
	int maximas = 0;
	int corners = 8;
	int pixeloffset = 3;
	int upperlimit = 75;
	const char* loadimg = nullptr;

	// Parse command-line arguments
	for (int args = 1; args < argc; ++args) {
		if (argv[args] == std::string("/help")) {
			cout << "Options: /maximas n /corners n /pixeloffset n /threshold n /image <path> /ratio n /minbrig n" << endl;
			return 0;
		}
		if (argv[args] == std::string("/maximas") && args + 1 < argc) maximas = stoi(argv[++args]);
		else if (argv[args] == std::string("/corners") && args + 1 < argc) corners = stoi(argv[++args]);
		else if (argv[args] == std::string("/pixeloffset") && args + 1 < argc) pixeloffset = stoi(argv[++args]);
		else if (argv[args] == std::string("/threshold") && args + 1 < argc) lowThreshold = stoi(argv[++args]);
		else if (argv[args] == std::string("/image") && args + 1 < argc) loadimg = argv[++args];
		else if (argv[args] == std::string("/ratio") && args + 1 < argc) ratiom = stoi(argv[++args]);
		else if (argv[args] == std::string("/minbrig") && args + 1 < argc) upperlimit = stoi(argv[++args]);
	}

	if (!loadimg) loadimg = "../stopr.jpg";
	image = imread(loadimg, IMREAD_COLOR);
	if (!image.data) {
		cerr << "Could not open or find the image." << endl;
		return -1;
	}

	// Pre-process image
	dst.create(image.size(), image.type());
	cvtColor(image, src_gray, COLOR_BGR2GRAY);

	// Display and trackbar setup
	namedWindow(window_name, WINDOW_AUTOSIZE);
	createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
	CannyThreshold(0, nullptr);

	// Handle maxima detection and processing
	if (maximas == 0) {
		maximas = upperlimit > 0 ? countMaximas(upperlimit,image) : 21;
	}
	externMaximas = maximas;

	// Allocate resources dynamically
	vector<vector<int>> maxPosArray(maximas, vector<int>(2));
	vector<vector<double>> globalPositions(maximas, vector<double>(2));
	vector<vector<double>> formelArray(maximas);
	vector<bool> formelFlip(maximas);
	vector<array<int, 4>> formelborder(maximas);

	vector<vector<vector<double>>> graphArray(maximas, vector<vector<double>>(maximas, vector<double>(5, 0.0)));
	vector<vector<double>> resultPoints(maximas, vector<double>(2));
	vector<bool> graphVisited(maximas, false);

	// Execute main functionality
	//
	hough.findMaxima(maximas);
	globalPositions = hough.getMaxPositions();

	FormulaGenerator formel(hough.getHoughSpace(), pixeloffset, pixeloffset);
	formel.setGlobalPositions(globalPositions);
	formel.generateFormel(maximas);

	formelArray = formel.getFormelArray();
	formelFlip = formel.getFormelFlip();

	LineDetector lineDet(maximas, pixeloffset, pixeloffset, pixeloffset);
	lineDet.processImage(image, globalPositions);
	lineDet.displayLines("Detected Lines");
	formelborder = lineDet.getFormelBorders();
	
	GraphSolver graphSolver(image.rows, image.cols, maximas);
	graphSolver.setFormulasAndBounds(formelArray, formelborder, formelFlip);
	graphSolver.findIntersections(maximas);
	bool is = graphSolver.solveGraph(corners);
	

	if (is) {
		cout << "We found the object with: " << corners << " corners." << endl;
	}
	else {
		cout << "We didn't find the object with: " << corners << " corners." << endl;
	}

	// Display results
	namedWindow("Hough-Space", WINDOW_AUTOSIZE);
	imshow("Hough-Space", mat);
	namedWindow("Picture", WINDOW_AUTOSIZE);
	imshow("Picture", image);

	waitKey(0);
	return 0;
}

// Function Definitions

void CannyThreshold(int, void*) {

	// Reduce noise with a kernel 3x3
	cv::blur(src_gray, detected_edges, cv::Size(3, 3));

	// Use CannyEdgeDetector for edge detection
	canny.setSourceImage(detected_edges);
	canny.process();
	detected_edges = canny.getEdgesImage();

	// Using Canny's output as a mask, display the result
	dst = cv::Scalar::all(0);
	image.copyTo(dst, detected_edges);

	cv::imshow(window_name, dst);

	// Convert the result to grayscale
	gray_dat.create(dst.size(), dst.type());
	gray_dat = cv::Scalar::all(0);
	cv::cvtColor(dst, gray_dat, cv::COLOR_BGR2GRAY);

	// Use HoughSpaceDetector for Hough transform
	
	hough.buildHoughSpace(gray_dat);
	cv::Mat houghSpace = hough.getHoughSpace();

	// Optional: Display Hough space for visualization
	cv::imshow("Hough Space", houghSpace);
}


