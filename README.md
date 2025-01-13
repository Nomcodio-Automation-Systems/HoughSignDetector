# HoughSignDetector

HoughSignDetector is a C++ project designed to recognize the corners of a stop sign using the Hough Transform. This project processes an image, generates a Hough Space representation, and detects the lines and intersections that form the shape of a stop sign. It leverages OpenCV for image processing and geometric computations.

## Features

- **Image Preprocessing**: Prepares the input image for Hough Space generation by applying necessary transformations (e.g., edge detection, noise removal).
- **Hough Transform**: Computes the Hough Space to detect lines in the image.
- **Line Intersection Detection**: Identifies where lines intersect and notes their connections.
- **Graph Representation**: Constructs a graph from the detected lines and intersections.
- **Shape Matching**: Compares the graph structure to a predefined stop sign shape.

## Requirements

To build and run this project, you will need:

- **C++ Compiler**
- **OpenCV**: Version 4.0 or newer.
- **Git**: For cloning the repository.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/HoughSignDetector.git
   cd HoughSignDetector```
2.
   It was build for Visual Studio but there is no OS specific code. You only need to get OpenCV and add the dependencies
   Into the project
