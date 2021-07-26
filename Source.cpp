#include <opencv2/opencv.hpp>
#include <opencv2/photo/photo.hpp>
#include <opencv2/photo/cuda.hpp>
#include <opencv2/photo.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <chrono>
#include <time.h>
#include <stdio.h>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

void help()
{
	cout << "\nThis program demonstrates circle finding.\n"
		"Usage:\n"
		"./houghlines <image_name>, Default is image.jpg\n" << endl;
}

void detectObject(const std::string& filename) {

	auto start = std::chrono::system_clock::now();
	//auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);

	Mat src = imread(filename, 1);
	if (src.empty())
	{
		help();
		cout << "can not open " << filename << endl;
		return;
	}

	// Convert input image to HSV
	cv::Mat gray_image, hsv_image;
	cv::cvtColor(src, hsv_image, cv::COLOR_BGR2HSV);
	imshow("HSV", hsv_image);

	//cvtColor(imHSV, imHSV, CV_BGR2GRAY);
	Mat hsv_channels[3];
	cv::split(hsv_image, hsv_channels);
	imshow("HSV to gray", hsv_channels[2]);

	imshow("BGR", src);
	cvtColor(src, gray_image, CV_BGR2GRAY);
	imshow("BGR to gray", gray_image);

	// Create a structuring element (SE)
	int morph_size = 2;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	cout << element;

	Mat dst; // result matrix
			 // Apply the specified morphology operation
	for (int i = 1; i < 2; i++)
	{
		morphologyEx(gray_image, dst, MORPH_OPEN, element, Point(-1, -1), i);
		//morphologyEx( gray_image, dst, MORPH_TOPHAT, element ); // here iteration=1
		imshow("result", dst);
	}
	// For now just using threshold without morphology gives better results
	Mat thr_image;
	threshold(gray_image, thr_image, 100, 255, THRESH_BINARY); //THRESH_TOZERO another possibility, 255(white)
	cv::medianBlur(thr_image, thr_image, 7);
	imshow("result 2", thr_image);

	// After that part one can use Hough transform to get circles and their centers to find the 
	// satellites which will be traced

	/*vector<Vec4i> lines;
	HoughLinesP(thr_image, lines, 1, CV_PI / 180, 60, 50, 10);
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, CV_AA);
		std::cout << "Laplacian Lines:" << lines[i] << ' ' << '\n';
	}
	imshow("detected lines 2", src);*/

	cv::Mat hsv_image2, src2;
	cv::medianBlur(src, src2, 3);
	// Convert input image to HSV
	cv::cvtColor(src2, hsv_image2, cv::COLOR_BGR2HSV);
	// Threshold the HSV image, keep only the red pixels
	cv::Mat lower_red_hue_range;
	cv::Mat upper_red_hue_range;
	cv::inRange(hsv_image2, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), lower_red_hue_range);
	cv::inRange(hsv_image2, cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255), upper_red_hue_range);
	// Combine the above two images
	cv::Mat red_hue_image;
	cv::addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
	
	//cv::GaussianBlur(red_hue_image, red_hue_image, cv::Size(9, 9), 2, 2);

	// Hough transform for circle detection

	// Use the Hough transform to detect circles in the combined threshold image
	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(thr_image, circles, CV_HOUGH_GRADIENT, 12, thr_image.rows/10 , 255, 50, 0, 40);
	// Loop over all detected circles and outline them on the original image
	if (circles.size() == 0) std::exit(-1);
	for (size_t current_circle = 0; current_circle < circles.size(); ++current_circle) {
		cv::Point center(std::round(circles[current_circle][0]), std::round(circles[current_circle][1]));
		int radius = std::round(circles[current_circle][2]);
		
		//cv::circle(src, center, radius, cv::Scalar(0, 255, 0), 5);
		
	}
	//imshow("circles", src);

	//// Set up the detector with default parameters.
	//Ptr<cv::SimpleBlobDetector> detector = SimpleBlobDetector::create();

	//// Detect blobs.
	//std::vector<KeyPoint> keypoints;
	//detector->detect(thr_image, keypoints);

	//// Draw detected blobs as red circles.
	//// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
	//Mat im_with_keypoints;
	//drawKeypoints(thr_image, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//// Show blobs
	//imshow("keypoints", im_with_keypoints);

	//cv::SimpleBlobDetector::Params params;
	//// Change thresholds
	//params.minThreshold = 10;
	//params.maxThreshold = 50;
	//params.minDistBetweenBlobs = 10.0;  // minimum 10 pixels between blobs
	//params.filterByArea = true;         // filter my blobs by area of blob
	//params.minArea = 9.0;              // min 20 pixels squared
	//params.maxArea = 500.0;             // max 500 pixels squared
	//
	//std::vector<cv::KeyPoint> myBlobs;

	//Ptr<cv::SimpleBlobDetector> myBlobDetector = cv::SimpleBlobDetector::create(params);
	//myBlobDetector->detect(gray_image, myBlobs);

	//cv::Mat blobImg;
	//cv::drawKeypoints(gray_image, myBlobs, blobImg);
	//cv::imshow("Blobs", blobImg);

	//for (std::vector<cv::KeyPoint>::iterator blobIterator = myBlobs.begin(); blobIterator != myBlobs.end(); blobIterator++) {
	//	std::cout << "size of blob is: " << blobIterator->size << std::endl;
	//	std::cout << "point is at: " << blobIterator->pt.x << " " << blobIterator->pt.y << std::endl;
	//}

	const int threshVal = 220;
	const int minArea = 2 * 2;
	const int maxArea = 7 * 7;

	cv::Mat bgr[3];
	cv::split(hsv_image, bgr);

	cv::Mat red_img = bgr[2];
	cv::threshold(red_img, red_img, threshVal, 255, cv::THRESH_BINARY);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	cv::findContours(red_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	for (int i = 0; i < contours.size(); i++)
	{
		int area = cv::contourArea(contours[i]);
		if (area < minArea || area > maxArea)
			continue;

		cv::Rect roi = cv::boundingRect(contours[i]);
		cv::rectangle(src, roi, cv::Scalar(0, 255, 0), 2);
		std::cout << "size of blob is: " << contours[i] << std::endl;
		//std::cout << "point is at: " << blobIterator->pt.x << " " << blobIterator->pt.y << std::endl;
	}

	cv::imshow("Contours", src);

	waitKey();


	auto end = std::chrono::system_clock::now();
	auto elapsed = end - start;
	std::cout << elapsed.count() << '\n';



}

// Read the image
int main(int argc, char** argv)
{
	std::string filename = argc >= 2 ? argv[1] : "./TDRS_4.jpg";

	detectObject(filename);

	waitKey();

	return 0;
}

