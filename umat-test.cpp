#include <opencv2/opencv.hpp>

using namespace cv;

int main (int argc, char** argv)
{
	UMat img, gray;
	imread("image.jpg", IMREAD_COLOR).copyTo(img);
	cvtColor(img, gray, COLOR_BGR2GRAY);
	GaussianBlur (gray, gray, Size(7,7), 1.5);
	Canny(gray, gray, 0, 50);

	imshow("Edges", gray);
	waitKey();
	return 0;
}

