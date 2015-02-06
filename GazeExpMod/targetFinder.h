#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>
#include <cmath>

using namespace std;
using namespace cv;

class TargetFinder
{

	//Functions here are described in depth inside of the targetFinder.cpp file
public:
	Rect getFace();
	void findPoints(Mat& frame);
	void getFaceDirection();
	void initializeCascades(const string& faceFilename, const string& leftEyeFilename, const string& rightEyeFilename, 
		const string& mouthFilename, const string& noseFilename);
	void getFacialPoints(CvPoint& rightEyePt, CvPoint& leftEyePt, CvPoint& nosePoint, CvPoint& mouthPoint, CvPoint& rightPupil, CvPoint& leftPupil);
	void getCenter();
	Point getDirection();
	void adjustReticule(Mat& frame, CvPoint& reticule);
	Point checkPupilMovement();
	bool isCentered();
	Point averagePupilPos();
	Point getRelativePupilPosition();
	bool getGazeDirection();

private:
	int detectEye(Mat& img, Mat& rtempl, Mat& ltempl, Mat& ftempl, Mat& ntempl, Mat& mtempl, Rect& rbox, Rect& lbox, Rect& fbox, Rect& nbox, Rect& mbox);
	void trackEye(Mat& img, Mat& rtempl, Mat& ltempl, Mat& ftempl, Mat& ntempl, Mat& mtempl, Rect& rbox, Rect& lbox, Rect& fbox, Rect& nbox, Rect& mbox);
};