#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include "targetFinder.h"

using namespace std;
using namespace cv;

IplImage* frame;
Mat frameMat;

//Points to draw:
CvPoint rightEyePt;
CvPoint leftEyePt;
CvPoint nosePoint;
CvPoint mouthPoint;
CvPoint rightPupil;
CvPoint leftPupil;
CvPoint reticule = cvPoint(0,0);
CvScalar retColor = cvScalar(0, 0, 0);
bool reset;
void drawgPoint(IplImage* aFrame, CvPoint aPoint){
	cvCircle(aFrame, aPoint, 6, cvScalar(0, 255, 0), -1, 8, 0);
}
void drawrPoint(IplImage* aFrame, CvPoint aPoint, TargetFinder t){
	if (t.getGazeDirection()){
		retColor.val[2] += 40;
		retColor.val[0] -= 40;
		if (retColor.val[2] > 255)
			retColor.val[2] = 255;
		if (retColor.val[0] < 0)
			retColor.val[0] = 0;
		//cout << "Looking right" << endl;
	}
	else{
		retColor.val[0] += 40;
		retColor.val[2] -= 40;
		if (retColor.val[0] > 255)
			retColor.val[0] = 255;
		if (retColor.val[2] < 0)
			retColor.val[2] = 0;
		//cout << "Looking left" << endl;
	}
		cvCircle(aFrame, aPoint, 40, retColor, 4, 8, 0);
}
void drawpPoint(IplImage* aFrame, CvPoint aPoint){
	cvCircle(aFrame, aPoint, 3, cvScalar(255, 0, 255), -1, 8, 0);
}
void drawbPoint(IplImage* aFrame, CvPoint aPoint){
	cvCircle(aFrame, aPoint, 3, cvScalar(255, 0, 0), -1, 8, 0);
}

int main(int argc, char **argv)
{
	//Create new TargetFinder
	TargetFinder tracker;
	//Initialize HaarCasacades for new TargetFinder, you will need these xml files form the OpenCV install (data folder)
	tracker.initializeCascades("haarcascade_frontalface_alt2.xml",
		"haarcascade_mcs_lefteye.xml", "haarcascade_mcs_righteye.xml", "haarcascade_mcs_mouth.xml",
		"haarcascade_mcs_nose.xml");

	cout << "Press C when you feel the user's face is centered" << endl;

	cvNamedWindow("Color", CV_WINDOW_NORMAL);
	CvCapture* capture = cvCreateCameraCapture(0);
	
	while (1) {
		frame = cvQueryFrame(capture);    //IplImage frame taken from video source
		//Flip frame, so user's appearance is mirrored
		cvFlip(frame, frame, 1);
		//Create Mat of frame for matrix operations
		frameMat = frame;
		if (reset){
			frameMat = Mat::zeros(frameMat.rows, frameMat.cols, frameMat.type());
			reset = false;
		}
		cvtColor(frameMat, frameMat, CV_BGR2GRAY);
		//If no frame available terminate program
		if (!frame) break;


		//Find trackable features on face - step one
		tracker.findPoints(frameMat);
		//Return location of trackable features on face - step two
		tracker.getFacialPoints(rightEyePt, leftEyePt, nosePoint, mouthPoint, rightPupil, leftPupil);
		//Draw points on frame
		drawgPoint(frame, rightEyePt);
		drawgPoint(frame, leftEyePt);
		drawgPoint(frame, nosePoint);
		drawgPoint(frame, mouthPoint);
		drawpPoint(frame, rightPupil);
		drawpPoint(frame, leftPupil);
		drawbPoint(frame, tracker.averagePupilPos());
		cvRectangle(frame, tracker.getFace().tl(), tracker.getFace().br(), cvScalar(255, 0, 0), 3, 8, 0);
		
		//adjust position of point "reticule" based on current position of head
		tracker.adjustReticule(frameMat, reticule);
		drawrPoint(frame, reticule, tracker);

		cvShowImage("Color", frame);
		char c = cvWaitKey(1);             //wait 10 ms between displaying frames
		if (c == 99){
			cout << "Attempting center" << endl;
			//this store the head position, that will be used to determine future relative head positions, this should be activated before adjusting the reticule
			tracker.getCenter();
			reticule = cvPoint(frameMat.cols / 2, frameMat.rows / 2);
		}
		if (c == 100){
			//this returns the direction a reticule will be moved in, it's mostly used internally but you can access it if you need to
			tracker.getDirection();
			reset = true;
			
		}
		if (c == 27) {
			cout << "Stopped." << endl << "Press escape again to exit, press any other key to resume" << endl;;
			cout << "Average pupil position: " << tracker.averagePupilPos() << endl;
			cout << "relative Average pupil position: " << tracker.getRelativePupilPosition() << endl;
			char d = cvWaitKey(0);
			if (d == 27){
				break;
			}
		}
	}
		cvReleaseCapture(&capture);
		cvDestroyWindow("Test");
		waitKey(0);
		return 0;
}