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

#pragma region "Variable Declarations"
CascadeClassifier face_cascade;
CascadeClassifier eye_cascade;
CascadeClassifier reye_cascade;
CascadeClassifier leye_cascade;
CascadeClassifier mouth_cascade;
CascadeClassifier nose_cascade;

Scalar tmean, tstddev;
bool badTemplate;
bool resetting = false;
bool pointing = true;
bool activeIris = true;
bool pupilOn = false;
bool lookingRight = true;

double pminVal;
double pmaxVal;
Point pminLoc;
Point pmaxLoc;

double rminVal;
double rmaxVal;
Point rminLoc;
Point rmaxLoc;

double lminVal;
double lmaxVal;
Point lminLoc;
Point lmaxLoc;

double fminVal;
double fmaxVal;
Point fminLoc;
Point fmaxLoc;

double nminVal;
double nmaxVal;
Point nminLoc;
Point nmaxLoc;

double mminVal;
double mmaxVal;
Point mminLoc;
Point mmaxLoc;

Mat pinv;

Mat demoDisplay;
Mat reye_tpl;
Rect reye_bb;
Mat leye_tpl;
Rect leye_bb;
Mat face_tpl;
Rect face_bb;
Mat nose_tpl;
Rect nose_bb;
Mat mouth_tpl;
Rect mouth_bb;

Mat rmat, lmat;

int pupilSensitivity = 100;
vector<Vec3f> circles;
vector<Point> rcenters;
vector<Point> lcenters;

CvPoint ri;
CvPoint li;
CvPoint ns;
CvPoint mt;
CvPoint rp;
CvPoint lp;

Point cr2n;
Point cl2n;
Point cr2l;
Point cn2m;
Point cr2m;
Point cl2m;
Point cpavg2n;

Point r2n;
Point l2n;
Point r2l;
Point n2m;
Point r2m;
Point l2m;
Point pavg2n;

Point rpup, lpup, rpuptemp, lpuptemp;

#pragma endregion "Variable Declarations"

//Returns the Rect form of the bounding box of the face
Rect TargetFinder::getFace(){
	return face_bb;
}

//When passed the image of a single eye, the location of the pupil within the eye is returned in a Point
Point findPupil(Mat &img_mat, int thresholdbit, CvPoint ref, bool rightSide){
	Point toReturn(0, 0);
	ref.x += img_mat.cols / 6;
	ref.y += img_mat.rows / 5;
	img_mat = Mat(img_mat, Rect(img_mat.cols / 6, img_mat.rows / 5, (img_mat.cols * 4) / 6, (img_mat.rows * 3) / 5));
	int mmean = mean(img_mat)[0];
	if (img_mat.cols * img_mat.rows == 0)
		return toReturn;
	int lastRadius = 1;
	int bestCircle = -1;
	//Mat img_gray = img_mat;
	Mat img_gray;
	img_mat.copyTo(img_gray);
	equalizeHist(img_gray, img_gray);
	GaussianBlur(img_gray, img_gray, Size(9, 9), 2, 2);
	if (thresholdbit != 0){
		threshold(img_gray, img_gray, mmean/2, 255, THRESH_BINARY);
	}
	HoughCircles(img_gray, circles, CV_HOUGH_GRADIENT, 1, img_gray.rows / 8, 100, pupilSensitivity, 0, 0);
	if (circles.size() > 1){
		pupilSensitivity++;
		//cout << "findPupil oversensitive, gradient threshold adjusted to: " << pupilSensitivity << endl;
	}
	if (circles.size() < 1 && pupilSensitivity > 1){
		pupilSensitivity--;
		//cout << "findPupil not sensitive enough, gradient threshold adjusted to: " << pupilSensitivity << endl;
	}
	for (size_t i = 0; i < circles.size(); i++){
		/*
		after hough transform circle i's x and y stored in circles[i][0], circles[i][1]
		radius stored in circles[i][2]
		*/
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int ax = 0;
		int ay = 0;
		Point avgCntr;
		if (rightSide){ 
			//averages points from right side and left side arrays, rebuild this with pointers for neater code
			for (int j = 0; j < rcenters.size(); j++){
				ax += rcenters[j].x;
				ay += rcenters[j].y;
			}
			avgCntr = Point((ax + center.x) / (rcenters.size() + 1), (ay + center.y) / (rcenters.size() + 1));
			rcenters.insert(rcenters.begin(), center);

			if (rcenters.size()>6){
				rcenters.pop_back();
			}
			//imshow("rightEye", img_mat);
		}
		else{
			for (int j = 0; j < lcenters.size(); j++){
				ax += lcenters[j].x;
				ay += lcenters[j].y;
			}
			avgCntr = Point((ax + center.x) / (lcenters.size() + 1), (ay + center.y) / (lcenters.size() + 1));
			lcenters.insert(lcenters.begin(), center);
			if (lcenters.size()>6){
				lcenters.pop_back();
			}
			//imshow("leftEye", img_mat);
		}
		int radius = cvRound(circles[i][2]);
		int val = int(img_gray.at<uchar>(cv::Point(center)));
		if (val < 200){
			circle(img_mat, avgCntr, 3, Scalar(255, 255, 255), -1, 8, 0);
			toReturn = Point(avgCntr.x + ref.x, avgCntr.y + ref.y);
		}
	}
	return toReturn;
}

//Initializes cascade classifiers from xml files, order must be face classifier, left eye classifier, right eye classifier, mouth classifier, nose classifier
void TargetFinder::initializeCascades(const string& faceFilename, const string& leftEyeFilename,
	const string& rightEyeFilename, const string& mouthFilename, const string& noseFilename){
	face_cascade.load(faceFilename);
	leye_cascade.load(leftEyeFilename);
	reye_cascade.load(rightEyeFilename);
	mouth_cascade.load(mouthFilename);
	nose_cascade.load(noseFilename);
	cout << "HaarCascades Succesfully initialized!" << endl;
}

//Detects the location of facial features and stores them
//right eye bounding box -> rbox
//left eye bounding box -> lbox
//face bounding box -> fbox
//nose bounding box -> nbox
//mouth bounding box - mbox
//Mat templates are constructed and stored in correspond Mat addresses following the same naming convention as the bounding boxes above
int TargetFinder::detectEye(Mat& img, Mat& rtempl, Mat& ltempl, Mat& ftempl, Mat& ntempl, Mat& mtempl, Rect& rbox, Rect& lbox, Rect& fbox, Rect& nbox, Rect& mbox){
	vector<Rect> faces, eyes;               //vectors of rectangles for storing potential locations of faces and eyes
	vector<Rect> reyes, leyes;
	vector<Rect> noses, mouthes;
	face_cascade.detectMultiScale(
		img, faces, 1.1, 2,
		CV_HAAR_SCALE_IMAGE,
		cv::Size(20, 20)
		);                                  //detects faces allowing for scaling factor of 1.1, minimum neighbours of 2, and a minimum size of 30,30
	if (faces.size()){
		fbox = faces[0];
		ftempl = img(fbox);
	}
	for (int i = 0; i < faces.size(); i++){
		Mat face = img(faces[i]);
		Mat rface = face(Rect(0, 0, face.cols / 2, face.rows));
		Mat lface = face(Rect(face.cols / 2, 0, face.cols / 2, face.rows));
		Mat bface = face(Rect(0, face.rows / 2, face.cols, face.rows / 2));
		//imshow("RightFace", rface);
		//imshow("LeftFace", lface);
		/*eye_cascade.detectMultiScale(face, eyes, 1.1, 2,
		CV_HAAR_SCALE_IMAGE,
		cv::Size(20,20)
		);*/

		reye_cascade.detectMultiScale(rface, reyes, 1.1, 2,
			CV_HAAR_SCALE_IMAGE,
			cv::Size(40, 40)
			);

		leye_cascade.detectMultiScale(lface, leyes, 1.1, 2,
			CV_HAAR_SCALE_IMAGE,
			cv::Size(40, 40)
			);

		nose_cascade.detectMultiScale(face, noses, 1.1, 2,
			CV_HAAR_SCALE_IMAGE,
			cv::Size(40, 40)
			);

		mouth_cascade.detectMultiScale(bface, mouthes, 1.1, 2,
			CV_HAAR_SCALE_IMAGE,
			cv::Size(40, 40)
			);

		//for( int j = 0; j < eyes.size(); j++){
		if (reyes.size()){
			rbox = reyes[0] + cv::Point(faces[i].x, faces[i].y);
			rtempl = img(rbox);
			meanStdDev(rtempl, tmean, tstddev, Mat());
			//imshow("Right Eye Template", rtempl);
			if ((int)tstddev[0] < 10){
				badTemplate = true;
			}
			else{
				badTemplate = false;
			}
		}
		if (leyes.size()){
			lbox = leyes[0] + cv::Point(faces[i].x + faces[i].width / 2, faces[i].y);
			ltempl = img(lbox);
			meanStdDev(ltempl, tmean, tstddev, Mat());
			//imshow("Left Eye Template", ltempl);
			if ((int)tstddev[0] < 10){
				badTemplate = true;
			}
			else{
				badTemplate = false;
			}
		}

		if (noses.size()){
			nbox = noses[0] + cv::Point(faces[i].x, faces[i].y);
			ntempl = img(nbox);
			meanStdDev(ntempl, tmean, tstddev, Mat());
			//imshow("NoseTemplate", ntempl);
			if ((int)tstddev[0] < 10){
				badTemplate = true;
			}
			else{
				badTemplate = false;
			}
		}

		if (mouthes.size()){
mbox = mouthes[0] + cv::Point(faces[i].x, faces[i].y + faces[i].height / 2);
mtempl = img(mbox);
meanStdDev(mtempl, tmean, tstddev, Mat());
//imshow("MouthTemplate", mtempl);
if ((int)tstddev[0] < 10){
	badTemplate = true;
}
else{
	badTemplate = false;
}
		}
	}
	return reyes.size() * leyes.size() * noses.size() * mouthes.size();
}

//Tracks the location of facial features assuming they were found in the last frame, and valid templates are stored in the template variable addresses
//locations and templates follow the same naming convention as the detectEye() method above
void TargetFinder::trackEye(Mat& img, Mat& rtempl, Mat& ltempl, Mat& ftempl, Mat& ntempl, Mat& mtempl, Rect& rbox, Rect& lbox, Rect& fbox, Rect& nbox, Rect& mbox){
	cv::Size rsize(rbox.width * 2, rbox.height * 2);
	cv::Size lsize(lbox.width * 2, lbox.height * 2);
	cv::Size fsize(fbox.width * 2, fbox.height * 2);
	cv::Size nsize(nbox.width * 2, nbox.height * 2);
	cv::Size msize(mbox.width * 2, mbox.height * 2);

	Rect rwindow = Rect(rbox + rsize - Point(rsize.width / 2, rsize.height / 2));
	Rect lwindow = Rect(lbox + lsize - Point(lsize.width / 2, lsize.height / 2));
	Rect fwindow = Rect(fbox + fsize - Point(fsize.width / 2, fsize.height / 2));
	Rect nwindow = Rect(nbox + nsize - Point(nsize.width / 2, nsize.height / 2));
	Rect mwindow = Rect(mbox + msize - Point(msize.width / 2, msize.height / 2));

	rwindow &= Rect(0, 0, img.cols, img.rows);
	lwindow &= Rect(0, 0, img.cols, img.rows);
	fwindow &= Rect(0, 0, img.cols, img.rows);
	nwindow &= Rect(0, 0, img.cols, img.rows);
	mwindow &= Rect(0, 0, img.cols, img.rows);

	Mat rdestination(rwindow.width - rtempl.cols + 1, rwindow.height - rtempl.rows + 1, CV_32FC1);
	Mat ldestination(lwindow.width - ltempl.cols + 1, lwindow.height - ltempl.rows + 1, CV_32FC1);
	Mat fdestination(fwindow.height - ftempl.cols + 1, fwindow.width - ftempl.rows + 1, CV_32FC1);
	Mat ndestination(nwindow.height - ntempl.cols + 1, nwindow.width - ntempl.rows + 1, CV_32FC1);
	Mat mdestination(mwindow.height - mtempl.cols + 1, mwindow.width - mtempl.rows + 1, CV_32FC1);

	matchTemplate(img(rwindow), rtempl, rdestination, CV_TM_SQDIFF_NORMED);
	matchTemplate(img(lwindow), ltempl, ldestination, CV_TM_SQDIFF_NORMED);

	matchTemplate(img(fwindow), ftempl, fdestination, CV_TM_SQDIFF_NORMED);

	matchTemplate(img(nwindow), ntempl, ndestination, CV_TM_SQDIFF_NORMED);
	matchTemplate(img(mwindow), mtempl, mdestination, CV_TM_SQDIFF_NORMED);

	minMaxLoc(rdestination, &rminVal, &rmaxVal, &rminLoc, &rmaxLoc);

	minMaxLoc(ldestination, &lminVal, &lmaxVal, &lminLoc, &lmaxLoc);

	minMaxLoc(fdestination, &fminVal, &fmaxVal, &fminLoc, &fmaxLoc);

	minMaxLoc(ndestination, &nminVal, &nmaxVal, &nminLoc, &nmaxLoc);

	minMaxLoc(mdestination, &mminVal, &mmaxVal, &mminLoc, &mmaxLoc);

	if (rminVal <= 0.2)
	{
		rbox.x = rwindow.x + rminLoc.x;
		rbox.y = rwindow.y + rminLoc.y;
	}
	else
	{
		rbox.x = rbox.y = rbox.width = rbox.height = 0;
	}

	if (lminVal <= 0.2)
	{
		lbox.x = lwindow.x + lminLoc.x;
		lbox.y = lwindow.y + lminLoc.y;
	}
	else
	{
		lbox.x = lbox.y = lbox.width = lbox.height = 0;
	}

	if (nminVal <= 0.2)
	{
		nbox.x = nwindow.x + nminLoc.x;
		nbox.y = nwindow.y + nminLoc.y;
	}

	if (mminVal <= 0.2)
	{
		mbox.x = mwindow.x + mminLoc.x;
		mbox.y = mwindow.y + mminLoc.y;
	}

	if (fminVal <= 0.2)
	{
		fbox.x = fwindow.x + fminLoc.x;
		fbox.y = fwindow.y + fminLoc.y;
	}
	else
	{
		fbox.x = fbox.y = fbox.width = fbox.height = 0;
	}

	if ((nbox.y + nbox.height*0.6 > mbox.y) && (nbox.y*nbox.height*mbox.y != 0)){
		cout << "Nose and Mouth Overlap" << endl;
		detectEye(img, reye_tpl, leye_tpl, face_tpl, nose_tpl, mouth_tpl, reye_bb, leye_bb, face_bb, nose_bb, mouth_bb);
		fbox.x = fbox.y = fbox.width = fbox.height = 0;
	}

	if ((lbox.x < rbox.x + rbox.width) /*|| (lbox.x+lbox.width>fbox.x+fbox.width) || (rbox.x<fbox.x)*/)
	{
		cout << "Eyes Overlap" << endl;
		detectEye(img, reye_tpl, leye_tpl, face_tpl, nose_tpl, mouth_tpl, reye_bb, leye_bb, face_bb, nose_bb, mouth_bb);
		fbox.x = fbox.y = fbox.width = fbox.height = 0;
	}
	if ((lbox.x > fbox.x + fbox.width) && fbox.x*fbox.width!=0){
		cout << "Left eye out of bounds" << endl;
		detectEye(img, reye_tpl, leye_tpl, face_tpl, nose_tpl, mouth_tpl, reye_bb, leye_bb, face_bb, nose_bb, mouth_bb);
		fbox.x = fbox.y = fbox.width = fbox.height = 0;
		cout << "Left edge of face: " << fbox.x + fbox.width << endl << " left edge of left eye + .25: " << lbox.x + lbox.width*1.25 << endl;
	}
	if ((rbox.x + rbox.width < fbox.x) && fbox.x*fbox.width != 0){
		cout << "Right eye out of bounds" << endl;
		detectEye(img, reye_tpl, leye_tpl, face_tpl, nose_tpl, mouth_tpl, reye_bb, leye_bb, face_bb, nose_bb, mouth_bb);
		fbox.x = fbox.y = fbox.width = fbox.height = 0;
	}
}

//Calls one of detectEye or trackEye based on whether or not these variables were succesfully identified in the last frame
void TargetFinder::findPoints(Mat& frameMat){
	
	/*
		If all bounding boxes have been identified in the preivious frame and the eye templates are good, procede to track,
		otherwise use haarcascades to find initial model again
	*/
	if ((reye_bb.width*leye_bb.width*face_bb.width*nose_bb.width*mouth_bb.width == 0 &&
		reye_bb.height*leye_bb.height*face_bb.height*nose_bb.height*mouth_bb.height == 0) || badTemplate){

		detectEye(frameMat, reye_tpl, leye_tpl, face_tpl, nose_tpl, mouth_tpl, reye_bb, leye_bb, face_bb, nose_bb, mouth_bb);
		cout << "Detecting..." << endl;
	}
	else{
		trackEye(frameMat, reye_tpl, leye_tpl, face_tpl, nose_tpl, mouth_tpl, reye_bb, leye_bb, face_bb, nose_bb, mouth_bb);
	}

	//Pupil stuff here
	if (activeIris){
		rmat = frameMat(reye_bb);
		lmat = frameMat(leye_bb);

		rpuptemp = findPupil(rmat, 1, cvPoint(reye_bb.x, reye_bb.y), true);
		lpuptemp = findPupil(lmat, 1, cvPoint(leye_bb.x, leye_bb.y), false);

		if (rpuptemp.x*rpuptemp.y*lpuptemp.x*lpuptemp.y != 0){ //if pupils are found, reassign, other wise keep last value
			rpup = rpuptemp;
			lpup = lpuptemp;
		}
	}

}

//Makes several points based on the location and dimensions of the bounding boxes for each feature
//These points are easier to draw and do calculations with than the rectangles they are made from
void TargetFinder::getFacialPoints(CvPoint& rightEyePt, CvPoint& leftEyePt, CvPoint& nosePoint, CvPoint& mouthPoint, CvPoint& rightPupil, CvPoint& leftPupil){
	nosePoint = cvPoint(nose_bb.x + nose_bb.width / 2, nose_bb.y);
	mouthPoint = cvPoint(mouth_bb.x + mouth_bb.width / 2, mouth_bb.y);
	rightEyePt = cvPoint(reye_bb.x + reye_bb.width / 2, reye_bb.y);
	leftEyePt = cvPoint(leye_bb.x + leye_bb.width / 2, leye_bb.y);

	if (activeIris){
		rightPupil = rpup;
		leftPupil = lpup;
	}
}

//When the users head and face are centered this method stores the relative location of facial features, for comparison in future frames to determine the desired movement
void TargetFinder::getCenter(){
	getFacialPoints(ri, li, ns, mt, rp, lp);
	Point pupilAvg = averagePupilPos();
	cr2l = Point(ri.x - li.x, ri.y - li.y);
	cr2n = Point(ri.x - ns.x, ri.y - ns.y);
	cl2n = Point(li.x - ns.x, li.y - ns.y);
	cr2m = Point(ri.x - mt.x, ri.y - mt.y);
	cl2m = Point(li.x - mt.x, li.y - mt.y);
	cn2m = Point(ns.x - mt.x, ns.y - mt.y);
	cpavg2n = Point(pupilAvg.x - ns.x, pupilAvg.y-ns.y);
	pointing = true;

	cout << "_CENTERED VALUES_" << endl;
	cout << "left eye to right eye: " << cr2l << endl;
	cout << "right eye to nose: " << cr2n << endl;
	cout << "left eye to nose: " << cl2n << endl;
	cout << "right eye to mouth: " << cr2m << endl;
	cout << "left eye to mouth: " << cl2m << endl;
	cout << "nose to mouth: " << cn2m << endl;
	cout << "nose to center: " << cpavg2n << endl;
}

//When called, the current relative position of facial features is compared against those recorded in the getCenter method to predict a desired direction of movement
Point TargetFinder::getDirection(){
	Point toreturn;
	if (pointing){
		getFacialPoints(ri, li, ns, mt, rp, lp);
		r2l = Point(ri.x - li.x, ri.y - li.y);
		r2n = Point(ri.x - ns.x, ri.y - ns.y);
		l2n = Point(li.x - ns.x, li.y - ns.y);
		r2m = Point(ri.x - mt.x, ri.y - mt.y);
		l2m = Point(li.x - mt.x, li.y - mt.y);
		n2m = Point(ns.x - mt.x, ns.y - mt.y);

		int avgcy = (cr2n.y + cl2n.y) / 2;
		int avgy = (r2n.y + l2n.y) / 2;
		int avgcx = cn2m.x;
		int avgx = n2m.x;
		toreturn = Point(avgx-avgcx, avgcy - avgy);
		//cout << "Direction: " << toreturn << endl;
		return toreturn;
	}
	else{
		toreturn = Point(0, 0);
		return toreturn;
	}
}

//This method attempts to determine whether the user is looking to the right or the left of the targeting reticule, it doesnt work very well and is currently inactive
//To activate set the bool variable pupilOn above, to true
bool TargetFinder::getGazeDirection(){
	return lookingRight;
}

//Calls getDirection and applies changes to the CvPoint passed in as reticule
void TargetFinder::adjustReticule(Mat& frame, CvPoint& reticule){
	if (pointing){
		Point adjustment = getDirection();
		reticule.x += adjustment.x * abs(adjustment.x);
		reticule.y += adjustment.y * abs(adjustment.y);
		if (reticule.x < 0)
			reticule.x = 0;
		else if (reticule.x > frame.cols)
			reticule.x = frame.cols;
		if (reticule.y < 0)
			reticule.y = 0;
		else if (reticule.y > frame.rows)
			reticule.y = frame.rows;

		if (pupilOn){
			if (isCentered()){
				averagePupilPos();
				//cout << "reticule position: " << ((reticule.x - (frame.cols / 2)) * 10) / frame.cols << endl;
				//cout << "pupil position: " << pavg2n.x << endl;
				if (pavg2n.x > ((reticule.x-(frame.cols/2))*10)/frame.cols){
					//cout << "Looking left of reticule" << endl;
					lookingRight = false;
				}
				else{
					//cout << "Looking right of reticule" << endl;
					lookingRight = true;
				}
			}
		}
	}
}

//Determines whether or not the users head is within a threshold of the centered position, collected from getCenter()
bool TargetFinder::isCentered(){
	if (pointing){
		Point adjustment = getDirection();
		if (abs(adjustment.x) < 2 && abs(adjustment.y) < 2){
			return true;
		}
		else
			return false;
	}
	return false;
}

//Returns a point that is the average position of the two pupil points
Point TargetFinder::averagePupilPos(){
	Point pavg = rpup + lpup;
	pavg.x = pavg.x / 2;
	pavg.y = pavg.y / 2;
	pavg2n = Point(pavg.x - ns.x, pavg.y - ns.y);
	return pavg;
}

//Does not function well, deactivated for now
Point TargetFinder::getRelativePupilPosition(){
	return pavg2n - cpavg2n;
}
//Does not function well, deactivated for now
Point TargetFinder::checkPupilMovement(){
	return Point(0, 0);
}