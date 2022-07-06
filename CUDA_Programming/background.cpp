
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <cv.h>
//#include <highgui.h>
//#include "calcSiftSurf.h"
#include "opticalflow.h"
#include "stdio.h"
//#include <algorithm>
//#include <array>
#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

Ptr<BackgroundSubtractor> pMOG2;
Mat fgMaskMOG2;


#undef _GLIBCXX_DEBUG
#include <stdio.h>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <vector>

using namespace cv;
using namespace std;

void shiftImages(vector<Mat>& vimg,int sn)
{
	for(int i=0;i<sn;i++)
	{
		vimg[i]=vimg[i+sn].clone();
	}
}
int main(int argc, char* argv[]){

	int imgnum = 10;
	vector<Mat> imgs_aux(imgnum);
	vector<Mat> imgs_final(imgnum);
	Mat mask;
	char path[200];
	char filecovariance[100];
	char filename[100];
	char filename_tracks[100];
	char file1[100];
	char file2[100];
	char filecov[100];

	//Mat avg1(480,640,CV_32FC3,Scalar::all(0));
	Mat res1;

	char filename2[100];
	int m = 0;
    bool isVideoReading;
	vector<Mat> img(10);


	int framenum = 10;
	VideoCapture capture;

	if( isVideoReading )
		capture.open( filename );
	else
	{
		capture.open( CAP_OPENNI2 );
		if( !capture.isOpened() )
			capture.open( CAP_OPENNI );
	}

	 for(int j = 0; j < 1000; j++){

		 if( !capture.grab() )
		 {
			 cout << "Can not grab images." << endl;
			 return -1;
		 }
		 else{

		 Mat frame;
		 capture.retrieve( frame, CAP_OPENNI_BGR_IMAGE );

		// GaussianBlur(frame,frame, Size(5,5), 0,0);
		 Mat avgd=frame.clone();
//		 avgd.convertTo(avgd,CV_32FC3);
//
//		 accumulateWeighted(avgd,avg1,0.01);
//		 convertScaleAbs(avg1, res1);
//		 imshow("res1",res1);
//		 cout<<"J:"<<j<<endl;

		 }



			if (waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
			{
				cout << "esc key is pressed by user" << endl;
				break;
				return 0;
			}
	 }

	 imwrite( "../PROYECTO/MOTION/BACKGROUND_CASA.jpg", res1);

}





