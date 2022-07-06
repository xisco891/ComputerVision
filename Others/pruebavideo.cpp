#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <iostream>
#include "opticalflow.h"
//#undef _GLIBCXX_DEBUG



using namespace cv;
using namespace std;




int main(int argc, char* argv[])
{

	double flag=0;
	Mat prevImg;
	Mat nextImg;

	char file_u[100];
	char file_v[100];

	VideoCapture cap(argv[1]); // open the video file for reading

	if ( !cap.isOpened() )  // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	//cap.set(CV_CAP_PROP_POS_MSEC, 300); //start the video at 300ms

	double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video

	cout << "Frame per seconds : " << fps << endl;

	namedWindow("MyVideo",CV_WINDOW_NORMAL); //create a window called "MyVideo"

	int fid=atoi(argv[2]);
	Mat frame;

	for(int i=0;i<fid;i++)

		bool bSuccess = cap.read(frame); // read a new frame from video
	namedWindow("u",CV_WINDOW_NORMAL);
	namedWindow("v",CV_WINDOW_NORMAL);

	while(1)
	{


		bool bSuccess = cap.read(frame); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read the frame from video file" << endl;
			break;
		}
		flag++;

		prevImg=frame.clone();
		namedWindow("Primer frame",CV_WINDOW_NORMAL);
		imshow("Primer frame",prevImg);


		bSuccess = cap.read(frame);
		nextImg=frame.clone();
		namedWindow("Segundo frame",CV_WINDOW_NORMAL);
		imshow("Segundo frame",nextImg);

		Mat primg_g,nxtimg_g;
		cvtColor(prevImg, primg_g, CV_BGR2GRAY);
		cvtColor(nextImg, nxtimg_g, CV_BGR2GRAY);


		opticalFlow of;

		of.sparseOpticalflow();
		of.denseflow(5);

		imshow("u",of.u1);
		imshow("v",of.v1);

		sprintf(file_u,"../U_V/U1_%03d",flag);
		sprintf(file_v,"../U_V/U1_%03d",flag);

		imwrite(file_u,of.u1);
		imwrite(file_v,of.v1);

//		of.drawFlow(nextImg,20);
//		imshow("flow",nextImg);
//
//		waitKey(10);




		imshow("Segundo frame",nextImg);
			waitKey(10);

			if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
			{
				cout << "esc key is pressed by user" << endl;
				break;
			}
		}

		return 0;

	}

//for (int i=0;i<of.corners_first_frame.size();i++)
//		{
//
//			Point2f x1=of.corners_first_frame[i];
//
//			Point2f x2=of.corners_second_frame[i];
//
//			float dx=(x1.x-x2.x);
//			float dy=(x1.y-x2.y);
//
//			float distance = sqrt( dx * dx  + dy * dy);
//
//		if(of.status[i]<1 || distance>50)
//				continue;
//
//			line(nextImg,of.corners_first_frame[i],of.corners_second_frame[i],Scalar(0,0,255),1);
//			circle(nextImg,of.corners_second_frame[i],2,Scalar(0,255,255),1);
//		}
//






