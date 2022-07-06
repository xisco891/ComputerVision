
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>

//#include "calcSiftSurf.h"
#include "opticalflow.h"
#include "stdio.h"
#include <eigen3/Eigen/Dense>
using namespace Eigen;
#undef _GLIBCXX_DEBUG
#include <stdio.h>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <vector>
#include <stdlib.h>
#include <time.h>

#include "opticalmotion.h"
using namespace cv;
using namespace std;



int main(int argc, char* argv[]) {

	int imgnum = 10;
	vector<Mat> imgs_aux(imgnum);
	vector<Mat> imgs_final(imgnum);

	char path[200];
	//char filecovariance[100];
	//char filename[100];
	//char filename_tracks[100];
	//char file1[100];
	//char file2[100];
	//char filecov[100];

	//char filename2[100];

	int m = 0;
	int start=0;

	vector<Mat> img(10);
	int h = 0;
	int framenum = 10;

            for (int i = start; i < start + 3; i++) {


			Mat perro;
			perro=imread("/home/usr/PROYECTO/MOTION/VIDEO/JAPAN.jpg");
			imshow("IMAGEN",perro);
			int width=(perro.rows)/3;
			int length=perro.cols;
			perro.resize(width,length);
			imshow("IMAGEN2",perro);
			Point p1;
			Point p2;
			p1.x=434;
			p1.y=178;
			p2.x=714;
			p2.y=449;

			perro.convertTo(perro, CV_32FC3);
			//rectangle(perro,p1,p2, Scalar( 255, 0 , 0 ),  2, 4 );

			for(int k=p1.x;k<p2.x-1;k++)
				{
					for(int l=p1.y;l<p2.y-1;l++)
					{
						Vec3f  color = perro.at<Vec3f>(Point(l,k));
						if( color[0]==255 && color[1]<50)
						{
							cout<<"R:"<<color[0];

							cout<<"\nG:"<<color[1];
							cout<<"\nB:"<<color[2];
							cout<<""<<endl;

							color[0]=7;
							color[1]=138;
							color[2]=42;
							perro.at<Vec3f>(Point(l,k)) = color;

						waitKey(100);
						}

					}
				}

			perro.convertTo(perro,CV_8UC1);
			imshow("IMAGEN3",perro);
			waitKey();

}
