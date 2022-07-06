

//OPENCV
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/video/background_segm.hpp>

//
#include <sstream>
#include <iostream>
#include <fstream>
#include <iostream>
//#include "calcSiftSurf.h"
#include "opticalflow.h"
#include "stdio.h"
#include <eigen3/Eigen/Dense>
using namespace Eigen;
#undef _GLIBCXX_DEBUG
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "opticalmotion.h"


using namespace cv;
using namespace std;


Ptr<BackgroundSubtractor> pMOG2;
Mat fgMaskMOG2;

void shiftImages(vector<Mat>& vimg, int sn) {
	for (int i = 0; i < sn; i++) {
		vimg[i] = vimg[i + sn].clone();
	}
}

int main(int argc, char* argv[]) {

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

	Mat avg1(144,180,CV_32FC3,Scalar::all(0));
	Mat res1;

	char filename2[100];
	int m = 0;

	vector<Mat> img(10);

	int training = 0;
	int start=40;
	int h = 0;
	opticalFlow opt1;
	int framenum = 10;
	int cont=0;

	while (training < 8) {
		opticalFlow opt;

		for (int i = start; i < start + 1; i++) {
			cont=0;
			int flag = 0;
			int step = 0;
			int end = 0;
			h = i;
			if (i == 14) {
				cont++;
			}

			sprintf(path, "/home/usr/PROYECTO/MOTION/VIDEO/%03d.avi", h);
//			sprintf(path, "/home/usr/PROYECTO/MOTION/VIDEO/050.avi");


			//waitKey(2000);
			puts(path);
			VideoCapture cap(path);

			if (!cap.isOpened()) {
				cout << "Cannot open the video file" << endl;
				return -1;
			}

			double fps = cap.get(CV_CAP_PROP_FPS);
			cout << "Frame per seconds : " << fps << endl;
			Mat frame;

			int frameNumbers = cap.get(CV_CAP_PROP_FRAME_COUNT);
			int videoLength = frameNumbers / fps;


			while (1) {
				for (int j = 0; j < framenum-step; j++) {
					bool bSuccess = cap.read(frame);

					if (!bSuccess || flag >= frameNumbers) {
						cout << "\nNO MORE FRAMES" << endl;
						cout << "\nLAST FRAME: NUMBER=" << flag;
						cout << "" << endl;
					//	waitKey(1000);
						end++;
						break;
					}
					cout<<"Hola"<<endl;
					GaussianBlur(frame,frame, Size(3,3), 0,0);
					img[step + j] = frame.clone();
					flag++;
				}

				if (end == 1) {
					break;
				}
				cout<<"Hola2"<<endl;
				for (int j = 0; j < framenum; j++) {
					cvtColor(img[j], imgs_final[j], CV_BGR2GRAY);
					// imwrite( "../PROYECTO/MOTION/IMAG.jpg", imgs_final[j]);

				}

//				if(i>1){
//						cout<<"HOLA"<<endl;
//					 	Mat foreground;
//						cvtColor(res1, res1, CV_BGR2GRAY);
//						cvtColor(img[step], img[step], CV_BGR2GRAY);
//						imwrite( "../PROYECTO/MOTION/BACKGROUND.jpg", res1);
//						imwrite( "../PROYECTO/MOTION/I0.jpg", img[step]);
//						img[step].convertTo(img[step],CV_8UC1);
//						res1.convertTo(res1,CV_8UC1);
//
//						absdiff(res1,img[step],foreground);
//						imshow("FOREGROUND MASK",foreground);
//						waitKey();
//				   }


				opt.loadImages(imgs_final);
				//GENERAMOS LOS BOUNDING BOXES PARA CADA LISTA DE 10 FRAMES
				cout<<"Hola3"<<endl;

//				opt.denseflow_2_1();
				cout<<"Hola4"<<endl;
            	opt.skeleton_box();
            	cout<<"He llegado hasta aqui"<<endl;
            	opt.skeleton_tracking(1);
            	cout<<"Y hasta aqui tb"<<endl;
            	opt.KmeansClust_skeleton(16);
            	opt.draw_skeleton();
            	cout<<"final"<<endl;

//GENERAMOS LAS TRACKS
//			opt.denseflow_2_2(2);
//			//GENERAMOS UN PASO DE 5 IMAGENES
            	step = 5;
				//LAS ULTIMAS 5 IMAGENES PASAN A SER LAS PRIMERAS DE LA SIGUIENTE LISTA
				shiftImages(img, step);
			}
			if (waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
			{
				cout << "esc key is pressed by user" << endl;
				break;
				return 0;
			}

		}


		//PROCESO DE CLUSTERING
//		opt.sum_final = 0;
//		cout<<"CLUSTERING"<<endl;
//		opt.KmeansClust(80);
//		cout<<"REMOVING_CLUSTERS"<<endl;
//		opt.remove_clusters_seeds();
//		cout << "clusters size:" << opt.clusters.size();
//		cout << "" << endl;
//		waitKey(1000);
//		sprintf(filename,"../PROYECTO/MOTION/TRAINING_DATA/visualwords_walking_%i_file",training);
//		cout << "CLUSTERING AND REMOVEMENT FINISHED" << endl;
//		opt.writetoFile(filename);

		//---------------------------------------------------------------------------------------------------
		//GENERANDO MATRICES DE COVARIANZA

//		opt.covariance();
//		sprintf(file1,"../PROYECTO/MOTION/TRAINING_DATA/covariances_0_%i_file",training);
//		opt.writetoFile_covariance(file1);
//		cout << "COVARIANCES FINISHED" << endl;

		//------------------------------------------------------------------------------------------------

		opt.num_points = 0;
		start += 10;
		training++;
	}
	cout << "ALL FINISHED" << endl;
}
