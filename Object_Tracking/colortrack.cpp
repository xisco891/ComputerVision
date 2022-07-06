#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <iostream>
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

void shiftImages(vector<Mat>& vimg, int sn) {
	for (int i = 0; i < sn; i++) {
		vimg[i] = vimg[i + sn].clone();
	}
}

int main(int argc, char* argv[]) {

	int imgnum = 10;
	vector<Mat> imgs_aux(imgnum);
	vector<Mat> imgs_final(imgnum);

	char path[200];
	char filecovariance[100];
	char filename[100];
	char filename_tracks[100];
	char file1[100];
	char file2[100];
	char filecov[100];

	char filename2[100];
	int m = 0;

	vector<Mat> img(10);

	int training = 0;
	int start = 30;
	int h = 0;

	int framenum = 10;

	while (training < 6) {
		opticalFlow opt;

		for (int i = start; i < start + 3; i++) {
			int flag = 0;
			int step = 0;
			int end = 0;
			h = i;
			if (i == 33 || i == 53) {
				h = i + 1;

			}
		//	sprintf(path, "/home/usr/PROYECTO/MOTION/VIDEO/%03d.avi", h);
			sprintf(path, "/home/usr/PROYECTO/MOTION/VIDEO/TRACK_ROJO.mov");

			waitKey(2000);
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

			cout << "numframes:" << frameNumbers;
			cout << "" << endl;
			waitKey(2000);

			while (1) {

				for (int j = 0; j < framenum-step; j++) {
					bool bSuccess = cap.read(frame);

					if (!bSuccess || flag >= frameNumbers) {
						cout << "\nNO MORE FRAMES" << endl;
						cout << "\nLAST FRAME: NUMBER=" << flag;
						cout << "" << endl;
						waitKey(1000);
						end++;
						break;
					}
					img[step + j] = frame.clone();

					flag++;
				}
				if (end == 1) {
					break;
				}

				for (int j = 0; j < framenum; j++) {
					cvtColor(img[j], imgs_final[j], CV_BGR2GRAY);
//				    imwrite( "../IMAG.jpg", imgs_final[j]);

				}

				opt.loadImages(imgs_final);
				//GENERAMOS LOS BOUNDING BOXES PARA CADA LISTA DE 10 FRAMES
				opt.colortrack_flow_2_1();
				//GENERAMOS LAS TRACKS
				opt.colortrack_flow_2_2(3);
				//GENERAMOS UN PASO DE 5 IMAGENES
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
		opt.sum_final = 0;
		cout<<"CLUSTERING"<<endl;
		opt.KmeansClust(80);
		cout<<"REMOVING_CLUSTERS"<<endl;
		opt.remove_clusters_seeds();
		cout << "clusters size:" << opt.clusters.size();
		cout << "" << endl;
		waitKey(1000);
		sprintf(filename,"../PROYECTO/MOTION/TRAINING_DATA/visualwords_walking_%i_file",training);
		cout << "CLUSTERING AND REMOVEMENT FINISHED" << endl;
		opt.writetoFile(filename);

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
