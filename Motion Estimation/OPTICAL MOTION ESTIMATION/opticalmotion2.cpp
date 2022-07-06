
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
	char file3[100];

	char filename2[100];
	int m = 0;

	vector<Mat> img(10);

	int training = 0;
	int start = 0;
	int h = 0;

	int framenum = 10;

	while (training < 6) {
		opticalFlow opt;
		cout << "tracks.size: " << opt.tracks.size();
		cout << "" << endl;
		waitKey(1500);

		for (int i = start; i < start + 3; i++) {
			cout << "I: " << i << endl;
			waitKey(1000);
			int flag = 0;
			int step = 0;
			int end = 0;
			h = i;
			if (i == 33 || i == 53) {
				h = i + 1;

			}
			sprintf(path, "/home/xisco/ros/action/%03d.avi", h);
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

				for (int j = 0; j < framenum - step; j++) {
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
					sprintf(file3, "../IMAGES/FOTO%03d.jpg", flag);
					imwrite(file3,img[step+j]);

					flag++;
				}
				if (end == 1) {
					break;
				}

				for (int j = 0; j < framenum; j++) {

					cvtColor(img[j], imgs_final[j], CV_BGR2GRAY);
					namedWindow("IMAGEN");

				}
				opt.loadImages(imgs_final);
				opt.denseflow_2_1();
				opt.denseflow_2_2(3);

				step = 5;
				shiftImages(img, step);

			}

			if (waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
					{
				cout << "esc key is pressed by user" << endl;
				break;
				return 0;
			}

		}

		opt.sum_final = 0;

		opt.KmeansClust(80);
		opt.remove_clusters_seeds();
		cout << "clusters size:" << opt.clusters.size();
		cout << "" << endl;
		waitKey(1000);
		sprintf(filename,
				"../trainingdata/right_left/minus_3/visualwords_walking_%i_file",
				training);
		cout << "CLUSTERING AND REMOVEMENT FINISHED" << endl;
		opt.writetoFile(filename);

		opt.covariance();

		sprintf(file1,
				"../trainingdata/right_left/minus_3/covariances_0_%i_file",
				training);
		opt.writetoFile_covariance(file1);
		cout << "COVARIANCES FINISHED" << endl;

//		opt.readFile(file1);
//		sprintf(file2,"../probesdata/minus_2/covariances_0_%i_file",training);
//		opt.distances_covariances(file1,file2);

//		cout<<""<<endl;
//		waitKey(2000);
//

		opt.num_points = 0;
		start += 10;
		training++;
//		cout<<"FINISHED"<<endl;
//		waitKey(4000);
	}
	cout << "ALL FINISHED" << endl;
}

namespace optical_flow {

opticalmotion::opticalmotion() {
	// TODO Auto-generated constructor stub

}

opticalmotion::~opticalmotion() {
	// TODO Auto-generated destructor stub
}

}
