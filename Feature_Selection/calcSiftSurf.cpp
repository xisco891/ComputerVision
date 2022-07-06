
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <iostream>
//#include "calcSiftSurf.h"
#include "opticalflow.h"
#include <eigen3/Eigen/Dense>
using namespace Eigen;




#undef _GLIBCXX_DEBUG
#include <stdio.h>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <vector>

using namespace cv;
using namespace std;


int main(int argc, char* argv[]){

	int imgnum=10;
	vector<KeyPoint> mkeypoints_aux(imgnum);
	vector<DMatch> matches_aux;
	vector<DMatch> matches1;
	vector<DMatch> matches2;
	Mat u1_aux[8];
	Mat v1_aux[8];
	vector<vector<KeyPoint> >keypoints;
	//	vector<Mat> imgs(60);
	vector<Mat> imgs_aux(imgnum);


	char path[200];

	vector<VectorXd > tracks_final;
	tracks_final.size()==0;


	opticalFlow opt;
	int j=0;
	do{
		int k=0;
		for (int i=j;i<(j+10);i++)
		{

			cout<<"i= "<<i<<endl;
			cout<<"HALLO1"<<endl;
			sprintf(path,"/home/xisco/ros/action/%03d.tif",i+1);
			cout<<"HALLO2"<<endl;
			puts(path);
			cout<<"HALLO3"<<endl;
			imgs_aux[k]=imread(path,CV_LOAD_IMAGE_GRAYSCALE);
			k++;


		}

//		calcSiftSurf sft1(imgs_aux);
//		sft1.SiftDetection();
//

//		opt.points.resize(10);
//		opt.points[0].resize(sft1.keypoints[0].size());
//
//		for(int k=0;k<sft1.keypoints[0].size();k++)
//		{
//			opt.points[0][k]=sft1.keypoints[0][k].pt;
//		}
		opt.loadImages(imgs_aux);
		opt.denseflow(5);

//		int size=opt.tracks.size()/2;
//
//		tracks_final.reserve(tracks_final.size()+size);
//		tracks_final.insert( tracks_final.end(), opt.tracks.begin(),opt.tracks.end()-size);
//		opt.tracks=tracks_final;
//		opt.KmeansClust(20);
		j+=5;


	}while(j<60);

	opt.KmeansClust(100);


}



