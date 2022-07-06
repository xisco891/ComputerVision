
//INCLUDING VARIOUS LIBRARIES........
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/types_c.h"
#include "opencv2/core/version.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include  <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect/objdetect.hpp"

//GPU AND CUDA ACCELERATION
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
//#include "opencv2/cudalegacy.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudastereo.hpp"
#include "opencv2/cudawarping.hpp"
//#include "opencv2/cudev.hpp"
//#include "opencv2/gpu/gpu.hpp"

#include <iostream>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdio.h>

#include "VISUAL.h"
#include <math.h>
#include <string>
#include <iterator>

#include <cmath>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <algorithm>
#include <time.h>

using namespace Eigen;
using namespace cv;
using namespace std;
//using namespace cuda;
#undef _GLIBCXX_DEBUG

/*@Classes*/
class SIFT{
public:
	//SIFT VARIABLES
	vector<Mat> img;
	vector<Mat> src;
	vector<vector<Mat> > Scaled_Images;
	vector<vector<Mat> > DoG;
	vector<vector<Mat> > Max_Min;
	vector<vector<vector<vector<float> > > > orientations;

	Mat blured;
	double PI=3.141592653589793238463;

	Mat sec_deriv,first_deriv, h, hessian;
	Mat  first_deriv_transposed, one_one;


	class KeyPoint_SIFT{
	public:

		vector<vector<vector<Point2f> > > KPoints;
		vector<vector<vector<vector<float> > > > orientations;
		vector<vector<vector<float> > > Sift_Descriptor;
		int octave;
		int levels;
		void Locate_Maxima_Minima();

	};

	KeyPoint_SIFT KPoint;
	int pyramids=0; int levels=0;


	void invertImages();
	vector<Mat> copyOriginalImage();
	void loadImages(vector<Mat>& I_i);
	void Sift_Detection();
	void Scale_Images(int pyramids_number, int levels_number);
	void Difference_of_Gaussians();
	void Compute_Maxima_Minima(int i, int j, int l, int m);
	void Locate_Maxima_Minima();
	void Compute_KeyPoints(int i,int j,int k);
	void Finding_KeyPoints();
	void Resize_KPoints(int i, int j, int k);
	void Compute_Elimination_Edge_Responses(int i,int j,int k);
	void Elimination_Edge_Responses();
	void Compute_Orientations(int i, int j, int k);
	void KeyPoint_Orientations();
	void Sift_Descriptor();
	void Show_Images();
	time_t start,end;

};

void SIFT::loadImages(vector<Mat>& I_i){
	src.resize(I_i.size());
	for(int i=0;i<I_i.size();i++)
	{
		src[i]=I_i[i].clone();
		src[i].convertTo(src[i],CV_32FC1);
		normalize(src[i],src[i],0,255,NORM_MINMAX);
	}
}

void SIFT::invertImages(){

	for(int i=0;i<img.size();i++){
		bitwise_not(img[i],img[i]);
	}
}

vector<Mat> SIFT::copyOriginalImage(){
		vector<Mat> image;
		for(int i=0;i<src.size();i++){
			image.push_back(src[i]);
		}
		return image;
}

//////////////////////////////////////////////////////////////
////////////////////SIFT ALGORITHM/////////////////////////////
///////////////////////////////////////////////////////////////

//1.SCALE_IMAGES.


void SIFT::Scale_Images(int pyramids_number, int levels_number){
	Mat dst;
	int it=0;
	char window_pyr[200];char window[200];
	pyramids = pyramids_number;
	levels = levels_number;

	Scaled_Images = vector<vector<Mat> > (pyramids,vector<Mat> (levels));

	for(int i=0;i<src.size();i++){
		dst = src[i].clone();
		for(int j=0;j<pyramids;j++){
			it=0;
			Scaled_Images[j][it]= dst.clone();
			it++;
			for(int k=3;k<levels*2;k=k+2){
					blur(dst,blured,Size(k,k));
					Scaled_Images[j][it]=blured.clone();
					it++;
			}
			pyrDown(dst, dst, Size(dst.cols/2,dst.rows/2));
		}
	}
}

//////////////////////////////////////////////
//2 BLUR THEM AND APPLY DIFFERENCE OF GAUSSIAN.
//////////////////////////////////////////////
void SIFT::Difference_of_Gaussians(){

	DoG = vector<vector<Mat> > (pyramids,vector<Mat>(levels-1));
	Mat subtraction;
	for(int i=0;i<Scaled_Images.size();i++){
		for(int j=0; j<Scaled_Images[i].size()-1; j++){
			subtraction = Scaled_Images[i][j] - Scaled_Images[i][j+1];
			DoG[i][j]=subtraction.clone();
		}
	}
}

//3.LOCATE MAXIMA AND MINIMA. FIND SUBPIXEL MINIMA AND MAXIMA.


void SIFT::Compute_Maxima_Minima(int i, int j, int l, int m){
	int it=0;
	int maxima_element = 0,minima_element = 0;
	vector<int> elements;
	int num_elements=0;
	float a=0,b=0,c=0,center=0;
	Point iterations[9] = {
	Point(m-1,l-1),Point(m-1,l),Point(m-1,l+1),
	Point(m,l+1),Point(m+1,l+1),Point(m+1,l),
	Point(m+1,l-1),Point(m,l-1),Point(m,l)};
	num_elements = sizeof(iterations)/sizeof(iterations[0]);
	//CELL BY CELL------
	center=DoG[i][j].at<float>(m,l);
	elements.resize(0);
	for(int n=0;n<num_elements;n++){
		//IF AN ELEMENT IS NOT A MAXIMA OR A MINIMA THEN IT CANT BE CONSIDERED AS A KEYPOINT.
		//INCLUDE THE CENTER PIXEL FOR THE ABOVE AND THE BELOW BLURRED IMAGE OF THE OCTAVE.
		elements.push_back(DoG[i][j-1].at<float>(iterations[n]));
		elements.push_back(DoG[i][j+1].at<float>(iterations[n]));
		elements.push_back(DoG[i][j].at<float>(iterations[n]));
		if(n == num_elements-1){
			elements.pop_back();
		}
	}
	//FIND MAX-MIN FOR THE 3 CELLS
	maxima_element = *max_element(elements.begin(),elements.end());
	minima_element = *min_element(elements.begin(),elements.end());

	if(center > maxima_element || center < minima_element){
		Max_Min[i][j-1].at<float>(m,l) = 255;
		KPoint.KPoints[i][j-1].push_back(Point2f(m,l));
	}
}

void SIFT::Locate_Maxima_Minima(){

	float m=0,l=0;
	KPoint.KPoints = vector<vector<vector<Point2f> > > (DoG.size(),vector<vector<Point2f> >(2)) ;
	Max_Min = vector<vector<Mat> > (pyramids, vector<Mat>(2));

	for(int i=0;i<DoG.size();i++){
		for(int j=1;j<DoG[i].size()-1;j++){
			Max_Min[i][j-1]= Mat(DoG[i][j-1].rows, DoG[i][j-1].cols, CV_32FC1, Scalar::all(0));
			for(l=0;l<DoG[i][j].cols;l++){
				for(m=0;m<DoG[i][j].rows;m++){
					Compute_Maxima_Minima(i,j,l,m);
				}
			}
		}
	}
}



void SIFT::Compute_KeyPoints(int i,int j,int k){


		float sample_points=0;
		int num_elements=0;
		float resizing=0;

		float dx, dy, ds;
		float dxx,dxy,dyy,dss,dxs,dys;

		float m=KPoint.KPoints[i][j][k].x;
		float l=KPoint.KPoints[i][j][k].y;

		//FIRST-ORDER DERIVATIVE
		dx = (DoG[i][j+1].at<float>(m-1,l) - DoG[i][j+1].at<float>(m+1,l)) / 2;
		dy = (DoG[i][j+1].at<float>(m,l-1) - DoG[i][j+1].at<float>(m, l+1)) / 2;
		ds = (DoG[i][j].at<float>(m,l) - DoG[i][j+2].at<float>(m, l))/2;

		first_deriv.at<float>(0,0) = dx; first_deriv.at<float>(1,0) = dy; first_deriv.at<float>(2,0) = ds;

		//SECOND-ORDER DERIVATIVE
		dxx = DoG[i][j+1].at<float>(m + 1,l) + DoG[i][j+1].at<float>(m - 1, l) - 2 * DoG[i][j+1].at<float>(m, l);

		dyy = DoG[i][j+1].at<float>(m, l + 1) + DoG[i][j+1].at<float>(m, l - 1) - 2 * DoG[i][j+1].at<float>(m, l);

		dss = DoG[i][j+2].at<float>(m,l) + DoG[i][j].at<float>(m,l) - 2 * DoG[i][j+1].at<float>(m,l);

		dxy = (DoG[i][j+1].at<float>(m + 1, l + 1) - DoG[i][j+1].at<float>(m - 1, l + 1) -DoG[i][j+1].at<float>(m + 1, l - 1)+
																					DoG[i][j+1].at<float>(m - 1, l - 1)) / 2;


		dxs = (DoG[i][j+2].at<float>(m + 1, l) - DoG[i][j+2].at<float>(m - 1, l) - DoG[i][j].at<float>(m + 1, l) +
																					DoG[i][j].at<float>(m - 1, l)) / 2;

		dys = (DoG[i][j+2].at<float>(m, l + 1) - DoG[i][j+2].at<float>(m,l + 1) - DoG[i][j].at<float>(m,l + 1) +
																					DoG[i][j].at<float>(m,l - 1)) / 2;

		sec_deriv.at<float>(0, 0) = dxx; sec_deriv.at<float>(1, 0) = dxy; sec_deriv.at<float>(2, 0) = dxs;
		sec_deriv.at<float>(0, 1) = dxy; sec_deriv.at<float>(1, 1) = dyy; sec_deriv.at<float>(2, 1) = dys;
		sec_deriv.at<float>(0, 2) = dxs; sec_deriv.at<float>(1, 2) = dys; sec_deriv.at<float>(2, 2) = dss;
		sec_deriv.inv();

		h = sec_deriv*first_deriv*(-1);
		transpose(first_deriv, first_deriv_transposed);
		one_one = first_deriv_transposed*h;
		float extrema_value = abs(DoG[i][j].at<float>(m,l)+(1/2)*(one_one.at<float>(0,0)));


		if(abs(h.at<float>(0,0)) > 0.5){
			sample_points++;
			if(h.at<float>(0,0)<0.0){
				KPoint.KPoints[i][j][k].x -= 1;
			}
			else{
				KPoint.KPoints[i][j][k].x += 1;
			}
		}
		else if(abs(h.at<float>(0,1)) > 0.5){
			if(h.at<float>(0,1)<0.0){
				KPoint.KPoints[i][j][k].y -= 1;
			}
			else{
				KPoint.KPoints[i][j][k].y += 1;
			}
		}
		if(extrema_value < 0.03){
			KPoint.KPoints[i][j].erase(KPoint.KPoints[i][j].begin()+k);
	   }


}

void SIFT::Finding_KeyPoints(){

	sec_deriv = (Mat_<float>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	first_deriv = (Mat_<float>(3,1) << 0,0,0);
	first_deriv_transposed = (Mat_<float>(1,3) << 0,0,0);
	one_one = (Mat_<float>(1,1) << 0);
	h = (Mat_<float>(3,1) << 0,0,0);



	for(int i=0;i<KPoint.KPoints.size();i++){
		for(int j=0;j<KPoint.KPoints[i].size();j++){

			for(int k=0;k<KPoint.KPoints[i][j].size();k++){
				Compute_KeyPoints(i,j,k);

			}
		}
	}
}


void SIFT::Compute_Elimination_Edge_Responses(int i,int j,int k){


	float m,l;
	int num_elements=0;
	vector<float> eigen_values;
	float dx, dy, ds;float dxx,dxy,dyy,dss,dxs,dys;

	m=KPoint.KPoints[i][j][k].x;
	l=KPoint.KPoints[i][j][k].y;
	//FIRST-ORDER DERIVATIVE
	dx = (DoG[i][j+1].at<float>(m-1,l) - DoG[i][j+1].at<float>(m + 1,l)) / 2;
	dy = (DoG[i][j+1].at<float>(m, l-1) - DoG[i][j+1].at<float>(m, l+ 1)) / 2;
	ds = (DoG[i][j].at<float>(m,l) - DoG[i][j+2].at<float>(m,l)) / 2;

	first_deriv.at<float>(0, 0) = dx; first_deriv.at<float>(1, 0) = dy; first_deriv.at<float>(2, 0) = ds;

		//SECOND-ORDER DERIVATIVE...

	dxx = DoG[i][j+1].at<float>(m + 1,l) + DoG[i][j+1].at<float>(m - 1, l) -
		  2 * DoG[i][j+1].at<float>(m, l);

	dyy = DoG[i][j+1].at<float>(m, l + 1) + DoG[i][j+1].at<float>(m, l - 1) -
		  2 * DoG[i][j+1].at<float>(m, l);

	dss = DoG[i][j+2].at<float>(m,l) + DoG[i][j].at<float>(m,l) - 2 * DoG[i][j+1].at<float>(m,l);

	dxy = (DoG[i][j+1].at<float>(m + 1, l + 1) - DoG[i][j+1].at<float>(m - 1, l + 1) -
			DoG[i][j+1].at<float>(m + 1, l - 1)+
			DoG[i][j+1].at<float>(m - 1, l - 1)) / 2;


	dxs = (DoG[i][j+2].at<float>(m + 1, l) - DoG[i][j+2].at<float>(m - 1, l) - DoG[i][j].at<float>(m + 1, l) +
																					DoG[i][j].at<float>(m - 1, l)) / 2;

	dys = (DoG[i][j+2].at<float>(m, l + 1) - DoG[i][j+2].at<float>(m,l + 1) -
		   DoG[i][j].at<float>(m,l + 1) +
		   DoG[i][j].at<float>(m,l - 1)) / 2;



	sec_deriv.at<float>(0, 0) = dxx; sec_deriv.at<float>(1, 0) = dxy; sec_deriv.at<float>(2, 0) = dxs;
	sec_deriv.at<float>(0, 1) = dxy; sec_deriv.at<float>(1, 1) = dyy; sec_deriv.at<float>(2, 1) = dys;
	sec_deriv.at<float>(0, 2) = dxs; sec_deriv.at<float>(1, 2) = dys; sec_deriv.at<float>(2, 2) = dss;
	sec_deriv.inv();

	hessian.at<float>(0,0)=dxx; hessian.at<float>(0,1)=dxy; hessian.at<float>(1,0)=dxy; hessian.at<float>(1,1)=dyy;
	eigen(hessian, eigen_values);
	double max = *max_element(eigen_values.begin(),eigen_values.end());
	double min = *min_element(eigen_values.begin(),eigen_values.end());
	double r = max/min ;  //THE NEW APPROACH USES r=10
	double Tr = dxx + dyy;
	double ratio = (r+1)*(r+1)/r ;
	double div = (Tr*Tr)/determinant(hessian);

	if(div > ratio){
		KPoint.KPoints[i][j].erase(KPoint.KPoints[i][j].begin()+k);
	}

}

void SIFT::Elimination_Edge_Responses(){

	sec_deriv = (Mat_<float>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	first_deriv = (Mat_<float>(3,1) << 0,0,0);
	h = (Mat_<float>(3,3) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	hessian = (Mat_ <float>(2,2) << 0,0,0,0);

	for(int i=0;i<KPoint.KPoints.size();i++){
			for(int j=0;j<KPoint.KPoints[i].size();j++){
				for(int k=0;k<KPoint.KPoints[i][j].size();k++){
					Compute_Elimination_Edge_Responses(i,j,k);
				}
				cout<<"KEYPOINTS"<<endl;
			}
		}
}

void SIFT::Compute_Orientations(int i, int j, int k){
	float m,l;
	int num_elements=0;
	float orientation=0,magnitude=0;
	float gx=0,gy=0;

	//HISTOGRAM GENERATION.///////////////
	float bin_size = 36;
	float scale=0.0;
	float hist_bin=0;
	vector<float> bins(bin_size);
	int item=0;
	double max_bin = 0.0;
	float sum=0;
	///////////////////////////////////////
	m=KPoint.KPoints[i][j][k].x;
	l=KPoint.KPoints[i][j][k].y;

	Point iterations[9] = {Point(m-1,m-1),Point(m-1,l),Point(m-1,l+1),
						   Point(m,l+1),Point(m+1,l+1),Point(m+1,l),
						   Point(m+1,l-1),Point(m,l-1),Point(m,l)};
	num_elements = sizeof(iterations)/sizeof(iterations[0]);
	for(int n=0;n<num_elements;n++){

		gx= DoG[i][j].at<float>(iterations[n].x+1,iterations[n].y)-
				DoG[i][j].at<float>(iterations[n].x-1,iterations[n].y);
		gy= DoG[i][j].at<float>(iterations[n].x,iterations[n].y+1) -
				DoG[i][j].at<float>(iterations[n].x,iterations[n].y-1);



		orientation = abs(atan(gy/gx));
		orientation = orientation *(360/PI);
//						if(isnan(orientation)){cout<<"changing value"<<endl;orientation=0;}
		magnitude = (gx*gx) +(gy*gy);
		sum = magnitude * 1.5;
		item = floor(orientation/5);
		if(magnitude > 0){
			bins[item] += sum;
		}
	}
	max_bin = *max_element(bins.begin(),bins.end());
	KPoint.orientations[i][j][k].push_back(max_bin);

	for(l=0;l<bins.size();l++){
		if(bins[l] > 0.8*max_bin){
			KPoint.orientations[i][j][k].push_back(bins[i]);
		}
	}


}


void SIFT::KeyPoint_Orientations(){


		KPoint.orientations = vector<vector<vector<vector<float> > > > (pyramids, vector<vector<vector<float> > > (levels-3)) ;



		for(int i=0;i<KPoint.KPoints.size();i++){
			for(int j=0;j<KPoint.KPoints[i].size();j++){
				scale = 1.5*(1/(i+1));
				GaussianBlur(DoG[i][j],DoG[i][j],Size(3,3),scale,scale);
				KPoint.orientations[i][j].resize(KPoint.KPoints[i][j].size());
				for(int k=0;k<KPoint.KPoints[i][j].size();k++){
					Compute_Orientations(i,j,k);
				}
			}
		}
}

void SIFT::Sift_Descriptor(){


	int m=0;
	int n=0;
	float gx,gy;
	float magnitude=0;
	float orientation=0;
	float sum=0; int item=0;
	int start_cols=0,start_rows=0;

	cout<<"CREATING THE SIFT DESCRIPTOR"<<endl;
	vector<float> bins(8);
	KPoint.Sift_Descriptor = vector<vector<vector<float> > >(pyramids, vector<vector<float> >(levels-3));

	for(int i=0;i<DoG.size();i++){
		for(int j=0;j<2;j++){

			for(int k=0;k<KPoint.KPoints[i][j].size();k++){

					m=KPoint.KPoints[i][j][k].x-8;
					n=KPoint.KPoints[i][j][k].y-8;
					start_cols=m;
					start_rows=n;

					if(start_cols>0 && start_rows > 0){
						do{
							do{
								for(int o=start_cols;o<start_cols+4;o++){
									for(int p=start_rows;p<start_rows+4;p++){
										gx= DoG[i][j].at<float>(p+1,o) - DoG[i][j].at<float>(p-1,o);
										gy= DoG[i][j].at<float>(p,o+1) - DoG[i][j].at<float>(p,o-1);
										orientation = abs(atan(gy/gx));
										orientation = orientation *(360/PI);
										magnitude = (gx*gx) +(gy*gy);
										sum = magnitude * 1.5;
										item = floor(orientation/20);

										if(magnitude > 0){
											bins[item]=+sum;}

									}
								}
								start_cols += 4;
								}while(start_cols < m+14);
								start_rows += 4;
								start_cols=m;
								}while(start_rows < n+14);
								start_cols=0;
								start_rows=0;
								}
								for(int s=0;s<bins.size();s++){
									KPoint.Sift_Descriptor[i][j].push_back(bins[s]);
								}
						}
				}
	}
cout<<"END"<<endl;
}


void SIFT::Sift_Detection(){
			Scale_Images(4,5);
			cout<<"1"<<endl;
			Difference_of_Gaussians();
			cout<<"2"<<endl;
			Locate_Maxima_Minima();
			cout<<"3"<<endl;
			Finding_KeyPoints();
			cout<<"4"<<endl;
			Elimination_Edge_Responses();
			cout<<"5"<<endl;
			KeyPoint_Orientations();
			cout<<"6"<<endl;
			Sift_Descriptor();
			//THE FINAL SIFT DESCRIPTOR FOR AN IMAGE WILL BE THE ORIENTATIONS AND THE SIFT DESCRIPTOR THAT CONTAINS THE HISTOGRAM.
}
