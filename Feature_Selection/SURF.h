

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
#include "opencv2/gpu/gpu.hpp"

//CUDA PARALLEL PROCESSING
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
#undef _GLIBCXX_DEBUG

/*@Classes*/
class SURF{
public:
	//VARIABLES
	vector<Mat> img;
	vector<Mat> src;
	vector<Mat> Summed_Area_Table;
	vector<vector<Mat> > Scaled_Images;
	vector<vector<Mat> > Det_Matrices;
	vector<vector<Mat> > Max_elements;

	vector<Mat> Integral_Images;
	vector<vector<vector<vector<float> > > > orientations;
	int num_octaves;
	int num_pyramids;
	Mat blured;
	double PI=3.141592653589793238463;
	float L0;
	float L;
	float m0,l0;


	class KeyPoint_SURF{
	public:

		vector<vector<vector<Point2f> > > KPoints;
		vector<vector<vector<vector<float> > > > orientations;
		vector<vector<vector<float> > > Sift_Descriptor;
		int octave;
		int levels;
		void Locate_Maxima_Minima();

	};
	KeyPoint_SURF KPoint;
	int levels;

	void invertImages();
	vector<Mat> copyOriginalImage();
	void loadImages(vector<Mat>& I_i);
	void Surf_Detection_Method();
	bool Surf_is_Refined(int j, int k,int m, int l);
	void Surf_Integral_Image();
	void Surf_Hessian_Matrix(int pyramids,int octaves);
	void Surf_Non_Maxima_Supresion();
	void Surf_Compute_Features();
	void Surf_Orientation_Computation(int i,int m,int l, int L);

	void Surf_Matching();
	time_t start,end;
};

void SURF::loadImages(vector<Mat>& I_i){
	src.resize(I_i.size());
	for(int i=0;i<I_i.size();i++)
	{
		src[i]=I_i[i].clone();
		src[i].convertTo(src[i],CV_32FC1);
		normalize(src[i],src[i],0,255,NORM_MINMAX);
	}
}

void SURF::invertImages(){
	for(int i=0;i<img.size();i++){
		bitwise_not(img[i],img[i]);
	}
}

vector<Mat> SURF::copyOriginalImage(){
		vector<Mat> image;
		for(int i=0;i<src.size();i++){
			image.push_back(src[i]);
		}
		return image;
}



void SURF::Surf_Integral_Image(){

	float S00=0,S01=0,S10=0;

	Integral_Images = vector<Mat> (src.size(), Mat(src[0].rows, src[0].cols, CV_32FC1, Scalar::all(0)));
	Summed_Area_Table = vector<Mat> (src.size(), Mat(src[0].rows, src[0].cols, CV_32FC1, Scalar::all(0)));

	for(int i=0;i<src.size();i++){
		for(int j=0;j<src[i].cols;j++){
			for(int k=0;k<src[i].rows;k++){

				if( (k-1) < 0 ) {S00=0; S10=0;}
				else{S00=Integral_Images[i].at<float>(k-1,j-1);S10=Integral_Images[i].at<float>(k-1,j);}

				if( (j-1) < 0) {S00=0; S01=0;}
				else{S00=Integral_Images[i].at<float>(k-1,j-1);S01=Integral_Images[i].at<float>(k,j-1);}

				Summed_Area_Table[i].at<float>(k,j)=src[i].at<float>(k,j)+S10+S01-S00;
				Integral_Images[i].at<float>(k,j) = Integral_Images[i].at<float>(k-1,j-1) + Summed_Area_Table[i].at<float>(k,j) -
												    Integral_Images[i].at<float>(k,j-1) -
													Integral_Images[i].at<float>(k-1,j);
			}
		}
	}
}

//////////////////////////////////////////////
//2 HESSIAN MATRIX
//////////////////////////////////////////////

void SURF::Surf_Hessian_Matrix(int pyramids,int octaves){

	num_pyramids = pyramids;
	num_octaves = octaves;
	Mat hessian_matrix = (Mat_<float>(2,2) << 0,0,0,0);

	float result=0;
	float first=0,second=0,third=0,fourth=0;
	float a=0,b=0,c=0,d=0,e=0,f=0,g=0,h=0;
	float i=0,j=0,k=0,x=0;
	float q=0,n=0,o=0,p=0;

	float xf=0;
	float step=2;
	float init=3;
	float value=0;
	char path[200];
	float dxx,dxy,dyy;

	Size filter_size;
	Scaled_Images = vector<vector<Mat> >(num_pyramids, vector<Mat> (num_octaves));
	Det_Matrices = vector<vector<Mat> > (num_pyramids, vector<Mat> (num_octaves));

	for(int i=0;i<src.size();i++){
		for(int j=0;j<num_pyramids;j++){
			L=init;
			for(int k=0;k<num_octaves;k++){

				filter_size = Size(L*3,L*3);
				Scaled_Images[j][k]= Mat(Summed_Area_Table[i].rows, Summed_Area_Table[i].cols, CV_32FC1,Scalar::all(0));
				Det_Matrices[j][k] = Mat(Summed_Area_Table[i].rows, Summed_Area_Table[i].cols, CV_32FC1,Scalar::all(0));
				GaussianBlur( Summed_Area_Table[i], Scaled_Images[j][k],filter_size, 0.4*L,0.4*L);

				for(int l=0;l<src[i].cols;l++){
					for(int m=0;m<src[i].rows;m++){

						cout<<"--------------"<<endl;


						////////////////////////DXX/////////////////////////////////
						a=(m+(3*(L-1))/2); b=(l-(3*(L-1)/2)); c=m+(L-1); d=l-(L-1);
						e=(m+((L-1)/2)) ; f=(l-((L-1)/2)) ; g=m+(L-1) ; h=l-(L-1);

						if(b>-1 && d>-1){dxx = +Scaled_Images[j][k].at<float>(b,d);}
						if(b>-1 && c>-1){dxx = -Scaled_Images[j][k].at<float>(b,c);}
						if(a>-1 && d>-1){dxx = -Scaled_Images[j][k].at<float>(a,d);}
						if(a>-1 && c>-1){dxx = +Scaled_Images[j][k].at<float>(a,c);}
						result=0;

						if(f>-1 && h>-1){result = +Scaled_Images[j][k].at<float>(f,h);}
						if(f>-1 && g>-1){result = -Scaled_Images[j][k].at<float>(f,g);}
						if(e>-1 && h>-1){result = -Scaled_Images[j][k].at<float>(e,h);}
						if(g>-1 && h>-1){result = +Scaled_Images[j][k].at<float>(g,h);}

						dxx = dxx - 3*result;

						////////////////////////DYY////////////////////////////////
						if(d>-1 && b>-1){dyy = +Scaled_Images[j][k].at<float>(d,b);}
						if(d>-1 && a>-1){dyy = -Scaled_Images[j][k].at<float>(d,a);}
						if(g>-1 && f>-1){dyy = -Scaled_Images[j][k].at<float>(g,f);}
						if(g>-1 && e>-1){dyy = +Scaled_Images[j][k].at<float>(a,d);}

						result=0;
						if(h>-1 && f>-1){result = +Scaled_Images[j][k].at<float>(h,f);}
						if(h>-1 && e>-1){result = -Scaled_Images[j][k].at<float>(h,e);}
						if(g>-1 && f>-1){result = -Scaled_Images[j][k].at<float>(g,f);}
						if(g>-1 && e>-1){result = +Scaled_Images[j][k].at<float>(g,e);}
						dyy = dyy - 3*result;

						cout<<"\ndxx:"<<dxx<<endl;
						cout<<"\ndyy:"<<dyy<<endl;

						///////////////////////DXY///////////////////////////////

						a=m+L; b=l+1; c=m+L; d=l+1;
						e=m-1; f=l-L; g=m-1;h=l-L;
						i=m-1;j=l-L;k=c;x=d;
						q=c;n=d;o=g;p=h;

						first=0;second=0;third=0;fourth=0;

						cout<<"[b,d]:"<<"["<<b<<","<<d<<"]"<<endl;
						cout<<"[b,c]:"<<"["<<b<<","<<c<<"]"<<endl;
						cout<<"[a,d]:"<<"["<<a<<","<<d<<"]"<<endl;
						cout<<"[a,c]:"<<"["<<a<<","<<c<<"]"<<endl;


						for(int t=l;t<4;t++){
							for(int u=m;u<4;u++){
								cout<<"Scaled_Images["<<u<<","<<t<<"]:"<<Scaled_Images[j][k].at<float>(u,t)<<endl;
							}
						}

						cout<<"0"<<endl;
						if(b>-1 && d>-1){first = +Scaled_Images[j][k].at<float>(b,d);}
						cout<<"0.1"<<endl;
						if(b>-1 && c>-1){first = -Scaled_Images[j][k].at<float>(b,c);}
						cout<<"0.2"<<endl;
						if(a>-1 && d>-1){first = -Scaled_Images[j][k].at<float>(a,d);}
						cout<<"0.3"<<endl;
						if(a>-1 && c>-1){first = +Scaled_Images[j][k].at<float>(a,c);}

						cout<<"1"<<endl;
						cout<<"M:"<<m<<endl;
						cout<<"L:"<<l<<endl;


						if(f>-1 && h>-1){second = +Scaled_Images[j][k].at<float>(f,h);}
						if(f>-1 && g>-1){second = -Scaled_Images[j][k].at<float>(f,g);}
						if(e>-1 && h>-1){second = -Scaled_Images[j][k].at<float>(e,h);}
						if(g>-1 && h>-1){second = +Scaled_Images[j][k].at<float>(e,g);}

						if(j>-1 && x>-1){third = +Scaled_Images[j][k].at<float>(j,x);}
						if(j>-1 && k>-1){third = -Scaled_Images[j][k].at<float>(j,k);}
						if(i>-1 && x>-1){third = -Scaled_Images[j][k].at<float>(i,x);}
						if(i>-1 && k>-1){third = +Scaled_Images[j][k].at<float>(i,k);}

						if(n>-1 && p>-1){fourth = +Scaled_Images[j][k].at<float>(n,p);}
						if(n>-1 && o>-1){fourth = -Scaled_Images[j][k].at<float>(n,o);}
						if(q>-1 && p>-1){fourth = -Scaled_Images[j][k].at<float>(q,p);}
						if(q>-1 && o>-1){fourth = +Scaled_Images[j][k].at<float>(q,o);}

						dxy = first+second-third-fourth;

						cout<<"dxy:"<<dxy<<endl;
						////////////////////////////////////////////////////////////////

						hessian_matrix.at<float>(0, 0) = dxx;
						hessian_matrix.at<float>(1, 0) = dxy;
						hessian_matrix.at<float>(0, 1) = dxy;
						hessian_matrix.at<float>(1, 1) = dyy;

						float determinant = hessian_matrix.at<float>(0,0)*hessian_matrix.at<float>(1,1) -
											(0.9129*hessian_matrix.at<float>(0,1))*(0.9129*hessian_matrix.at<float>(0,1));
						determinant=determinant*(1/(L*L*L*L));
						Det_Matrices[j][k].at<float>(m,l) = determinant;

					}
				}
				Mat show_original; Scaled_Images[j][k].convertTo(show_original,CV_8U);
				imshow("ORIGINAL_BLURED_IMAGE",show_original);
				cout<<"\nSCALING:"<<L<<endl;
				Mat show_DoH;Det_Matrices[j][k].convertTo(show_DoH,CV_8U);
				imshow("DETERMINANT OF HESSIAN",show_DoH);
				waitKey(1000);
			    L += step;
			}
			init=init+step;
			step=2*step;
		}
	}
waitKey(0);
}

bool SURF::Surf_is_Refined(int j, int k, int m, int l){


	Mat Hessian_DoH = (Mat_<float>(3,3) << 0,0,0,0,0,0,0,0,0);
	Mat inverted = (Mat_<float>(3,3) << 0,0,0,0,0,0,0,0,0);
	Mat first_deriv = (Mat_<float>(3,1) << 0,0,0);
	Mat lambda = (Mat_<float>(3,1) << 0,0,0);

	float Hxx=0;float Hxy=0;float Hyy=0;
	float HxL=0;float HyL=0;float HLL=0;
	float dx=0,dy=0,dL=0;
	float p=pow(2,j-1);

	double max_lambda=0,min_lambda=0;

	dx=(1/(2*p))*(Det_Matrices[j][k].at<float>(m+p,l) - Det_Matrices[j][k].at<float>(m-p,l));
	dy=(1/(2*p))*(Det_Matrices[j][k].at<float>(m,l+p) - Det_Matrices[j][k].at<float>(m,l-p));
	dL=(1/(4*p))*(Det_Matrices[j][k+2*p].at<float>(m,l) - Det_Matrices[j][k-2*p].at<float>(m,l));
	first_deriv.at<float>(0,0)=dx;	first_deriv.at<float>(1,0)=dy;	first_deriv.at<float>(2,0)=dL;


	Hxx=Det_Matrices[j][k].at<float>(m+p,l) + Det_Matrices[j][k].at<float>(m-p,l)- 2*Det_Matrices[j][k].at<float>(m,l);
	Hxx *= 1/(p*p);


	Hyy=Det_Matrices[j][k].at<float>(m,l+p) + Det_Matrices[j][k].at<float>(m,l-p)- 2*Det_Matrices[j][k].at<float>(m,l);
	Hyy *= 1/(p*p);

	Hxy=Det_Matrices[j][k].at<float>(m+p,l+p) + Det_Matrices[j][k].at<float>(m-p,l-p)-Det_Matrices[j][k].at<float>(m-p,l+p)-
				Det_Matrices[j][k].at<float>(m+p,l-p);
	Hxy *= 1/(4*p*p);


	HxL=Det_Matrices[j][k].at<float>(m+p,l) + Det_Matrices[j][k].at<float>(m-p,l) -
		Det_Matrices[j][k+2*p].at<float>(m-p,l)- 2*Det_Matrices[j][k].at<float>(m,l);
	HxL *= 1/(8*p*p);

	HyL=Det_Matrices[j][k+2*p].at<float>(m,l+p) + Det_Matrices[j][k-2*p].at<float>(m,l-p) -
			Det_Matrices[j][k+2*p].at<float>(m,l-p)- Det_Matrices[j][k-2*p].at<float>(m,l+p);

	HyL *= 1/(8*p*p);

	HLL=Det_Matrices[j][k+2*p].at<float>(m,l) + Det_Matrices[j][k-2*p].at<float>(m,l)-2*Det_Matrices[j][k].at<float>(m,l);
	HLL *= 1/(4*p*p);

	Hessian_DoH.at<float>(0, 0) = Hxx; Hessian_DoH.at<float>(1, 0) = Hxy; Hessian_DoH.at<float>(2, 0) = HxL;
	Hessian_DoH.at<float>(0, 1) = Hxy; Hessian_DoH.at<float>(1, 1) = Hyy; Hessian_DoH.at<float>(2, 1) = HyL;
	Hessian_DoH.at<float>(0, 2) = HxL; Hessian_DoH.at<float>(1, 2) = HyL; Hessian_DoH.at<float>(2, 2) = HLL;


	cout<<"HESSIAN MATRIX:"<<Hessian_DoH<<endl;
	float det = determinant(Hessian_DoH);
	cout<<"DETERMINANT OF THE ORIGINAL HESSIAN MATRIX:"<<det<<endl;

	Hessian_DoH.inv(DECOMP_SVD);
	cout<<"HESSIAN MATRIX_INVERTED:"<<Hessian_DoH<<endl;

	Hessian_DoH = Hessian_DoH*(-1);
	cout<<"HESSIAN MATRIX:"<<Hessian_DoH<<endl;

	cout<<"\nFIRST_DERIV"<<first_deriv<<endl;

	lambda = Hessian_DoH * first_deriv;
	lambda /= det;

	lambda.at<float>(2,0) *= 1/2;

	cout<<"lambda:"<<lambda<<endl;
	minMaxLoc(lambda, &min_lambda, &max_lambda);
	cout<<"max_lambda"<<max_lambda<<endl;
	cout<<"min_lambda"<<min_lambda<<endl;


	if(abs(max_lambda) < p || abs(min_lambda) < p){
		m0=m+lambda.at<float>(0,0);
		l0=l+lambda.at<float>(1,0);
		L0=k+lambda.at<float>(2,0);
		cout<<"RETURN TRUE"<<endl;
		return true;
	}
	else{
		cout<<"RETURN FALSE"<<endl;
		return false;
	}
}

void SURF::Surf_Non_Maxima_Supresion(){

	int num_elements=0;
	float center=0;
	vector<float>elements(36);
	float maxima_element=0;
	float suma = 0;
	float n=0;
	float Number_elements=0;
	Max_elements = vector<vector<Mat> > (num_pyramids,vector<Mat> (num_octaves-2));
	KPoint.KPoints = vector<vector<vector<Point2f> > > (num_pyramids, vector<vector<Point2f> >(num_octaves-2));

	for(int i=0;i<src.size();i++){
		for(int j=0;j<num_pyramids;j++){
				for(int k=1;k<num_octaves-2;k++){
					cout<<"OCTAVE NUMBER:"<<k<<endl;
					Max_elements[j][k-1] = Mat(Summed_Area_Table[i].rows, Summed_Area_Table[i].cols, CV_32FC1, Scalar::all(0));
					for(int l=0;l<Det_Matrices[j][k].cols;l++){
							for(int m=0;m<Det_Matrices[j][k].rows;m++){

								Point iterations[9] = { Point(m-1,l-1),Point(m-1,l),Point(m-1,l+1),
								Point(m,l+1),Point(m+1,l+1),Point(m+1,l),
								Point(m+1,l-1),Point(m,l-1),Point(m,l)};
								num_elements = sizeof(iterations)/sizeof(iterations[0]);
								center=Det_Matrices[j][k].at<float>(m,l);
								elements.resize(0);
								suma=0;

								for(int n=0;n<num_elements;n++){
									elements.push_back(Det_Matrices[j][k-1].at<float>(iterations[n]));
									suma += Det_Matrices[j][k-1].at<float>(iterations[n]);
									elements.push_back(Det_Matrices[j][k+1].at<float>(iterations[n]));
									suma += Det_Matrices[j][k+1].at<float>(iterations[n]);
									elements.push_back(Det_Matrices[j][k].at<float>(iterations[n]));
									suma += Det_Matrices[j][k].at<float>(iterations[n]);

									if(n == num_elements-1){
										elements.pop_back();
										suma += center;
									}

								}

								maxima_element = *max_element(elements.begin(),elements.end());

								if( suma > 1000){
									  if(center > maxima_element){
										 if(Surf_is_Refined(j,k,m,l)){
											Number_elements++;
											Max_elements[j][k-1].at<float>(m,l) = maxima_element;
											KPoint.KPoints[j][L0].push_back(Point(m0,l0));
											}
									  }
								}
							}
							waitKey(100);
					}
					cout<<"NUMBER_ELEMENTS:"<<Number_elements<<endl;
					Number_elements = 0;

				}
			}
	}

}

//float SURF::Surf_Orientation_Computation(int i,int m,int l, int L){
//
//
//	int num_elements=0;
//	float number=0;
//	float dx=0;
//	float dy=0;
//	float p=2^(i-1);
//	float dev=0.4*L;
//	float x=0,y=0;
//	float max_orientation=0;
//	float orientation=0;
//	vector<float> orientations;
//	float Gaussian_Weight=0;
//	Mat dL = (Mat_<uchar>(2,0) << 0,0);
//	Mat dLt = (Mat_<uchar>(0,2) << 0,0);
//	Mat canonical_transposed = (Mat_<uchar>(2,0) << 1,0);
//	Mat orientation_result = (Mat_<uchar>(0,2)<< 0,0);
//
//	for(int i=l-6;i<l+6;i++){
//		for(int j=m-6;j<m+6;j++){
//			number = m*m+l*l;
//
//			if(number < 36 || number == 36){
//
//				dL.at<uchar>(0,0) = src[i].at<uchar>(j+p,i) - Det_Matrices[i][j-2*p].at<uchar>(j-p,i);
//				dL.at<uchar>(1,0)=src[i].at<uchar>(j,i+p) - Det_Matrices[i][j-2*p].at<uchar>(j,i-p);
//
//				transpose(dL,dLt);
//
//				float diff_x=m-j;
//				float diff_y=l-i;
//
//				Gaussian_Weight = (1/(2*PI))*exp((diff_x)+(diff_y)/2);
//				orientation = dLt * src[i].at<uchar>(j,i)*Gaussian_Weight;
//
//				y=static_cast<float>(orientation_result.at<uchar>(1,0));
//				x=static_cast<float>(orientation_result.at<uchar>(0,0));
//
//				cout<<"ORIENTATION<<"<<orientation<<endl;
//				orientation = atan2(y,x)*180/PI;
//
//				float num = x * 1;
//				float den = sqrt(x*y);
//
//				float angle = cos(num/den);
//
//				angle=1/angle;
//				cout<<"ANGLE:"<<angle<<endl;
//
//				orientations.push_back(orientation);
//			}
//			for(int k=0;k<39;k++){
//
//
//			}
//			//max_orientation = *max_element(orientations.begin(),orientations.end());
//		}
//	}
//	return max_orientation;
//}


//
//void SURF::Surf_Matching(){
//	float m,l;
//	int num_elements=0;
//	float orientation=0,magnitude=0;
//	float gx=0,gy=0;
//	KPoint.orientations = vector<vector<vector<vector<float> > > > (num_pyramids, vector<vector<vector<float> > > (levels-3)) ;
//
//	//HISTOGRAM GENERATION.
//	float bin_size = 36;
//	float scale=0.0;
//	float hist_bin=0;
//	vector<float> bins(bin_size);
//	int item=0;
//	double max_bin = 0.0;
//	float sum=0;
//
//			for(int i=0;i<KPoint.KPoints.size();i++){
//				for(int j=0;j<KPoint.KPoints[i].size();j++){
//					scale = 1.5*(1/(i+1));
//					GaussianBlur(Summed_Area_Table[i],Summed_Area_Table[i],Size(3,3),scale,scale);
//					KPoint.orientations[i][j].resize(KPoint.KPoints[i][j].size());
//
//					for(int k=0;k<KPoint.KPoints[i][j].size();k++){
//						m=KPoint.KPoints[i][j][k].x;
//						l=KPoint.KPoints[i][j][k].y;
//
//						Point iterations[9] = {Point(m-1,m-1),Point(m-1,l),Point(m-1,l+1),
//											   Point(m,l+1),Point(m+1,l+1),Point(m+1,l),
//											   Point(m+1,l-1),Point(m,l-1),Point(m,l)};
//						int num_elements = sizeof(iterations)/sizeof(iterations[0]);
//						for(int n=0;n<num_elements;n++){
//
//							gx= Summed_Area_Table[i].at<uchar>(iterations[n].x+1,iterations[n].y)-
//									Summed_Area_Table[i].at<uchar>(iterations[n].x-1,iterations[n].y);
//							gy= Summed_Area_Table[i].at<uchar>(iterations[n].x,iterations[n].y+1) -
//									Summed_Area_Table[i].at<uchar>(iterations[n].x,iterations[n].y-1);
//
//
//							orientation = abs(atan(gy/gx));
//							orientation = orientation *(360/PI);
//	//						if(isnan(orientation)){cout<<"changing value"<<endl;orientation=0;}
//							magnitude = (gx*gx) +(gy*gy);
//							sum = magnitude * 1.5;
//							item = floor(orientation/5);
//							if(magnitude > 0){
//								bins[item] += sum;
//							}
//						}
//						max_bin = *max_element(bins.begin(),bins.end());
//						KPoint.orientations[i][j][k].push_back(max_bin);
//
//						for(l=0;l<bins.size();l++){
//							if(bins[l] > 0.8*max_bin){
//								KPoint.orientations[i][j][k].push_back(bins[i]);
//							}
//						}
//					}
//				}
//			}
//	}


//void SURF::Surf_Descriptor(){
//
//
//	int m=0;
//	int n=0;
//	float gx,gy;
//	float magnitude=0;
//	float orientation=0;
//	float sum=0; int item=0;
//	int start_cols=0,start_rows=0;
//	vector<float> bins(8);
//	KPoint.Sift_Descriptor = vector<vector<vector<float> > >(pyramids, vector<vector<float> >(levels-3));
//
//	for(int i=0;i<DoG.size();i++){
//		for(int j=0;j<2;j++){
//
//			for(int k=0;k<KPoint.KPoints[i][j].size();k++){
//
//					m=KPoint.KPoints[i][j][k].x-8;
//					n=KPoint.KPoints[i][j][k].y-8;
//					start_cols=m;
//					start_rows=n;
//
//					if(start_cols>0 && start_rows > 0){
//						do{
//							do{
//								for(int o=start_cols;o<start_cols+4;o++){
//									for(int p=start_rows;p<start_rows+4;p++){
//										gx= DoG[i][j].at<uchar>(p+1,o) - DoG[i][j].at<uchar>(p-1,o);
//										gy= DoG[i][j].at<uchar>(p,o+1) - DoG[i][j].at<uchar>(p,o-1);
//										orientation = abs(atan(gy/gx));
//										orientation = orientation *(360/PI);
//										magnitude = (gx*gx) +(gy*gy);
//										sum = magnitude * 1.5;
//										item = floor(orientation/20);
//
//										if(magnitude > 0){
//											bins[item]=+sum;}
//
//									}
//								}
//								start_cols += 4;
//								}while(start_cols < m+14);
//								start_rows += 4;
//								start_cols=m;
//								}while(start_rows < n+14);
//								start_cols=0;
//								start_rows=0;
//								}
//								for(int s=0;s<bins.size();s++){
//									KPoint.Sift_Descriptor[i][j].push_back(bins[s]);
//								}
//						}
//				}
//	}
//cout<<"END"<<endl;
//}
//


void SURF::Surf_Detection_Method(){

			Surf_Integral_Image();
			Surf_Hessian_Matrix(4,4);
			Surf_Non_Maxima_Supresion();
//			Surf_KeyPoint_Orientation();
//			Surf_Descriptor();


			//1.COMPUTE INTEGRAL IMAGE OF A GIVEN IMAGE: OVER A PIXEL LOCATION.


			//2.COMPUTE HESSIAN MATRIX. -> RELY ON THE DETERMINANT OF THE HESSIAN.
			//2.1.As Gaussian filters are non-ideal in any case, and given Lowe’s success with LoG
				//approximations, we push the approximation even further with box filters.
				//These approximate second order Gaussian derivatives, and can be
				//evaluated very fast using integral images, independently of size

				//Det(H)=Dxx*Dyy-(0.9Dxy)^2;

			//2.2 THE FILTER RESPONSES ARE NORMALISED WITH RESPECT OF THE MASK SIZE.
			//2.3Due to the use of box filters and integral images, we do not have to iteratively apply the same filter to the output of a
			//previously filtered layer, but instead can apply such filters of any size at exactly
			//the same speed directly on the original image.

			//2.4 The output of the filter is considered as the initial scale layer, to which we will refer as
				//scale s=1. The following layers are obtained by filtering the image with gradually bigger masks, taking
				//into account the discrete nature of integral images and the specific structure of
				//our filters. Specifically, this results in filters of size 9×9, 15×15, 21x21, 27×27,
				//etc. Hence, for each new octave, the filter size increase is doubled (going
				//from 6 to 12 to 24). Simultaneously, the sampling intervals for the extraction of
				//the interest points can be doubled as well.

			// 2.5 In order to  localise  interest points  in  the  image  and over scales, a  non-
				 //maximum suppression in a 3×3×3 neighbourhood is applied. The maxima
				 //of the determinant of the Hessian matrix are then interpolated in scale and
			     //image space with the method proposed

			//3.COMPUTE FEATURE DESCRIPTOR.

				//3.1 ORIENTATION ASSIGNMENT.
					//COMPUTE HAAR-WAVELET RESPONSES IN X AND Y.
					//WE COMPUTE IT AROUND A CIRCULAR NEIGHBOURHOOD WITH RADIUS 6s, WITH S THE SCALE AT WHICH
					//THE INTEREST POINT WAS DETECTED.
					//THE WAVELET RESPONSES ARE COMPUTED AT THE CURRENT SCALE S.
//IMPORTANT NOTE	//AT HIGH SCALES THE SIZE OF THE WAVELETS IS BIG -> WE USE INTEGRAL IMAGES FOR FAST FILTERING.
					//
					//Once the wavelet responses are calculated and WEIGHTED WITH A GAUSSIAN (σ=2.5s)
					//centered at the interest point, the responses are represented as vectors in a
					//space with the horizontal response strength along the abscissa and the vertical
					//response strength along the ordinate. The dominant orientation is estimated by
					//calculating the sum of all responses within a sliding orientation window covering
					//an angle of π3. The horizontal and vertical responses within the window are
					//summed. The two summed responses then yield a new vector. The longest such
					//vector lends its orientation to the interest point. The size of the sliding window
					//is a parameter, which has been chosen experimentally. Small sizes fire on single
					//dominating wavelet responses, large sizes yield maxima in vector length that are
					//not outspoken. Both result in an unstable orientation of the interest region. Note
					//the U-SURF skips this step.


			//4.FEATURE DESCRIPTION

				//4.1 DESCRIPTOR COMPONENTS.

					//For the extraction of the descriptor, the first step consists of constructing a
					//square region centered around the interest point, and oriented along the orienta-
					//tion selected in the previous section. For the upright version, this transformation
					//is not necessary. The size of this window is 20s. The region is split up regularly into smaller 4×4
					//square sub-regions. This keeps important spatial information in. For each sub-region, we compute a few
					//simple features at 5×5 regularly spaced sample points. For reasons of simplicity,
					//we call dx the Haar wavelet response in horizontal direction and dy the Haar
					//wavelet response in vertical direction (filter size 2s). ”Horizontal” and ”vertical”
					//here is defined in relation to the selected interest point orientation. To increase
					//the robustness towards geometric deformations and localisation errors, the re-
					//sponses dx and dy are first weighted with a Gaussian (σ=3.3s) centered at the interest point.


			//5.MATCHING STAGE.

			//In the matching stage, we only compare features if they have the same type of contrast. Hence,
			//this minimal information allows for faster matching and gives a slight increase in performance.
			//The matching is carried out as follows. An interest point in the test image
			//is compared to an interest point in the reference image by calculating the Eu-
			//clidean distance between their descriptor vectors. A matching pair is detected,
			//if its distance is closer than 0.7 times the distance of the second nearest neigh-
			//bour. This is the nearest neighbour ratio matching strategy

}
