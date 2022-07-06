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
#include "opencv2/ximgproc.hpp"
#include "opencv2/ximgproc/edge_filter.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include "opencv2/ximgproc/sparse_match_interpolator.hpp"
#include "opencv2/ximgproc/structured_edge_detection.hpp"
#include "opencv2/ximgproc/seeds.hpp"
#include "opencv2/ximgproc/segmentation.hpp"
#include "opencv2/ximgproc/fast_hough_transform.hpp"
#include "opencv2/ximgproc/estimated_covariance.hpp"
#include "opencv2/ximgproc/weighted_median_filter.hpp"
#include "opencv2/ximgproc/slic.hpp"
#include "opencv2/ximgproc/lsc.hpp"
#include "opencv2/ximgproc/paillou_filter.hpp"
#include "opencv2/ximgproc/fast_line_detector.hpp"
#include "opencv2/ximgproc/deriche_filter.hpp"
/////////////////////////////////////////////
//////////////////////CUDA//////////////////
#include "opencv2/cudaobjdetect.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


/////////////////////////////
//#include "calcSiftSurf.h"
#include "opticalflow.h"
//////////////////////////////77

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

///////////////////////END_INCLUDES///////////////////////////////////////////
Mat src, src_gray,dst,detected_edges;
int edgeThresh = 1, lowThreshold, ratio = 3, kernel_size = 3, scale = 1, delta = 0, thresh = 100, max_thresh = 255;
int const max_lowThreshold = 100;
RNG rng(12345);
////////////////////////////////////////////////////////////////////////
Mat myHarris_dst, myHarris_copy, Mc, myShiTomasi_dst, myShiTomasi_copy;
int myShiTomasi_qualityLevel = 50, myHarris_qualityLevel = 50, max_qualityLevel = 100;
double myHarris_minVal, myHarris_maxVal, myShiTomasi_minVal, myShiTomasi_maxVal;
const char* myHarris_window = "My Harris corner detector";
const char* myShiTomasi_window = "My Shi Tomasi corner detector";


///////BACK-PROJECTION////////////////////////////////////////////
int bins = 25;
///////FIND-CONTOURS/////////////
/////////////////////////////////
char window_Find_Contours[200];


/*@Classes*/

class CVision{
public:
	vector<Mat> img;
	vector<Mat> src;
	vector<vector<Point> >  contours;
	vector<vector<vector<Point> > > vect_contours;
	Rect bounding_rect;
	vector<Vec4i> hierarchy;
	vector<int> index_biggest_contour;
	Scalar color;
	static const int MAX_KERNEL_LENGTH = 5;
	vector<vector<Point2f> > points;
	char path[200];

	vector<Mat> LBP;
	vector<MatND> vect_Hist;

	//SIFT VARIABLES
	vector<vector<Mat> > Scaled_Images;
	vector<vector<Mat> > DoG;
	vector<vector<Mat> > Max_Min;
	Mat blured;

	//HOG VARIABLES

	Mat gx[3],gy[3];
	Mat GX,GY;
	Mat magnit, angle;


	class KeyPoint_SIFT{
	public:

		vector<vector<vector<Point2f> > > KPoints;
		vector<vector<vector<vector<float> > > > orientations;
		vector<vector<float> > Sift_Descriptor;
		int octave;
		int levels;
		void Locate_Maxima_Minima();

	};


	vector<vector<vector<vector<float> > > > orientations;
	KeyPoint_SIFT KPoint;
	int pyramids=0; int levels=0;

	//HEADERS///////////////////
	void Menu();
	void Initialize();
	void Spatial_Filtering();
	void Morphological_Processing();
	void Image_Segmentation();
	void Store_Images();

	//FUNCTIONALITIES///////////
	void invertImages();
	vector<Mat> copyOriginalImage();

	//INITIALIZATION//////////////
	void loadImages(vector<Mat>& I_i);
	//PROJECTS/////////////


	//SPATIAL FILTERING & TRANSFORMATIONS
		void blur_filtering();
		void bilateral_filtering();
		void gaussian_filtering();
		void box_filtering();
		void median_filtering();
		void roberts_cross_filtering();
		void Histogram_Equalization();
		void Histogram_Matching();
		void Histogram_Processing();
		void Histogram_Local_Processing();
		void Histogram_Statistics();
		void Object_Recognition();
		void Haar_Cascade_Detection();
		void Generate_LBP_Features();
		void Generate_Histograms(Mat cod_Hist);
		void Concatenate_Histograms();
		void Sift_Detection();
		void Scale_Images(int pyramids_number, int levels_number);
		void Difference_of_Gaussians();
		void Locate_Maxima_Minima();
		void Finding_KeyPoints();
		void Resize_KPoints(int i, int j, int k);
		void Elimination_Edge_Responses();
		void KeyPoint_Orientation();
		void Sift_Descriptor();
		void Show_Images();
		///////SURF METHOD//////////
		void Surf_Detection_Method();
		void Surf_Scale_Images(int pyramids_number, int levels_number);
		void Surf_Difference_of_Gaussians();
		void Surf_Locate_Maxima_Minima();
		void Surf_Finding_KeyPoints();
		void Surf_Resize_KPoints(int i, int j, int k);
		void Surf_Elimination_Edge_Responses();
		void Surf_KeyPoint_Orientation();
		void Surf_Descriptor();

		///////HOG METHOD///////////
		void HoG_Detection_Method();
		void Gradient_Images();
		void Gradient_Color();
		void Histogram_of_Gradients();
		//void write_Data(MatND histogram,int i);


		time_t start,end;


	//BASIC-MORPHOLOGY ALGORITHMS.
		//CONVEX-HULL
		//CLOSE CONTOURS:approxPolyDP;
		//CREATE A BOUNDING RECT AROUND A CONTOUR AREA;boundingRect();
		//CHECK CONVEXITY -> isContourConvex();
		//FIT A LINE -> fitline();
		//CHECK SIZES OF CONTOUR AREAS OF 2D;
		//MATCH SHAPES; -> matchShapes();
		//TEST POLYGONS; -> PointPolygonTest();



	//MORPHOLOGY PROCESSING
		vector<Mat> Dilation(int,void*);
		vector<Mat> Erosion(int,void*);
		vector<Mat> Opening(int,void*);
		vector<Mat> Closing(int,void*);
		void Thinning(int option);
		void Morphological_Gradient(int, void*);
		void Top_Hat(int, void*);
		void Black_Hat(int, void*);

	//IMAGE SEGMENTATION.

	  //POINT,LINE AND EDGE DETECTION.
		void Sobel_Mode();
		void Laplacian_Mode();
		void Laplacian_of_Gaussian();
		void Zero_Crossings(Mat result,int i);
		void Canny_Mode();
		void Finding_Contours();
		void Draw_Contours();
		void Analyze_Contours();
		void Resize_Contours(int i, int j);
		void findCorners();
		void Harris_Corner_Mode();
		void HarrisCorner(Mat image, string window_HC,int i);
		void Build_Detector();
		void Hough_Line_Method();
		void Hough_Line_Circle();
		void Sobel_Canny_Find_Contours();
		void Sobel_Harris_Hough_Line();
		void Watershed_Distance_Transform();
		void Fusing_Methods();

	//THRESHOLDING_METHODS
		void Thresholding_Methods();
		void Simple_Threshold();
		void Otsu_Threshold();
		void Otsu_Gaussian_Threshold();
		void Adaptive_Threshold(int choice);



};

void CVision::loadImages(vector<Mat>& I_i){
	src.resize(I_i.size());
	for(int i=0;i<I_i.size();i++)
	{
		src[i]=I_i[i].clone();
	}
}

void CVision::invertImages(){

	for(int i=0;i<img.size();i++){
		bitwise_not(img[i],img[i]);
	}
}

vector<Mat> CVision::copyOriginalImage(){
		vector<Mat> image;
		for(int i=0;i<src.size();i++){
			image.push_back(src[i]);
		}
		return image;
}


template <typename T>
int signum(const T& val) {
	int value = 0;
	for(int j=0; j<val.size(); j++){

		if(val[j] > 0){
			val[j]=1;
		}
		else if(val[j] == 0){
			val[j] = 0;
		}
		else{
			val[j]=-1;
		}
		//IF VAL IS COMPLEX THEN DO THIS -> x./abs(x);
	}

	return val;
}


// STORE IMAGES.

void CVision::Store_Images(){

	char path_one[200];
	string writeFile;
	char name[200];
	char it_path[200];

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);



	printf("PLEASE SELECT THE PATH WHERE TO STORE THE IMAGES: ");
	scanf("%s",path_one);


	printf("\nNOW SELECT THE NAME OF THE FILE: ");
	scanf("%s",name);

	for(int i=0; i<img.size();i++){

		sprintf(it_path, "_%i.png",i);
		writeFile.append(path_one);
		writeFile.append("/");
		writeFile.append(name);
		writeFile.append(it_path);
		imwrite(writeFile,img[i],compression_params);
		writeFile = "";
	}

}





////////////////////////SEGMENTATION METHODS////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////1.SOBEL METHOD///////////////////////////////

////////////////////////////////8.HISTOGRAM OF GRADIENTS.////////////////////////////////////////////


void HOG(vector<Mat> image){


	for(int i=0;i<image.size();i++){


	}

}



////////////////////////////////9.BACK_PROJECTION///////////////////////////////


void Hist_and_Backproj(Mat hue)
	{
	  MatND hist;
	  int histSize = MAX( bins, 2 );
	  float hue_range[] = { 0, 180 };
	  const float* ranges = { hue_range };

	  /// Get the Histogram and normalize it
	  calcHist( &hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false );
	  normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );

	  /// Get Backprojection
	  MatND backproj;

	  calcBackProject( &hue, 1, 0, hist, backproj, &ranges, 1, true );

	  /// Draw the backproj
	  imshow( "BackProj", backproj);

	  /// Draw the histogram
	  int w = 400; int h = 400;
	  int bin_w = cvRound( (double) w / histSize );
	  Mat histImg = Mat::zeros( w, h, CV_8UC3 );

	  for( int i = 0; i < bins; i ++ )
	     {
		  rectangle( histImg, Point( i*bin_w, h ), Point( (i+1)*bin_w, h -
				  cvRound( hist.at<float>(i)*h/255.0 ) ), Scalar( 0, 0, 255 ), -1 );
	     }

	  imshow( "Histogram", histImg );
	  waitKey(200);

	}



void back_projection(vector<Mat> image){

	Mat src; Mat hsv; Mat hue;
	Mat rgb;
	int bins = 25;

	for(int i=0;i<image.size();i++){


		cvtColor( image[i], rgb, CV_GRAY2RGB );
		cvtColor(rgb, hsv, CV_BGR2HSV);
		/// Use only the Hue value
		hue.create( hsv.size(), hsv.depth() );
		int ch[] = { 0, 0 };
		imshow("HSV IMAGE",hsv);
		imshow("HUE BEFORE MIX-UP", hue);
		mixChannels( &hsv, 1, &hue, 1, ch, 1 );
		imshow("HUE AFTER mixChannels", hue);
		waitKey(0);
		Hist_and_Backproj(hue);
		/// Show the image
		imshow( "Source image", image[i] );

		}
	waitKey(0);
}



//////////////////////////////////////////////////////////////////////
///////////////////FUSED METHODS FOR SEGMENTATION////////////////////
///////////////////////////////////////////////////////////////////////


void CVision::Sobel_Harris_Hough_Line(){

	Mat fg;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat src, src_gray;
	Mat grad;
	char path[200];
    int ddepth = CV_16S;


	//HARRIS_CORNER_DETECTION

    Mat dst, dst_norm, dst_norm_scaled;
    int blockSize = 1;
    int apertureSize = 5;
    double k = 0.04;

    //HOUGH-LINE
	Mat  cdst;

	//LAPLACE
	int kernel_size = 3;
	char window_laplace[200];
	int c;
	Mat abs_dst;


	for(int i=0; i<img.size(); i++){


					/////////////LAPLACE//////////////////

					Laplacian( img[i], dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
					convertScaleAbs( dst, abs_dst );
					sprintf(window_laplace,"Laplacian_Raw_Image_%d",i);
					imshow( window_laplace, abs_dst );
					waitKey(1000);

					////////////////////////////////////////////////////////

					//HARRIS

					dst = Mat::zeros( abs_dst.size(), CV_32FC1 );
					cornerHarris( abs_dst, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
					normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
					convertScaleAbs( dst_norm, dst_norm_scaled );
					imshow("AFTER SOBEL_AND_HARRIS", dst_norm_scaled);
					waitKey(2000);


					////////////////////////////////////////////////////////

					//CANNY

					Canny(dst_norm_scaled, dst, 50, 200, 3);
					cvtColor(dst, cdst, CV_GRAY2BGR);
					vector<Vec4i> lines;
					HoughLinesP(dst, lines, 1, CV_PI/180, 20, 5 , 8 );

					for( size_t j = 0; j < lines.size(); j++ )
					{
							Vec4i l = lines[j];
							line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
					 }

					 imshow("source", img[i]);
					 imshow("detected lines", cdst);
					 waitKey(2000);
					}
}



void CVision::Sobel_Canny_Find_Contours(){

	Mat fg;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat src, src_gray;
	Mat grad;
	char path[200];
    int ddepth = CV_16S;

    //HOUGH-LINE
	Mat  cdst;


	for(int i=0; i<img.size(); i++){


					/// Gradient X
					Sobel( img[i], grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
					/// Gradient Y
					Sobel( img[i], grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

					convertScaleAbs( grad_x, abs_grad_x );
					convertScaleAbs( grad_y, abs_grad_y );
					addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
					sprintf(path, "Sobel_Image_%d",i);
					cout<<"1"<<endl;
					imshow( path,grad);
					waitKey(2000);

					/////////////////////////
					Mat canny_output;
					vector<vector<Point> > contours;
					vector<Vec4i> hierarchy;
					thresh=100;
					///DETECT EDGES USING CANNY
					Canny( grad, canny_output, 50, 200 , 3 ); //20,225
					imshow("CANNY_DETECTED_EDGES",canny_output);
					waitKey();
					///FIND COUNTOURS
					findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

					///DRAW CONTOURS
					Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
					for( int i = 0; i< contours.size(); i++ )
					 {
					   Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
					   drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
					 }

					///SHOW IN WINDOW
					imshow( "CONTOURS", drawing );
					waitKey(2000);
	}
}


void DOG(vector<Mat> image){

	for(int i=0; i<image.size();i++){


	}
}


void compute_histogram(vector<Mat> image){

		    // Set histogram bins count
	    int bins = 256;
	    int histSize[] = {bins};
	    // Set ranges for histogram bins
	    float lranges[] = {0, 256};
	    const float* ranges[] = {lranges};
	    // create matrix for histogram
	    Mat hist;
	    int channels[] = {0};
	    char window[200];

	    // create matrix for histogram visualization
	    int const hist_height = 256;
	    Mat3b hist_image = cv::Mat3b::zeros(hist_height, bins);

	    for(int i=0; i<image.size();i++){

			calcHist(&image[i], 1, channels, Mat(), hist, 1, histSize, ranges, true, false);

			double max_val=0;
			minMaxLoc(hist, 0, &max_val);

			// visualize each bin
			for(int b = 0; b < bins; b++) {
				float const binVal = hist.at<float>(b);
				int const height = cvRound(binVal*hist_height/max_val);
				line(hist, Point(b, hist_height-height),Point(b, hist_height),Scalar::all(255));
			}
			sprintf(window,"HISTOGRAM FOR RAWIMAGE_%d",i);
			imshow(window, hist_image);
			waitKey(2000);

	    }
	    waitKey();
	}


/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////THRESHOLDING METHODS////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

void CVision::Thresholding_Methods(){

	int option;
	int adaptive_option;
	bool go_back=true;

	string options[5] = {"1.SIMPLE_THRESHOLDING......",
						 "2.OTSU THRESHOLDING........",
						 "3.OTSU + GAUSSIAN..........",
						 "4.ADAPTIVE THESHOLDING.....",
						 "5.MENU"
	};

	while(go_back){
	cout<<"THRESHOLDING OPTIONS."<<endl;
	int num_elem = sizeof(options)/sizeof(options[0]);

	for(int i=0;i<num_elem;i++){
		cout<<""<<options[i]<<""<<endl;
	}
	printf("\nCHOOSE YOUR OPTION:");
	scanf("%i",&option);

		switch(option){
			case 1:Simple_Threshold();break;
			case 2:Otsu_Threshold();return;
			case 3:Otsu_Gaussian_Threshold();return;
			case 4:printf("\nCHOOSE BETWEEN MEAN_C(0) OR GAUSSIAN_METHOD(1):");scanf("%i",&adaptive_option);
			Adaptive_Threshold(adaptive_option);break;
			case 5:go_back=false;break;
		}
	}
}


void CVision::Simple_Threshold(){
	Mat simple;
	for(int i=0; i<img.size(); i++){
		threshold(img[i],simple,50,200,THRESH_BINARY);
		imshow("SIMPLE THRESHOLDING",simple);
		waitKey(1000);
		img[i] = simple.clone();
	}
}


void CVision::Otsu_Threshold(){
	Mat otsu;
	for(int i=0; i<img.size(); i++){

		threshold(img[i],otsu,0,255,THRESH_BINARY+THRESH_OTSU);
		imshow("OTSU METHOD",otsu);
		waitKey(1000);
		img[i] = otsu.clone();
	}

}

void CVision::Otsu_Gaussian_Threshold(){

	Mat otsu_gaussian;

	for(int i=0; i<img.size(); i++){
		GaussianBlur(img[i],otsu_gaussian,Size(5,5),0,0);
		threshold(otsu_gaussian,otsu_gaussian,0,255,THRESH_BINARY+THRESH_OTSU);
		imshow("OTSU + GAUSSIAN METHOD",otsu_gaussian);
		waitKey(100);
		img[i] = otsu_gaussian.clone();


	}
}



void CVision::Adaptive_Threshold(int choice){

		Mat mean_c, gaussian_c;
		median_filtering();

		for(int i=0; i<img.size();i++){

			threshold(img[i],img[i],127,255,THRESH_BINARY);

			if(choice == 0){
			adaptiveThreshold(img[i],mean_c, 255, ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY, 9,2);
			imshow("MEAN_C METHOD",mean_c);
			}
			else{
			adaptiveThreshold(img[i],gaussian_c, 255, ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY, 9,2);
			imshow("GAUSSIAN_METHOD", gaussian_c);
			}
			waitKey(100);
		}
}



//////////////////////////////////////////////////////////////////////////////
////////////////////////FREQUENCY DOMAIN FILTERING.///////////////////////////
/////////////////////////////////////////////////////////////////////////////


///////////////////SELECTION OF THE METHOD///////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void CVision::Spatial_Filtering(){

	int option;
	bool go_back=1;


	while(go_back){


	string options[8] = {"1.BOX FILTERING.",
					     "2.GAUSSIAN_FILTERING",
					     "3.MEDIAN_FILTERING",
						 "4.BLUR_FILTERING",
					     "5.BILATERAL_FILTERING",
					     "6.ROBERT_CROSS_FILTERING",
						 "7.HISTOGRAM PROCESSING",
						 "8.MENU"
	};




	int num_elem = sizeof(options)/sizeof(options[0]);

	for(int i=0;i<num_elem;i++){
		cout<<""<<options[i]<<""<<endl;
	}

	printf("CHOOSE YOUR OPTION:");
	scanf("%i",&option);


		switch(option){
					case 1:box_filtering();break;
					case 2:gaussian_filtering();break;
					case 3:median_filtering();break;
					case 4:blur_filtering();break;
					case 5:bilateral_filtering();break;
					case 6:roberts_cross_filtering();break;
					case 7:Histogram_Processing();break;
					case 8:go_back=0;break;
		}
		waitKey(1000);
	}
}


/////////////////////////FILTERS/////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

//////////////////////BOX FILTERING/////////////////////////////////////////

void CVision::box_filtering(){
	Mat box;
	for(int i=0;i<img.size();i++){

		for ( int j = 1; j < MAX_KERNEL_LENGTH; j = j + 2 ){
			blur( img[i], box, Size( j, j ), Point(-1,-1) );
			imshow("BOX FILTERING", box);
			waitKey(500);

		}
		img[i] = box.clone();
	}
}

//////////////////////GAUSSIAN FILTERING/////////////////////////////////////


void CVision::gaussian_filtering(){
	Mat gaussian;

	for(int i=0;i<img.size();i++){
			GaussianBlur(img[i],gaussian,Size(5,5),0,0);
			imshow("GAUSSIAN FILTERING", gaussian);
			waitKey(1000);
			img[i] = gaussian.clone();
			waitKey(100);

	}
}


//////////////////////MEDIAN FILTERING/////////////////////////////////////

void CVision::median_filtering(){
	for(int i=0;i<img.size();i++){
		Mat median;


		for ( int j = 1; j < MAX_KERNEL_LENGTH; j = j + 2 )
		{
			medianBlur ( img[i], median, j );
			imshow("MEDIAN_BLUR_FILTER",median);
			waitKey(500);
		}
		img[i] = median.clone();

	}
	waitKey(4000);

}

//////////////////////BLUR FILTERING/////////////////////////////////////

void CVision::blur_filtering(){
	Mat blurred;

	for(int i=0;i<img.size();i++){
			blur(img[i],blurred,Size(5,5),Point(-1,-1),0);
			imshow("BLURRED IMAGE",blurred);
			img[i] = blurred.clone();

	}
	waitKey(4000);
}
//////////////////////BILATERAL_FILTERING/////////////////////////////////////

void CVision::bilateral_filtering(){
	Mat  bilateral;

	for(int i=0;i<img.size();i++){

		for ( int j = 1; j < MAX_KERNEL_LENGTH; j = j + 2 ){
			bilateralFilter ( img[i], bilateral, j, j*2, j/2 );
			imshow("BILATERAL IMAGE", bilateral);
			waitKey(500);
		}
		img[i] = bilateral.clone();
	}
}

void CVision::roberts_cross_filtering(){

	Mat conv;
	Mat conv2;
	Mat grad;
	Point anchor;
	double delta;
	int ddepth;
	/////////////////////////////////////////////
	anchor = Point( -1, -1 );
	delta = 0;
	ddepth = -1;
	//INITIALIZE KERNEL.
	Mat kernel = (Mat_<int>(3, 3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);
	Mat kernel_2 = (Mat_<int>(3,3)<< 1,0,-1,1,0,-1,1,0,-1);

	///////////////////////////////////////////////
	for(int i=0;i<img.size();i++){
		      filter2D(img[i],conv, ddepth , kernel, anchor, delta, BORDER_DEFAULT );
		      filter2D(img[i],conv2, ddepth , kernel_2, anchor, delta, BORDER_DEFAULT );

		      grad = Mat::zeros(conv.size(),CV_32F);
		      conv.convertTo(conv,CV_32F);
		      conv2.convertTo(conv2,CV_32F);

		      magnitude(conv, conv2, grad);
		      imshow("ROBERT_CROSS_FILTERING",grad);
		      waitKey(2000);
		      img[i]=grad.clone();
	}
	waitKey(4000);
}



void CVision::Histogram_Processing(){

	int option;
	bool go_back=true;
	string options[4] = {"1.HISTOGRAM EQUALIZATION",
						 "2.HISTOGRAM MATCHING",
						 "3.LOCAL HISTOGRAM PROCESSING",
						 "4.GENERATE HISTOGRAM STATISTICS"
						 "5.GO_BACK"
						};


	while(go_back){

		int num_elem = sizeof(options)/sizeof(options[0]);

		for(int i=0; i<num_elem; i++){
				cout<<""<<options[i]<<endl;
		}

		printf("CHOOSE OPTION");
		scanf("%i",&option);

		switch(option){
			case 1:Histogram_Equalization();break;
			case 2:Histogram_Matching();break;
			case 3:Histogram_Local_Processing();break;
			case 4:Histogram_Statistics();break;
			case 5:go_back=false;break;
		}
		waitKey(1000);
	}

}

void CVision::Histogram_Equalization(){

	for(int i=0;i<img.size();i++){

	}
}

void CVision::Histogram_Matching(){

	for(int i=0;i<img.size();i++){
	}
}

void CVision::Histogram_Local_Processing(){

	for(int i=0;i<img.size();i++){

	}
}

void CVision::Histogram_Statistics(){

	for(int i=0;i<img.size();i++){


	}



}

///////////////////////////////////////////////////////////////////////////////
/////////////////////MORPHOLOGICAL PROCESSING/////////////////////////////////
/////////////////////////////////////////////////////////////////////////////


void CVision::Morphological_Processing(){


			int option;
			int thin_option;
			bool go_back = true;

			while(go_back){
			cout<<"\nMORPHOLOGICAL OPERATIONS"<<endl;
			string options[9]= {"1.EROSION",
								"2.DILATION",
								"3.OPENING",
								"4.CLOSING",
								"5.THINNING",
								"6.MORPH_GRADIENT",
								"7.TOP_HAT",
								"8.BLACK_HAT.........",
								"9.GO BACK TO MENU..."};



			int num_elem = sizeof(options)/sizeof(options[0]);
						for(int i=0;i<num_elem;i++){
							cout<<""<<options[i]<<""<<endl;
			}

				printf("CHOOSE YOUR OPTION:");
				scanf("%i",&option);

				cout<<"OPTION:"<<option;
				cout<<""<<endl;
				switch(option){
					case 1:Erosion(0,0);break;
					case 2:Dilation(0,0);break;
					case 3:Opening(0,0);break;
					case 4:Closing(0,0);break;
//					case 5:printf("SELECT OPTION:THINNING_ZHANGSUEN(0) or THINNING_GUOHALL(1)");scanf("%d",&thin_option);
//							Thinning(thin_option);break;
					case 6:Morphological_Gradient(0,0);break;
					case 7:Top_Hat(0,0);break;
					case 8:Black_Hat(0,0);break;
					case 9:go_back=false;break;
				}
				waitKey(1000);
			}
}


Mat erosion_dst, dilation_dst;

int erosion_elem = 0;
int erosion_size = 4;
int dilation_elem = 0;
int dilation_size = 4;
int const max_elem = 2;
int const max_kernel_size = 21;


vector<Mat> CVision::Erosion(int, void*)
{


	  int erosion_type;
	  if( erosion_elem == 0 ){
		  erosion_type = MORPH_RECT;
	  }
	  else if( erosion_elem == 1 ){
		  erosion_type = MORPH_CROSS;
	  }
	  else if( erosion_elem == 2) {
		  erosion_type = MORPH_ELLIPSE;
	  }
	  else if(erosion_elem == 3){
		  erosion_type = CV_SHAPE_CUSTOM;
	  }




	  Mat element = getStructuringElement( erosion_type,
                                  Size( 2*erosion_size + 1, 2*erosion_size+1 ),
								  Point( erosion_size, erosion_size ) );

	  for(int i=0;i<img.size();i++){
			  erode( img[i], erosion_dst, element );
			  imshow( "EROSION", erosion_dst );
			  img[i] = erosion_dst.clone();
			  waitKey(100);
	   }
	   return img;
}

//@Erosion

vector<Mat> CVision::Dilation(int, void*){


  int dilation_type;
  if( dilation_elem == 0 ){
	  dilation_type = MORPH_RECT;
  }
  else if( dilation_elem == 1 ){
	  dilation_type = MORPH_CROSS;
  }
  else if( dilation_elem == 2) {
	  dilation_type = MORPH_ELLIPSE;
  }
  else if(dilation_elem == 3){
		  dilation_type = CV_SHAPE_CUSTOM;
  }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );


  for(int i=0;i<img.size();i++){
		  dilate( img[i], dilation_dst, element );
		  imshow( "DILATION", dilation_dst );
		  img[i] = dilation_dst.clone();
		  waitKey(100);

  }
  return img;

}

//@Opening

vector<Mat> CVision::Opening(int, void*)
{
	Mat erosion_dst;
	Mat opening_dst;
	int erosion_type;

	  if( erosion_elem == 0 ){
		  erosion_type = MORPH_RECT;
	  }
	  else if( erosion_elem == 1 ){
		  erosion_type = MORPH_CROSS;
	  }
	  else if( erosion_elem == 2) {
		  erosion_type = MORPH_ELLIPSE;
	  }
	  else if(erosion_elem == 3){
	  		  erosion_type = CV_SHAPE_CUSTOM;
	  }


	  Mat element = getStructuringElement( erosion_type,
                                  Size( 2*erosion_size + 1, 2*erosion_size+1 ),
								  Point( erosion_size, erosion_size ) );

		  for(int i=0;i<img.size();i++){
			  erode( img[i], erosion_dst, element);
			  dilate(erosion_dst, opening_dst, element);
			  imshow( "OPENING", opening_dst );
			  waitKey(100);
			  img[i] = opening_dst.clone();

		  }
		  return img;
  }

//@Closing///////////////

vector<Mat> CVision::Closing(int, void*)
{
	Mat erosion_dst;
	Mat closing_dst;

	int erosion_type;

	  if( erosion_elem == 0 ){
		  erosion_type = MORPH_RECT;
	  }
	  else if( erosion_elem == 1 ){
		  erosion_type = MORPH_CROSS;
	  }
	  else if( erosion_elem == 2) {
		  erosion_type = MORPH_ELLIPSE;
	  }
	  else if(erosion_elem == 3){
	  		  erosion_type = CV_SHAPE_CUSTOM;
	  }


	  Mat element = getStructuringElement( erosion_type,
                                  Size( 2*erosion_size + 1, 2*erosion_size+1 ),
								  Point( erosion_size, erosion_size ) );

		  for(int i=0;i<img.size();i++){
			  dilate( img[i], dilation_dst, element);
			  erode(  dilation_dst, closing_dst, element);
			  imshow( "CLOSING", closing_dst );
			  waitKey(2000);
			  img[i] = closing_dst.clone();
		  }

		 return img;

  }

//void CVision::Thinning(int option){
//
//	Mat thinned;
//	/*
//	 THINNING_ZHANGSUEN = 0,
//	  THINNING_GUOHALL = 1
//	 */
//
//	for(int i=0;i<img.size();i++){
//
//		thinning(img[i],thinned,option);
//
//		imshow("Thinning Result",thinned);
//	}
//}


void CVision::Morphological_Gradient(int, void*){

	vector<Mat> grad;
	Mat erotion_dst;
	Mat dilation_dst;
	Mat subtraction;
	int erosion_type;
	vector<Mat> dilation;
	vector<Mat> erosion;
	dilation = Dilation(0,0);
	//morph = copyVectorImage();
	erosion = Erosion(0,0);
	for(int j=0;j<img.size();j++){

		subtract(dilation[j],erosion[j],subtraction);
		grad.push_back(subtraction);
	}

	for(int j=0;j<grad.size();j++){
		imshow("MORPHOLOGICAL GRADIENT",grad[j]);
		img[j]=grad[j];
		waitKey(100);

	}
}


void CVision::Top_Hat(int, void*){

	vector<Mat> top_hat;
	int erosion_type;
	Mat subtraction;
	vector<Mat> open = Opening(0,0);
	for(int i=0;i<img.size();i++){
		subtract(src[i],open[i],subtraction);
		top_hat.push_back(subtraction);
	}
	for(int j=0;j<top_hat.size();j++){
		imshow("MORPHOLOGICAL GRADIENT",top_hat[j]);
		waitKey(100);
		img[j] = top_hat[j];
	}
}


void CVision::Black_Hat(int, void*){

	vector<Mat> black_hat;
	int erosion_type;
	Mat subtraction;
	vector<Mat> open = Opening(0,0);

	for(int j=0;j<img.size();j++){
		subtract(open[j],src[j],subtraction);
		black_hat.push_back(subtraction);
		}
	//black_hat = Opening - img;
	for(int j=0;j<black_hat.size();j++){
		imshow("MORPHOLOGICAL GRADIENT",black_hat[j]);
		waitKey(100);
		img[j] = black_hat[j].clone();
	}

}


///////////////////////////////////////////////////////////////////////////
///////////////////////////SEGMENTATION METHODS//////////////////////////////
///////////////////////////////////////////////////////////////////////////



void CVision::Image_Segmentation(){

		int option;
		bool go_back = 1;

		while(go_back){

		cout<<"IMAGE SEGMENTATION OPTIONS"<<endl;
		cout<<"̣̣̣----------------------------"<<endl;
		string options[16] = {"1.SOBEL MODE",
							  "2.LAPLACIAN_MODE",
							  "3.LAPLACIAN OF GAUSSIAN",
							  "4.CANNY_MODE",
							  "5.FINDING_CONTOURS",
							  "6.DRAW CONTOURS",
							  "7.ANALYZE CONTOURS",
							  "8.HARRIS_CORNER_DETECTION",
							  "9.BUILD_DETECTOR",
							  "10.HOUGH_LINE_METHOD",
							  "11.HOUGH_CIRCLE_METHOD",
							  "12.WATERSHED_ALGORITH_DISTANCE_TRANSFORM",
							  "13.THRESHOLDING_METHODS",
							  "14.BACK-PROJECTION_METHOD",
							  "15.FUSING METHODS"
							  "16.MENU"
		};

		int num_elem = sizeof(options)/sizeof(options[0]);

		for(int i=0;i<num_elem;i++){
			cout<<""<<options[i]<<""<<endl;
		}

		printf("CHOOSE YOUR OPTION:");
		scanf("%i",&option);

		cout<<"OPTION:"<<option;
		cout<<""<<endl;


		switch(option){
				case 1:Sobel_Mode();break;
				case 2:Laplacian_Mode();break;
				case 3:Laplacian_of_Gaussian();break;
				case 4:Canny_Mode();break;
				case 5:Finding_Contours();break;
				case 6:Draw_Contours();break;
				case 7:Analyze_Contours();break;
				case 8:Harris_Corner_Mode();break;
				case 9:Build_Detector();break;
				case 10:Hough_Line_Method();break;
				case 11:Hough_Line_Circle();break;
				case 12:Watershed_Distance_Transform();break;
				case 13:Thresholding_Methods();break;
				case 14:/*Back_Projection()*/;break;
				case 15:Fusing_Methods();break;
				case 16:go_back=0;break;
		}

		waitKey(1000);
	}

}


void CVision::Sobel_Mode(){
	Mat fg;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat src, src_gray;
	Mat grad;
	char path[200];

	int ddepth = CV_16S;



	for(int i=0; i<img.size();i++){
			cout<<""<<endl;
		/// Gradient X
			Sobel( img[i], grad_x, ddepth, 2, 0, 1, scale, delta, BORDER_DEFAULT );
		/// Gradient Y
			Sobel( img[i], grad_y, ddepth, 0, 2, 1, scale, delta, BORDER_DEFAULT );

		//  ksize – size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
		//  scale – optional scale factor for the computed derivative values; by default, no scaling is applied (see getDerivKernels() for details).
		//  delta – optional delta value that is added to the results prior to storing them in dst.
		//  borderType – pixel extrapolation method (see borderInterpolate() for details).


			convertScaleAbs( grad_x, abs_grad_x );
			convertScaleAbs( grad_y, abs_grad_y );
			addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
			sprintf(path, "SOBEL_IMAGE_%d",i);
			imshow( path,grad);
			img[i]=grad.clone();
			waitKey(100);
	}
}
//////////////2.LAPLACIAN METHOD///////////////////////////


void CVision::Laplacian_Mode(){
	Mat  dst;
	int kernel_size = 1;
	int ddepth = CV_16S;
	char window_laplace[200];
	int c;
	Mat abs_dst;


	for(int i=0; i<img.size();i++){

			Laplacian( img[i], dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
			convertScaleAbs( dst, abs_dst );
			sprintf(window_laplace,"Laplacian_Raw_Image_%d",i);
			imshow( window_laplace, abs_dst );
			waitKey(100);
			img[i] = abs_dst.clone();
	}
}



void CVision::Laplacian_of_Gaussian(){


  Mat result;
  result.create(src[0].size(), src[0].type());

  for(int i=0;i<img.size();i++){

  Mat kernelGauss(3, 3, CV_32F, Scalar(0.0625));
  kernelGauss.at<float>(0, 1) = 0.125;
  kernelGauss.at<float>(1, 0) = 0.125;
  kernelGauss.at<float>(1, 1) = 0.25;
  kernelGauss.at<float>(1, 2) = 0.125;
  kernelGauss.at<float>(2, 1) = 0.125;

  filter2D(img[i], result, img[i].depth(), kernelGauss);

  Mat kernelLaplace(3, 3, CV_32F, cv::Scalar(0.0));
  kernelLaplace.at<float>(0, 1) = -1.0;
  kernelLaplace.at<float>(1, 0) = -1.0;
  kernelLaplace.at<float>(1, 1) = +4.0;
  kernelLaplace.at<float>(1, 2) = -1.0;
  kernelLaplace.at<float>(2, 1) = -1.0;

  filter2D(result, result, img[i].depth(), kernelLaplace);
  /* float matrizLaplace [] = { 0f,  1f,  0f, 1f, -4f,  1f, 0f,  1f,  0f};*/

  Zero_Crossings(result,i);
  }

}

void CVision::Zero_Crossings(Mat result,int i){

	float a=0;
	float b=0;
	float sign_a=0;
	float sign_b=0;
	  for(int i=0;i<img.size();i++){

		  for(int j=0;j<result.rows;j++){

			  for(int k=0;k<result.cols;k++){
				  a=result.at<float>(i,j);
				  sign_a = signbit(a);
				  b=result.at<float>(i,j+1);
				  sign_b = signbit(b);

				  if(sign_a != sign_b){
					  result.at<float>(i,j)=255;
					  result.at<float>(i,j+1)=255;
				  }
//				  else if(){
//					  result.at<float>(i,j)=0;
//					  result.at<float>(i,j+1)=0;
//				  }

				  }


		  }
		  imshow("LAPLACIAN_OF_GAUSSIAN",result);
		  img[i]= result.clone();
		  waitKey(100);

	  }

//	  import numpy as np
//
//	  range_inc = lambda start, end: range(start, end+1)
//
//	  # Find the zero crossing in the l_o_g image # Done in the most naive way possible
//	  def z_c_test(l_o_g_image):
//	      print(l_o_g_image)
//	      z_c_image = np.zeros(l_o_g_image.shape)
//	      for i in range(1, l_o_g_image.shape[0]-1):
//	          for j in range(1, l_o_g_image.shape[1]-1):
//	              neg_count = 0
//	              pos_count = 0
//	              for a in range_inc(-1, 1):
//	                  for b in range_inc(-1, 1):
//	                      if(a != 0 and b != 0):
//	                          print("a " + str(a) + " b " + str(b))
//	                          if(l_o_g_image[i+a, j+b] < 0):
//	                              neg_count += 1
//	                              print("neg")
//	                          elif(l_o_g_image[i+a, j+b] > 0):
//	                              pos_count += 1
//	                              print("pos")
//	                          else:
//	                              print("zero")
//
//	              # If all the signs around the pixel are the same and they're not all zero, then it's not a zero crossing and an edge.
//	              # Otherwise, copy it to the edge map.
//	              z_c = ( (neg_count > 0) and (pos_count > 0) )
//
//	              if(z_c):
//	                  print("True for " + str(i) + "," + str(j))
//	                  print("pos " + str(pos_count) + " neg " + str(neg_count))
//	                  z_c_image[i, j] = 1
//
//	      return z_c_image
//
//	  test1 = np.array([[0,0,1], [0,0,0], [0,0,0]])
//	  test2 = np.array([[0,0,1], [0,0,0], [0,0,-1]])
//	  test3 = np.array([[0,0,0], [0,0,-1], [0,0,0]])
//	  test4 = np.array([[0,0,0], [0,0,0], [0,0,0]])
//	  true_result = np.array([[0,0,0], [0,1,0], [0,0,0]])
//	  false_result = np.array([[0,0,0], [0,0,0], [0,0,0]])
//
//	  real_result1 = z_c_test(test1)
//	  real_result2 = z_c_test(test2)
//	  real_result3 = z_c_test(test3)
//	  real_result4 = z_c_test(test4)
//
//	  assert(np.array_equal(real_result1, false_result))
//	  assert(np.array_equal(real_result2, true_result))
//	  assert(np.array_equal(real_result3, false_result))
//	  assert(np.array_equal(real_result4, false_result))

}

/////////////3.CANNY METHOD////////////////////////



void CVision::Canny_Mode(){

	for(int i=0;i<img.size();i++){

		Canny( img[i], detected_edges, 100, 200, kernel_size );
		imshow("DETECTED EDGES", detected_edges );
		img[i] = dst.clone();
		 waitKey(100);

	}
}

/////////////////////////4.FINDING_CONTOURS//////////////////////
void CVision::Draw_Contours(){

	Mat drawing = Mat::zeros( src[0].size(), CV_8UC3 );
	char window[200];

	for(int i=0;i<vect_contours.size(); i++){
			for( int j= 0; j<vect_contours[i].size(); j++ )
			{
			if(j == index_biggest_contour[i]){
				color = Scalar(255,0,0);
				}
				color = Scalar( 0,0, 255 );
				drawContours( drawing, vect_contours[i], j, color, 0.5, 8, hierarchy, 0, Point() );
			}
			img[i]=drawing.clone();
			sprintf(window,"CONTOURS_IMAGE_%i",i);
			imshow(window, drawing );
	}

}

void CVision::Resize_Contours(int i, int j){

	for(int l=j;l<vect_contours[i].size()-1; l++){
		vect_contours[i][l] = vect_contours[i][l+1];
	}
	vect_contours[i].resize(vect_contours[i].size()-1);
}
void CVision::Analyze_Contours(){

	vector<vector<vector<Point> > > hull(vect_contours.size());
	double max=0.0;

	for(int i=0;i<img.size();i++){
		for(int j=0;j<vect_contours[i].size(); j++){
			double a=contourArea( vect_contours[i][j],false);  //  Find the area of contour
			cout<<"Area Size:%d"<<a<<endl;
			waitKey(100);
			//convexHull( Mat(vect_contours[i][j]), hull[i][j], false );
			if(a>70){
				if(a > max){
					max = a;
					index_biggest_contour[i] = j;
				}
			Resize_Contours(i,j);
			}
		}

		cout<<"CONTOURS SIZE AFTER CALCULATION %d"<<vect_contours[i].size()<<endl;
		waitKey(500);
	}

}


void CVision::Finding_Contours(){

	  Mat canny_output;

	  for(int i=0; i<img.size();i++){
			//blur( image[i], image[i], Size(3,3) );
	          sprintf(window_Find_Contours,"RAW_IMAGE%d_FIND_CONTOURS",i);
	          cout<<"1"<<endl;
			  ///DETECT EDGES USING CANNY
			  Canny( img[i], canny_output, 100, 200 , 3 ); //20,225
			  imshow("CANNY_DETECTED_EDGES",canny_output);
			  cout<<"2"<<endl;
			  ///FIND COUNTOURS
			  findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
			  vect_contours.push_back(contours);
			  ///DRAW CONTOURS
			  Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
			  for( int j = 0; j< contours.size(); j++ )
			     {
			       Scalar color = Scalar( 0,0, 255 );
			       drawContours( drawing, contours, j, color, 0.5, 8, hierarchy, 0, Point() );
			     }

			  ///SHOW IN WINDOW
			  imshow( window_Find_Contours, drawing );
			  waitKey(100);
			  img[i]=drawing.clone();
			  cout<<"CONTOURS SIZE FOR THE IMAGE %d"<<vect_contours[i].size()<<endl;


	  }
}


//////////////////////5.HARRIS CORNER DETECTION////////////////////



void CVision::findCorners(){

	int maxCorners=500;
	double qualityLevel=0.01;
	double minDistance=10;


	for(int i=0; i<img.size()-1;i++){
		goodFeaturesToTrack(img[i], points[i],  maxCorners, 0.01, 10, Mat(),  3, 0, 0.04);
	}
}


void CVision::HarrisCorner(Mat image, string window_HC,int i)
{

  char write[200];
  Mat dst, dst_norm, dst_norm_scaled;
  dst = Mat::zeros( image.size(), CV_32FC1 );


  /// Detector parameters
  int blockSize = 2;
  int apertureSize = 7;
  double k = 0.04;

  /// Detecting corners
  cornerHarris( image, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
  /// Normalizing
  normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
  convertScaleAbs( dst_norm, dst_norm_scaled );


  /// Drawing a circle around corners
//  for( int j = 0; j < dst_norm.rows ; j++ )
//     { for( int i = 0; i < dst_norm.cols; i++ )
//          {
//            if( (int) dst_norm.at<float>(j,i) > 127)
//              {
//               //circle( dst_norm_scaled, Point( i, j ), 2,  Scalar(0,0,255), 2, 8, 0 );
//
//              }
//          }
//     }
  /// Showing the result
  bitwise_not(dst_norm_scaled,dst_norm_scaled);
  imshow(window_HC, dst_norm_scaled);
  img[i]=dst_norm_scaled.clone();
  cout<<"IMG TYPE:"<<img[i].type()<<endl;
  waitKey(100);
}



void CVision::Harris_Corner_Mode()
{
	Mat src, src_gray;
	int thresh = 200;
	int max_thresh = 255;
	char window_HC[200];

	for(int i=0; i<img.size();i++){
		sprintf(window_HC, "HARRIS_CORNERS_RAW_IMAGE_%d", i);
		HarrisCorner(img[i], window_HC,i);
	}
}



///////////////6.IMPROVED_CORNER_DETECTION//////////////


void myShiTomasi_function(Mat img, double myShiTomasi_minVal, double myShiTomasi_maxVal)
{
  myShiTomasi_copy = img.clone();

  if( myShiTomasi_qualityLevel < 1 ) {
	  myShiTomasi_qualityLevel = 1;
  }

  for( int j = 0; j < img.rows; j++ )
     { for( int i = 0; i < img.cols; i++ )
          {
            if( myShiTomasi_dst.at<float>(j,i) > myShiTomasi_minVal + ( myShiTomasi_maxVal - myShiTomasi_minVal )*myShiTomasi_qualityLevel/max_qualityLevel )
              {
            	circle( myShiTomasi_copy, Point(i,j),
            		4, Scalar(0,0,255),-1, 8, 0 );
              }
          }
     }
  imshow( myShiTomasi_window, myShiTomasi_copy );
  waitKey(100);
}


void myHarris_function(Mat img, double myHarris_minVal, double myHarris_maxVal)
{
  myHarris_copy = img.clone();

  if( myHarris_qualityLevel < 1 ) {
	  myHarris_qualityLevel = 1; }

  for( int j = 0; j < img.rows; j++ )
     { for( int i = 0; i < img.cols; i++ )
          {
            if( Mc.at<float>(j,i) > myHarris_minVal + ( myHarris_maxVal - myHarris_minVal )*myHarris_qualityLevel/max_qualityLevel )
              {
             	circle( myHarris_copy, Point(i,j), 4,
            		Scalar(0,255,0), -1, 8, 0 ); }
          }
     }

  imshow( myHarris_window, myHarris_copy );
  waitKey(100);
}

void CVision::Build_Detector(){
  int blockSize = 3; int apertureSize = 7;

	  for(int i=0; i<img.size(); i++)
	  {
		  myHarris_dst = Mat::zeros( img[i].size(), CV_32FC(6));
		  Mc = Mat::zeros( img[i].size(), CV_32FC1 );
		  cornerEigenValsAndVecs( img[i], myHarris_dst, blockSize, apertureSize, BORDER_DEFAULT );

	  /* calculate Mc */
	  for( int j = 0; j < img[i].rows; j++ )
		 { for( int i = 0; i < img[i].cols; i++ )
			  {
				float lambda_1 = myHarris_dst.at<Vec6f>(j, i)[0];
				float lambda_2 = myHarris_dst.at<Vec6f>(j, i)[1];
				Mc.at<float>(j,i) = lambda_1*lambda_2 - 0.04f*pow( ( lambda_1 + lambda_2 ), 2 );
			  }
		 }

	  minMaxLoc( Mc, &myHarris_minVal, &myHarris_maxVal, 0, 0, Mat() );
	  myHarris_function(img[i],myHarris_minVal, myHarris_maxVal);
	  myShiTomasi_dst = Mat::zeros( img[i].size(), CV_32FC1 );
	  cornerMinEigenVal( img[i], myShiTomasi_dst, blockSize, apertureSize, BORDER_DEFAULT );
	  minMaxLoc( myShiTomasi_dst, &myShiTomasi_minVal, &myShiTomasi_maxVal, 0, 0, Mat() );
	  myShiTomasi_function(img[i],myShiTomasi_minVal, myShiTomasi_maxVal);
	  waitKey(100);
	}
}

//////////////////////////7.HOUGH-LINE-CIRCLES//////////////////////////////////////

void CVision::Hough_Line_Method()
{
	Mat dst, cdst;

	for(int i=0; i<img.size(); i++){
		 //APPLY OTSUS METHOD - DETECT MAX_THRESHOLD AND MIN_THRESHOLD.

		 Canny(img[i], dst, 50, 200, 3);
		 cvtColor(dst, cdst, CV_GRAY2BGR);
		  vector<Vec4i> lines;
		  HoughLinesP(dst, lines, 1, CV_PI/180, 20, 5 , 8 );
		  //GOOD RESULTS FOR:
		  	  	  //(20,5,5)
		  	  	  //(20,5,7)
		  	  	  //(20,5,8)

		  for( int j = 0; j < lines.size(); j++ )
		  {
		    Vec4i l = lines[j];
		    line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
		  }

		 imshow("SOURCE_IMG", src[i]);
		 imshow("DETECTED_LINES", cdst);
		 img[i] = cdst;
		 waitKey(100);
	}
}




void CVision::Hough_Line_Circle(){


	for(int i=0;i<img.size();i++){

		 /// Reduce the noise so we avoid false circle detection
		  //GaussianBlur( image[i], image[i], Size(3, 3), 2, 2 );

		  vector<Vec3f> circles;

		  /// Apply the Hough Transform to find the circles
		  HoughCircles( img[i], circles, CV_HOUGH_GRADIENT, 1, 50, 200, 100, 0, 0 );

		  cout<<"NR OF CIRCLES DETECTED:"<<circles.size();
		  cout<<""<<endl;
		  waitKey(200);

		  /// Draw the circles detected
		  for( int j = 0; j < circles.size(); j++ )
		  {
		      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		      int radius = cvRound(circles[i][2]);
		      // circle center
		      circle( img[i], center, 3, Scalar(0,255,0), -1, 8, 0 );
		      // circle outline
		      circle( img[i], center, radius, Scalar(0,0,255), 3, 8, 0 );
		   }

		  /// Show your results
		  namedWindow( "HOUGH CIRCLE", CV_WINDOW_AUTOSIZE );
		  imshow( "HOUGH_CIRCLE", img[i] );
		  waitKey(100);
	}
}




void CVision::Watershed_Distance_Transform(){

	invertImages();
	vector<Mat> src_copy = copyOriginalImage();

	for(int i=0;i<img.size();i++){

		Mat kernel = (Mat_<float>(3,3) << 1,  1, 1, 1, -8, 1, 1,  1, 1); // an approximation of second derivative, a quite strong kernel
		Mat imgLaplacian;
		Mat sharp = src_copy[i];
		filter2D(sharp, imgLaplacian, CV_32F, kernel);
		src_copy[i].convertTo(sharp, CV_32F);
		Mat imgResult = sharp - imgLaplacian;

		imgResult.convertTo(imgResult, CV_8UC3);
		imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

		//////////////////////////////////////////////////

		src_copy[i] = imgResult; // copy back

		Mat bw = img[i].clone();

		threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		imshow("Binary Image", bw);
		// Perform the distance transform algorithm
		Mat dist;
		distanceTransform(bw, dist, CV_DIST_L2, 3);
		// Normalize the distance image for range = {0.0, 1.0}
		// so we can visualize and threshold it
		normalize(dist, dist, 0, 1., NORM_MINMAX);
		imshow("DISTANCE_TRANSFORM_IMAGE", dist);
		// Threshold to obtain the peaks
		// This will be the markers for the foreground objects
		threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
		// Dilate a bit the dist image
		Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
		dilate(dist, dist, kernel1);
		imshow("Peaks", dist);

		////////////////////////////////////////////////////////////

		// Create the CV_8U version of the distance image
		// It is needed for findContours()
		Mat dist_8u;
		dist.convertTo(dist_8u, CV_8U);

		// Find total markers
		vector<vector<Point> > contours;
		findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		// Create the marker image for the watershed algorithm
		Mat markers = Mat::zeros(dist.size(), CV_32SC1);
		// Draw the foreground markers
		for (size_t j = 0; j < contours.size(); j++){
		  	drawContours(markers, contours, static_cast<int>(j), Scalar::all(static_cast<int>(j)+1), -1);
		}

		cout<<"2"<<endl;
		circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
		imshow("Markers", markers*10000);

		//WATERSHED ALGORITHM

		cvtColor(src_copy[i],src_copy[i] ,CV_GRAY2BGR, 3 );
		Mat channels[3];
		split(src_copy[i],channels);

		imshow("CHANNEL_0",channels[0]);
		imshow("CHANNEL_1",channels[1]);
		imshow("CHANNEL_2",channels[2]);

		img[i]=channels[0].clone();
//		break;
//
//		watershed(src_copy[i], markers);
//		Mat mark = Mat::zeros(markers.size(), CV_8UC1);
//		markers.convertTo(mark, CV_8UC1);
//		bitwise_not(mark, mark);
//
//		// Generate random colors
//		vector<Vec3b> colors;
//		for (size_t j = 0; j < contours.size(); j++)
//		{
//			int b = theRNG().uniform(0, 255);
//			int g = theRNG().uniform(0, 255);
//			int r = theRNG().uniform(0, 255);
//			colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
//		}
//
//		//DRAW THE RESULT.
//
//		Mat dst = Mat::zeros(markers.size(), CV_8UC3);
//		// Fill labeled objects with random colors
//		for (int k = 0; k < markers.rows; k++)
//		{
//			for (int l = 0; l < markers.cols; l++)
//			{
//				int index = markers.at<int>(k,l);
//				if (index > 0 && index <= static_cast<int>(contours.size()))
//					dst.at<Vec3b>(k,l) = colors[index-1];
//				else
//					dst.at<Vec3b>(k,l) = Vec3b(0,0,0);
//			}
//		}
//		// Visualize the final image
//		imshow("Final Result", dst);
//		img[i] = dst.clone();
//		waitKey(100);
		}

	}


void CVision::Fusing_Methods(){

	char fuse_method_one[200];
	char fuse_method_two[200];
	vector<int> compression_params;
	Mat method_one;
	Mat method_two;
	float a;
	float b;


	for(int i=0; i<img.size(); i++){

		//sprintf(fuse_method_one,"/home/xisco/Escritorio/CV_PROJECTS/BUILDING_OUTLINES/IMAGES/GAUSSIAN_SOBEL_%d.png",i);
		sprintf(fuse_method_one,"/home/xisco/Escritorio/CV_PROJECTS/BUILDING_OUTLINES/IMAGES/CLOSING_SOBEL_%d.png",i);
		sprintf(fuse_method_two,"/home/xisco/Escritorio/CV_PROJECTS/BUILDING_OUTLINES/IMAGES/OTSU_HARRIS_CORNER_%d.png",i);

		method_one = imread(fuse_method_one,CV_LOAD_IMAGE_GRAYSCALE);
		method_two = imread(fuse_method_two,CV_LOAD_IMAGE_GRAYSCALE);

		method_one.convertTo(method_one,CV_32FC1);
		method_two.convertTo(method_two,CV_32FC1);

		Mat comb(method_one.size(),CV_32FC1,Scalar::all(0));

		for(int l=0;l<method_one.cols;l++){
			for(int k=0;k<method_one.rows;k++){
				comb.at<float>(k,l) = (method_one.at<float>(k,l)*method_two.at<float>(k,l))/255;
			}
		}

		comb = comb/.255;
		cout<<"3"<<endl;
		comb.convertTo(comb,CV_8UC1);
		normalize( comb, comb, 0, 255, NORM_MINMAX, -1, Mat() );
		imshow("FUSED_METHODS",comb);
		img[i]=comb.clone();
	}

}



void CVision::Object_Recognition(){


	int option;


	string options[7] = {"1.GENERATE HAAR FEATURES",
					     "2.GENERATE LBP FEATURES",
					     "3.HAAR CASCADE DETECTION",
						 "4.SIFT DETECTION",
						 "5.SURF DETECTION",
						 "6.HoG DESCRIPTION",
					     "7.LATENT SVM"
	};


	printf("OBJECT RECOGNITION METHODS");


	int size = sizeof(options)/sizeof(options[0]);


	for(int i=0;i<size;i++){
		cout<<"\n"<<options[i]<<endl;
	}
	printf("\nCHOOSE YOUR OPTION:");
	scanf("%d",&option);

	switch(option){
	     case 1:/*Generate_Haar_Features()*/;break;
	 	 case 2:Generate_LBP_Features();break;
	 	 case 3:Haar_Cascade_Detection();break;
	 	 case 4:Sift_Detection();break;        //FAILS
	 	// case 5:Surf_Detection_Method();break;
	 	 case 6:HoG_Detection_Method();break;  //DONE
	 	 case 7:/*Latent_SVM()*/;break;
	}

}

//
//void CVision::Generate_Haar_Features(){
	//1.COLLECT NEGATIVE OR BACKGROUND IMAGES;
		//ANY IMAGE WILL DO, MAKE SURE YOUR OBJECT IS NOT PRESENT
		//IN THEM. GET THOUSANDS.
		//NEGATIVE IMAGES:
			//CREATE A TEXT FILE WHERE THE PATH TO THE IMAGES IS STORED.

	//2.COLLECT OR CREATE POSITIVE IMAGES;
		//THOUSANDS OF IMAGES OF YOUR OBJECT. CAN MAKE THESE BASED ON ONE IMAGE.
		//OR MANUALLY CREATHE THEM.
			//CREATE A FILE THAT CONTAINS A PATH TO THE IMAGES ALONG WITH HOW MANY
			//OBJECTS AND WHERE THEY ARE LOCATED.


//3.CREATE A POSITIVE FILE BY STITCHING TOGETHER ALL POSITIVES.
		//THIS IS DONE WITH OPENCV COMMAND.
	//4.TRAIN CASCADE.
		//DONE WITH OPENCV COMMAND.

	//IMPORTANT:
	//TO TRAIN THE SYSTEM TRY TO USE IMAGES OF 50X50.
	//YOU WANT NEGATIVE IMAGES TO BE LARGER THAN POSITIVE IMAGES.
	//DOUBLE IMAGES COMPARED TO THE NEGATIVE.

//}




void CVision::Concatenate_Histograms(){

	MatND concatenated_Histogram(vect_Hist[0].size(),CV_32FC1, Scalar::all(0));
	int histSize = 59;
	float range[] = { 0, 255 };
	const float *ranges[] = { range };


	for(int i=0;i<vect_Hist.size();i++){
		concatenated_Histogram += vect_Hist[i];
	}

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );

	Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
	normalize(concatenated_Histogram, concatenated_Histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	for( int j = 1; j < histSize; j++ )
	{
	  line( histImage, Point( bin_w*(j-1), hist_h - cvRound(concatenated_Histogram.at<float>(j-1)) ) ,
					   Point( bin_w*(j), hist_h - cvRound(concatenated_Histogram.at<float>(j)) ),
					   Scalar( 255, 0, 0), 2, 8, 0  );
	}

	imshow( "CONCATENATED_HISTOGRAM",histImage);
	//write_Data(concatenated_Histogram);
	waitKey(2000);


}

void CVision::Generate_Histograms(Mat cod_Hist){

	//59 bins, values from 0 to 256;
    // Initialize parameters

		int histSize = 59;    // bin size
		float range[] = { 0, 255 };
		const float *ranges[] = { range };

        	// Calculate histogram
		MatND hist;
		calcHist(&cod_Hist, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false );

		// Show the calculated histogram in command window
		double total;
		total = cod_Hist.rows * cod_Hist.cols;
		for( int h = 0; h < histSize; h++ )
			 {
				float binVal = hist.at<float>(h);
				//cout<<" "<<binVal;
			 }

		// Plot the histogram
		int hist_w = 512; int hist_h = 400;
		int bin_w = cvRound( (double) hist_w/histSize );

		Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
		normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

		for( int i = 1; i < histSize; i++ )
		{
		  line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
						   Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
						   Scalar( 255, 0, 0), 2, 8, 0  );
		}
		vect_Hist.push_back(hist);
}

void CVision::Generate_LBP_Features(){

	int j=0,k=0;
	LBP = copyOriginalImage();
	Size size_iteration = img[0].size()/10;
	Mat cod_Hist (size_iteration,CV_32FC1,Scalar::all(0));
	int start_cols=0,start_rows=0;
	float number_LBP = size_iteration.width * size_iteration.height;

	float value=0;
	float codification=0;
	float bin=0;
	int m=0,n=0;

	for(int i=0;i<img.size();i++){
		do{
			do{
				for(j=start_cols;j<start_cols + size_iteration.width-1;j++){
					for(k=start_rows;k<start_rows + size_iteration.height-1;k++){
						Point iterations[8] = {Point(k-1,j-1),
											   Point(k-1,j),
											   Point(k-1,j+1),
											   Point(k,j+1),
											   Point(k+1,j+1),
											   Point(k+1,j),
											   Point(k+1,j-1),
											   Point(k,j-1)};

						int num_elem=sizeof(iterations)/sizeof(iterations[0]);
						for(int l=0;l<num_elem;l++){

								if(  img[i].at<float>(iterations[l].x, iterations[l].y) > img[i].at<float>(k,j)
								  || img[i].at<float>(iterations[l].x, iterations[l].y) == img[i].at<float>(k,j))
								{
									value=1;
								}
								else{
									value=0;
								}
						codification += (2^(num_elem-l))*value;
						}
						LBP[i].at<float>(k,j)=codification;
						cod_Hist.at<float>(n,m)=codification;
						codification=0;
						n++;
					}
					m++;
					n=0;
				}
				m=0;
				n=0;
				Generate_Histograms(cod_Hist);
				start_cols += size_iteration.width;
		}while(start_cols < (img[i].cols-5));
			cout<<"NEW ROW"<<endl;
			start_cols = 0;
			start_rows +=size_iteration.height;
		}while(start_rows < img[i].rows);
		start_cols=0;
		start_rows=0;
		Concatenate_Histograms();
	}
}





void CVision::Haar_Cascade_Detection(){


 vector<Rect> faces;
 Mat frame_gray;

 /** Global variables */
 String face_cascade_name = "haarcascade_frontalface_alt.xml";
 String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
 String haar_cascade_eye =  "haarcascade_eye.xml";
 String eye_tree_eyeglasses ="haarcascade_eye_tree_eyeglasses.xml";
 String haar_cascade_frontalcat_face = "haarcascade_frontalcatface.xml";
 String haar_cascade_frontalcat_face_extended = "haarcascade_frontalcatface_extended.xml";
 String haar_cascade_frontalface_alt = "haarcascade_frontalface_alt.xml";
 String haar_cascade_frontalface_alt2 = "haarcascade_frontalface_alt2.xml";
 String haar_frontalface_alt_tree = "haarcascade_frontalface_alt_tree.xml";
 String haarcascade_frontalface_default = "haarcasce_frontalface_default.xml";
 String haarcascade_fullbody = "haarcascade_fullbody.xml";
 String haarcascade_lefeye_2splits = "haarcascade_lefteye_2splits.xml";
 String haarcascade_license_plate_rus_16stages = "haarcascade_license_plate_rus_16stages.xml";
 String haarcascade_lowerbody = "haarcascade_lowerbody.xml";
 String haarcascade_profileface = "haarcascade_profileface.xml";
 String haarcascade_righteye_2splits = "haarcascade_righteye_2splits.xml";
 String haarcascade_russian_plate_number = "haarcascade_russian_plate_number.xml";
 String haarcascade_smile = "haarcascade_smile.xml";
 String haarcascade_upperbody = "haarcascade_upperbody.xml";


 CascadeClassifier face_cascade;
 CascadeClassifier eyes_cascade;
 string window_name = "Capture - Face detection";
 RNG rng(12345);


 //-- 1. Load the cascades
   if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return; };
   if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return; };


 for(int l=0; l<img.size();l++){

	 cvtColor( img[l], frame_gray, CV_BGR2GRAY );
	 equalizeHist( frame_gray, frame_gray );
	 //DETECTING FACES
	 face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

	 for( size_t i = 0; i < faces.size(); i++ )
	 {
	   Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
	   ellipse( frame_gray, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

	   Mat faceROI = frame_gray( faces[i] );
	   vector<Rect> eyes;

	   //-- In each face, detect eyes
	   eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

	   for( size_t j = 0; j < eyes.size(); j++ )
		{
		  Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
		  int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
		  circle( frame_gray, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
		}
	 }
	 //-- Show what you got
	 imshow( window_name, frame_gray );
	 img[l]=frame_gray.clone();

	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////SIFT EDGE DETECTION SYSTEM/////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//1.SCALE_IMAGES.

void CVision::Scale_Images(int pyramids_number, int levels_number){

	vector<Mat> resized = copyOriginalImage();
	Mat dst;
	int it=0;
	char window_pyr[200];char window[200];
	pyramids = pyramids_number;
	levels = levels_number;
	Scaled_Images = vector<vector<Mat> > (pyramids,vector<Mat>(levels));

	for(int i=0;i<img.size();i++){
		dst = resized[i].clone();
		for(int j=0;j<pyramids;j++){
			it=0;
			for(int k=1;k<levels*2;k=k+2){
				blur(dst,blured,Size(k,k));
				Scaled_Images[j][it] = blured.clone();
				it++;
			}
			pyrDown(dst, dst, Size(dst.cols/2,dst.rows/2));
		}
	}
}

//////////////////////////////////////////////
//2 BLUR THEM AND APPLY DIFFERENCE OF GAUSSIAN.
//////////////////////////////////////////////

void CVision::Difference_of_Gaussians(){

	DoG = vector<vector<Mat> > (pyramids,vector<Mat>(levels-1));

	for(int i=0;i<Scaled_Images.size();i++){
		Mat subtraction = Mat(Scaled_Images[i][0].size(),CV_8U, Scalar::all(0));
		for(int j=0; j<Scaled_Images[i].size()-1; j++){
			subtraction = Scaled_Images[i][j] - Scaled_Images[i][j+1];
			DoG[i][j]=subtraction.clone();
		}
	}
}

//3.LOCATE MAXIMA AND MINIMA. FIND SUBPIXEL MINIMA AND MAXIMA.

void CVision::Locate_Maxima_Minima(){
	int it=0; float m=0,l=0;
	int maxima_element = 0,minima_element = 0;
	vector<int> elements;
	int num_elements=0;
	float a=0,b=0,c=0,center=0;

	KPoint.KPoints = vector<vector<vector<Point2f> > > (DoG.size()) ;

	for(int i=0;i<DoG.size();i++){
			for(int j=1;j<DoG[i].size()-1;j++){
				for(l=0;l<DoG[i][j].cols;l++){
					for(m=0;m<DoG[i][j].rows;m++){
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
					    //	Max_Min[i][j-1].at<float>(m,l) = 1;
					    	cout<<"M:"<<m<<"\nL:"<<l<<endl;
							KPoint.KPoints[i][j-1].push_back(Point2f(m,l));
					    	cout<<"Kpoint size:"<<KPoint.KPoints[i][j-1].size()<<endl;

					    }

					}
				}
				//imshow("MAX_MIN RESULT",Max_Min[i][j-1]);
				waitKey(2000);
			}

	}
}



void CVision::Resize_KPoints(int i, int j, int k){

	for(int k=0;k<KPoint.KPoints[i][j].size()-1;k++){
		KPoint.KPoints[i][j][k] = KPoint.KPoints[i][j][k+1];
	}
	KPoint.KPoints[i][j].resize(KPoint.KPoints[i][j].size()-1);
}




void CVision::Finding_KeyPoints(){

	Mat sec_deriv = (Mat_<float>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	Mat first_deriv = (Mat_<float>(3,1) << 0,0,0);
	Mat h = (Mat_<float>(3,3) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );

	float m,l;
	int num_elements=0;

	float dx, dy, ds;float dxx,dxy,dyy,dss,dxs,dys;


	cout<<"FINDING_KEYPOINTS"<<endl;
	for(int i=0;i<KPoint.KPoints.size();i++){

		for(int j=0;j<KPoint.KPoints[i].size();j++){

			for(int k=0;k<KPoint.KPoints[i][j].size();k++){
				m=KPoint.KPoints[i][j][k].x;
				l=KPoint.KPoints[i][j][k].y;

			//Point iterations[5] = {	Point(m+1,l),Point(m-1,l),Point(m,l+1),Point(m,l-1),Point(m,l)};
			//num_elements=sizeof(iterations)/sizeof(iterations[0]);

				cout<<"INSIDE LOOP"<<endl;
				//FIRST-ORDER DERIVATIVE
				 dx = (DoG[i][j+1].at<float>(m- 1, l) - DoG[i][j+1].at<float>(m + 1,l )) / 2;
				 dy = (DoG[i][j+1].at<float>(m, l-1) - DoG[i][j+1].at<float>(m, l+ 1)) / 2;
				 ds = (DoG[i][j].at<float>(m,l) - DoG[i][j+2].at<float>(m, l)) / 2;

				first_deriv.at<float>(0, 0) = dx; first_deriv.at<float>(1, 0) = dy; first_deriv.at<float>(2, 0) = ds;

				//SECOND-ORDER DERIVATIVE...

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

		    h = first_deriv * sec_deriv *(-1); double offset = determinant(h);
		    transpose(first_deriv, first_deriv);
		    double extrema_value = determinant(DoG[i][j].at<float>(m,l) + (1/2)*offset*(first_deriv));

		    if(offset>0.5){
		    	cout<<"THE SAMPLE POINT MUST BE CHANGED"<<endl;
		    	waitKey(1000);
		    }
		   if(extrema_value < 0.03){
			   	cout<<"EXTREMA THAN NEEDS TO BE DISCARDED"<<endl;
			   	Resize_KPoints(i,j,k);
		   }


		}

	}

	//IF THE CONTRAST THRESHOLD IS LESS THAN 0.03 THEN IS REJECTED.
	}
}


void CVision::Elimination_Edge_Responses(){
	Mat sec_deriv = (Mat_<float>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	Mat first_deriv = (Mat_<float>(3,1) << 0,0,0);
	Mat h = (Mat_<float>(3,3) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	Mat hessian = (Mat_ <float>(2,2) << 0,0,0,0);
	float m,l;
	int num_elements=0;
	vector<float> eigen_values;

		float dx, dy, ds;float dxx,dxy,dyy,dss,dxs,dys;

		for(int i=0;i<KPoint.KPoints.size();i++){

			for(int j=0;j<KPoint.KPoints[i].size();j++){

				for(int k=0;k<KPoint.KPoints[i][j].size();k++){
					m=KPoint.KPoints[i][j][k].x;
					l=KPoint.KPoints[i][j][k].y;

					//FIRST-ORDER DERIVATIVE
					dx = (DoG[i][j+1].at<float>(m- 1, l) - DoG[i][j+1].at<float>(m + 1,l )) / 2;
					dy = (DoG[i][j+1].at<float>(m, l-1) - DoG[i][j+1].at<float>(m, l+ 1)) / 2;
					ds = (DoG[i][j].at<float>(m,l) - DoG[i][j+2].at<float>(m, l)) / 2;

					first_deriv.at<float>(0, 0) = dx; first_deriv.at<float>(1, 0) = dy; first_deriv.at<float>(2, 0) = ds;

						//SECOND-ORDER DERIVATIVE...

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

					hessian.at<float>(0,0)=dxx; hessian.at<float>(0,1)=dxy; hessian.at<float>(1,0)=dxy; hessian.at<float>(1,1)=dyy;

					eigen(hessian, eigen_values);
					double max = *max_element(eigen_values.begin(),eigen_values.end());
					double min = *min_element(eigen_values.begin(),eigen_values.end());

					double r = max/min ;  //THE NEW APPROACH USES r=10
					double Tr = dxx + dyy;
					double ratio = (r+1)*(r+1)/r ;
					double div = (Tr*Tr)/determinant(hessian);

					if(div > ratio){
					   	Resize_KPoints(i,j,k);
					}

				}
			}
		}


}


void CVision::KeyPoint_Orientation(){


		float m,l;
		int num_elements=0;
		double orientation=0,magnitude=0;
		double gx=0,gy=0;

		KPoint.orientations = vector<vector<vector<vector<float> > > > (pyramids,vector<vector<vector<float> > >(levels-3));

		//HISTOGRAM GENERATION.
		float bin_size = 36;
		float scale=0;
		float hist_bin=0;
		vector<int> bins;
		int item=0;
		double max_bin = 0.0;
		vector<vector<Mat> > Magnitude_Image;
		float sum=0;

		for(int i=0;i<KPoint.KPoints.size();i++){

			for(int j=0;j<KPoint.KPoints[i].size();j++){

				//LAPLACIAN OF ORIGINAL IMAGE
				scale = 1.5*(1/i+1);
				GaussianBlur(DoG[i][j],Magnitude_Image[i][j], Size(3,3),scale,scale);
				cout<<"1"<<endl;
				KPoint.orientations[i][j].resize(KPoint.KPoints[i][j].size());
				cout<<"2"<<endl;
				for(int k=0;k<KPoint.KPoints[i][j].size();k++){
					m=KPoint.KPoints[i][j][k].x;
					l=KPoint.KPoints[i][j][k].y;
					Point iterations[9] = {Point(m-1,m-1),Point(m-1,l),Point(m-1,l+1),
										   Point(m,l+1),Point(m+1,l+1),Point(m+1,l),
										   Point(m+1,l-1),Point(m,l-1),Point(m,l)};

					int num_elements = sizeof(iterations)/sizeof(iterations[0]);
					for(int n=0;n<num_elements;n++){

						gx= Magnitude_Image[i][j].at<float>(iterations[n].x + 1,iterations[n].y)-
								Magnitude_Image[i][j].at<float>(iterations[n].x-1,iterations[n].y);
						gy= Magnitude_Image[i][j].at<float>(iterations[n].x,iterations[n].y+1) -
								Magnitude_Image[i][j].at<float>(iterations[n].x,iterations[n].y-1);
						orientation = atan(gy/gx);
						magnitude = (gx*gx) +(gy*gy);
						hist_bin = magnitude;
						sum = magnitude * 1.5;
						item = floor(orientation / 10);
						item = item % 35;
						bins[item] += sum;
					}
					max_bin = *max_element(bins.begin(),bins.end());
					KPoint.orientations[i][j][k].push_back(max_bin);

					for(l=0;l<bins.size();l++){
						if(bins[i] > 0.8*max_bin){
							KPoint.orientations[i][j][k].push_back(bins[i]);
						}

					}


				}
			}
		}
}



void CVision::Sift_Descriptor(){

	int m=0;
	int n=0;
	float gx,gy; float magnitude=0; float orientation=0; float angles=0;
	float sum=0; int item=0;
	int start_cols=0,start_rows=0;
	vector<float> bins;
	for(int i=0;i<DoG.size();i++){

		for(int j=0;j<DoG[i].size();j++){

			for(int k=0;k<KPoint.KPoints[i][j].size();k++){

				for(int l=0;l<KPoint.orientations[i][j][k].size();l++){
					m=KPoint.KPoints[i][j][k].x;
					n=KPoint.KPoints[i][j][k].y;
					start_cols =m;
					start_rows =n;
					do{
						for(int o=start_cols;o<start_cols+4;o++){
							for(int p=start_rows;p<start_rows+4;n++){
								gx= DoG[i][j].at<float>(m-1,n) - DoG[i][j].at<float>(m+1,n);
								gy= DoG[i][j].at<float>(m,n-1) - DoG[i][j].at<float>(m,n+1);
								magnitude = (gx*gx)+(gy*gy); orientation = atan(gy/gx);
								sum = magnitude * 1.5;
								angles = 360/8;
								item = floor(orientation / angles);
								item = item%7;
								bins[item]=+sum;
							}
						}
						for(int s=0;s<bins.size();s++){
							KPoint.Sift_Descriptor[k].push_back(bins[s]);
						}
						start_cols += 4;
						start_rows += 4;
					}while((start_cols < m+8) || (start_rows < n+8));

				}
			}

		}
	}

}

void CVision::Sift_Detection(){

			Scale_Images(4,5);
			Difference_of_Gaussians();
			Locate_Maxima_Minima();
			Finding_KeyPoints();
			Elimination_Edge_Responses();
			KeyPoint_Orientation();
			Sift_Descriptor();
}

//////////////////////////////////////////////////////////////////////////////////////
/////////////////////////SURF DETECTION METHOD////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////


void CVision::Gradient_Images(){


	Mat mag, angle;
	Mat channels[3];
	double max_gx=0,max_gy=0;
	double min_gx=0,min_gy=0;
	int num_elements=0;

	for(int i=0;i<img.size();i++){
			split(img[i],channels);
			num_elements = sizeof(channels)/sizeof(channels[0]);
			for(int j=0;j<num_elements;j++){
				Sobel(channels[j], gx[j],CV_32F, 1,0,1);
				Sobel(channels[j], gy[j],CV_32F, 0,1,1);
			}
			Gradient_Color();
		}
}

void CVision::Gradient_Color(){


	GX = Mat(gx[0].size(),gx[0].type(),Scalar(0,0,0));
	GY = Mat(gy[0].size(),gy[0].type(),Scalar(0,0,0));
	float max_gx=0,max_gy=0;

	for(int j=0;j<gx[0].cols;j++){
		for(int k=0;k<gx[0].rows;k++){
			for(int i=0;i<3;i++){
				if(gx[i].at<float>(k,j) > max_gx){
					max_gx=gx[i].at<float>(k,j);
				}
				if(gy[i].at<float>(k,j) > max_gy){
					max_gy=gy[i].at<float>(k,j);
				}
			}
			GX.at<float>(k,j) = max_gx;
			GY.at<float>(k,j) = max_gy;
			max_gx=0;max_gy=0;
		}

	}
	cartToPolar(GX, GY, magnit, angle, 1);
}


vector<vector<double> > Calculate_norm(vector<vector<double> > vector_norm){
	double norm_value=0;

	for(int i=0;i<vector_norm.size();i++){
		for(int j=0;j<vector_norm[i].size();j++){
			norm_value += vector_norm[i][j]*vector_norm[i][j];
		}
	}
	norm_value = sqrt(norm_value);
	for(int k=0;k<vector_norm.size();k++){
			for(int l=0;l<vector_norm[k].size();l++){
				vector_norm[k][l]=vector_norm[k][l]/norm_value;
			}
	}
	return vector_norm;
}


void CVision::Histogram_of_Gradients(){

	vector<double> bins(9);
	vector<vector<double> > concatenated_bins(4,vector<double>(9));
	vector<vector<double> > concatenated_bins_normalized(4,vector<double>(9));
	vector<vector<vector<double> > > concatenated_histogram;
	double normalized_value=0.0;

	float item=0; float weight_one=0; float weight_two=0;
	int start_cols=0,init_cols=0; int start_rows=0,init_rows=0;

	for(int i=0;i<img.size();i++){

			do{
				do{
					do{
					    do{
						for(int j=start_cols;j<start_cols+8;j++){
							for(int k=start_rows;k<start_rows+8;k++){
								 item = floor((angle.at<float>(k,j)/ 20));
								 weight_one = abs(angle.at<float>(k,j))-(item*20);
								 weight_one = 1 - (weight_one/20);
								 weight_two = 1 - weight_one;

								 weight_one = weight_one * abs(magnit.at<float>(k,j));
								 weight_two = weight_two * abs(magnit.at<float>(k,j));

								 bins[item] =+ weight_one;
								 if(item==9){item=-1;}
								 bins[item+1] =+ weight_two;
								}
						}
						concatenated_bins.push_back(bins);
						start_cols += 8;
					    }while( start_cols - init_cols < 15);
						start_cols = init_cols;
						start_rows +=8;
						}while(start_rows - init_rows < 15);
						init_cols += 8;
						start_cols= init_cols;
						start_rows = init_rows;
						concatenated_bins = Calculate_norm(concatenated_bins);
						concatenated_histogram.push_back(concatenated_bins);
						concatenated_bins.resize(0);
						}while(start_cols < img[i].cols -10);
						init_rows +=8;
						init_cols=0;
						start_cols=init_cols;
						start_rows=init_rows;
						}while(start_rows < img[i].rows -10);
						start_cols=0;
						start_rows=0;
						cout<<"SIZE OF THE CONCATENATED HISTOGRAM:"<<concatenated_histogram.size()<<endl;
					}
}

void CVision::HoG_Detection_Method(){

	Gradient_Images();
	Histogram_of_Gradients();

}


////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////MENU_COMPUTER_VISION/////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////


void CVision::Menu(){



			int opt_menu;
			char* answer;

			img = copyOriginalImage();
			bool get_out = true;
			string options[12] = {"1.INTENSITY TRANSFORMATIONS AND SPATIAL FILTERING.",
								  "2.FILTERING IN FREQUENCY DOMAIN",
								  "3.IMAGE RESTORATION AND RECONSTRUCTION",
								  "4.COLOR IMAGE PROCESSING",
								  "5.WAVELETS AND MULTIRESOLUTION PROCESSING",
							      "6.IMAGE COMPRESSION",
								  "7.MORPHOLOGICAL PROCESSING",
								  "8.IMAGE SEGMENTATION",
								  "9.GENERATE DESCRIPTORS",
								  "10.OBJECT RECOGNITION",
								  "11.SAVE IMAGES",
								  "12.FINISH"};





			cout<<"--WELCOME TO THE COMPUTER VISION_PROGRAM--------"<<endl;
			waitKey(1000);


			while(get_out){


			printf("LOAD ORIGINAL IMAGE/S YES(Y) OR NO(N)?");
			scanf("%c",&answer);
			//time (&start);

			if(answer == "Y"){
				cout<<"ORIGINAL IMAGE"<<endl;
				img = copyOriginalImage();
			}


			cout<<"\nLIST OF AVAILABLE OPTIONS:"<<endl;
			cout<<"----------------------------"<<endl;
			waitKey(100);
			int num_elem = sizeof(options)/sizeof(options[0]);

			for(int i=0;i<num_elem;i++){
				cout<<""<<options[i]<<""<<endl;
			}

			//BEST COMBINATIONS

			//1.SOBEL or LAPLACE + DISTANCE_TRANSFORM
			//2.OTSU + HARRIS_CORNER_DETECTION;
			//3.



			printf("\nNOW-> CHOOSE YOUR OPTION:");
			scanf("%i",&opt_menu);

			switch(opt_menu){
						case 1:Spatial_Filtering();break;
						case 2:/*Filtering_Frequency_Domain()*/;break;
						case 3:/*Image_Rest_Reconst()*/;break;
						case 4:/*Color_Image_Processing()*/;break;
						case 5:/*Wavelets_Multiresolution()*/;break;
						case 6:/*Image_Compression*/;break;
						case 7:Morphological_Processing();break;
						case 8:Image_Segmentation();break;
						case 9:/*Representation_Description*/;break;
						case 10:Object_Recognition();break;
						case 11:Store_Images();break;
						case 12:get_out=false;break;

			}

		}

}

			//SEGMENTATION FUSING METHODS.


			//4.CHANGE STRUCTURAL ELEMENT TO CV_SHAPE AND CREATE A SHAPE LIKE THE BUILDING.
																				  //TOMORROW.IMPLEMENTATION

			//6.CHECK WICH FEATURES EXTRACT THE SIFT METHOD							TOMORROW.RESEARCH

			//7.TEMPLATE MATCHING FOR OBJECTS.										TOMORROW.RESEARCH
				//USE matchTemplate OpenCV Function.
			//8.CASCADE CLASSIFIER.                                         		TOMORROW.RESEARCH
			//9.LATENT SVM. SVM FOR OBJECT DETECTION.								TOMORROW.RESEARCH
			//10.RANDOMIZED HOUGH TRANSFORM											TOMORROW.IMPLEMENTATION
			//11.GAUSSIAN CLASSIFIERS.												TOMORROW.IMPLEMENTATION
			//12.KMEANS FOR GROUPING THE REGIONS WITH A SIMILAR AREA?.				TOMORROW.RESEARCH
			//13.REFINEMENT OF CORNERS.												24.RESEARCH.
			//14.HAAR CLASSIFIER.													24.RESEARCH.
			//17.IMAGE MOMENTS AND HU MOMENTS.										TOMORROW.IMPLEMENTATION.




			//1. SPATIAL FILTERING.

				//ENHANCE CONTRAST OF THE IMAGE WITH HISTOGRAM EQUALIZATION.
				//HISTOGRAM STRETCHING FOR CONSTRAST STRETCHING.
				//PIXEL INTENSITY ADJUSTMENT TECHNIQUES USING OPENCV
				//LOCAL INTENSITIES WEIGHTING.

				//2.6 HISTOGRAM CALCULATIONS
				//compute_histogram(img)
				//2.9 IMPLEMENT LoG. 				TODAY.
				//Spatial_Filtering(5);

				//2.10 EIGENVALUES AND EIGENVECTORS.
				//2.11 LOCAL BINARY PATTERNS AND HOG.
				//2.12 MULTI-VARIATE SHAPE MATCHING WITH GAUSSIANS.


				//2.13.SHAPE ANALYSIS.

					//1.1. EXTRACT CONTOURS - CREATE FEATURE DESCRIPTOR WITH A KERNEL OF THE APPROXIMATE SIZE OF THE
						   //BUILDING SIZE. THESE CONTOUR POINTS SHOULD MATCH AN SPECIFIC CRITERIA : THEY SHOULD BE PART OF
						   //A LINE OR HAVE AN AREA OF AN SPECIFIC SIZE.
						   //ALL THE EXTRACTED SHAPES WITH A SIMILAR AREA MUST BE COMPARED.
					//1.2. THERE ARE TWO TYPES OF SHAPE MATCHING, FEATURE BASED MODELS(SPATIAL ARRANGEMENTS) AND
						//APPEARANCE/BRIGHTNESS MODELS.
					//1.3.FEATURE-BASED MODELS - DISTANCE TRANSFORM, EIGENVALUES, DECISION TREES FOR RECOGNITION(LEARNING
					//DISCRIMINATIVE SPATIAL CONFIGURATIONS OF KEYPOINT).
					//1.4.APPEARANCE MODELS - PCA(PRINCIPAL COMPONENT ANALYSIS), RICH DESCRIPTORS(GREY VALUES)
					//1.5 IMAGE MOMENTS ARE GOOD TO FIND SHAPES REGARDLESS OF THE ROTATION, SIZE OR SCALE OF THE SHAPE.

				//2.14 FUZZY SPACIAL TEHCNIQUES.


			//5.SEGMENTATION

				//5.BACK PROJECTION COMPARING WITH HISTOGRAM OF THE BUILDING OUTLINES.	TOMORROW.


				//IMPLEMENT A MORPHOLOGICAL PROCESSING FUNCTION THAT CALCULATES THE NUMBER OF POINTS PER SHAPE, OUR BUILDING HAS
				//FROM 4 TO 8 POINTS. FIND CONTOURS -> approxPolyDP -> Threshold(number of points) -> DRAW SHAPES.
				//REFINEMENT OF CORNERS. -> USE UNIFORM QUADRATIC BASIS SPLINE METHODOLOGY.









