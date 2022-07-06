//////////////////////////////////////
////////GENERIC OPENCV_LIBRARIES/////
/////////////////////////////////////

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
#include "opencv2/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
//#include <opencv2/nonfree/nonfree.hpp>

//////////////////////////////////
/////////XIMGPROC/////////////////
//////////////////////////////////
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

//////////////////////CUDA//////////////////
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudastereo.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/flann.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
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
using namespace cv::cuda;
using namespace cv::xfeatures2d;

struct Match_Logo
{
  float distance;
  string path_train;
};

//struct CV_EXPORTS_W_SIMPLE Match_Logo
//{
//    CV_WRAP Match_Logo() : queryIdx(-1), trainIdx(-1), imgIdx(-1), distance(FLT_MAX) {}
//    CV_WRAP DMatch( int _queryIdx, int _trainIdx, float _distance ) :
//            queryIdx(_queryIdx), trainIdx(_trainIdx), imgIdx(-1), distance(_distance) {}
//    CV_WRAP DMatch( int _queryIdx, int _trainIdx, int _imgIdx, float _distance ) :
//            queryIdx(_queryIdx), trainIdx(_trainIdx), imgIdx(_imgIdx), distance(_distance) {}

//    CV_PROP_RW string match; // query descriptor index
//    CV_PROP_RW int trainIdx; // train descriptor index
//    CV_PROP_RW float distance;

//    // less is better
//    bool operator<( const DMatch &m ) const
//    {
//        return distance < m.distance;
//    }
//};

class SIFT_SURF_HOG {

public:
	/////////////HOG//////////////////////////////////////////////////////
	Mat channels_object[3];
	Mat channels_scene[3];
	Mat gx[3];
	Mat gy[3];
	Mat GX,GY;
	Mat magnit, angle;
    vector<vector<double> > concatenated_bins =  vector<vector<double> >(4,vector<double>(9));
	vector<vector<double> > concatenated_bins_normalized = vector<vector<double> >(4,vector<double>(9));
	vector<vector<vector<double> > > concatenated_histogram;
	vector<vector<vector<double> > > concatenated_histogram_test;
	vector<vector<vector<double> > > concatenated_histogram_train;

	///////////SIFT AND SURF//////////////////////////////////////////////////////////////////////
	vector<KeyPoint> kpoints;
	vector<KeyPoint> kpoints_test;
	vector<KeyPoint> kpoints_train;
	Mat descriptors_surf;
	Mat descriptors_sift;
    /////////////OTHER DESCRIPTORS////////////////////////////////////////////////////////////////
    Mat img_object;
    Mat img_scene;
    Mat img_gray;
    Mat descriptors_akaze;
    Mat descriptors_brisk;
    Mat descriptors_fast;
    Mat descriptors_gftt;
    Mat descriptors_kaze;
    Mat descriptors_mser;
    Mat descriptors_simple_blob_detector;

	vector<Mat> descriptors_train;
    vector<Mat> descriptors_test;

    Mat centers_sift_surf = Mat(40,128,CV_32FC1);
    Mat centers_kaze = Mat(40,64,CV_32FC1);
    Mat centers_brisk = Mat(40,64, CV_32FC1);

    Mat Logo_Descriptor_Sift_Surf = Mat(40,128,CV_32FC1);
    Mat Logo_Descriptor_Kaze = Mat(10,64,CV_32FC1);
    Mat Logo_Descriptor_Brisk = Mat(10,64,CV_32FC1);

    vector<vector<cv::DMatch> > good_matches;
    vector<Match_Logo> match_logo;
    Mat centers;
	Mat points_train;
	Mat points_test;

	vector<DMatch> distances;


	/////////////SURF/////////////////////////////////////////////////////////////////////////
	void Compute_CPU_Surf_Descriptors(string sceneInputFile);
	void SURF_processWithGpu(string objectInputFile, string sceneInputFile, string outputFile);
	///////////SIFT/////////////////////////////////////////////////////////////////////////
	void Compute_CPU_Sift_Descriptors(string objectInputFile);
	void SIFT_processWithGpu(string objectInputFile, string sceneInputFile, string outputFile);
    //////////OTHER DESCRIPTORS//////////////////////////////////////////////////////////////
    void Compute_CPU_Agast_Descriptors(string objectInputFile);
    void Compute_CPU_Akaze_Descriptors(string objectInputFile);
    void Compute_CPU_Brisk_Descriptors(string objectInputFile);
//    void Compute_CPU_Fast_Descriptors(string objectInputFile);
//    void Compute_CPU_GFTT_Descriptors(string objectInputFile);
    void Compute_CPU_Kaze_Descriptors(string objectInputFile);
    void Compute_CPU_Mser_Descriptors(string objectInputFile);
    void Compute_CPU_SimpleBlobDetector_Descriptors(string objectInputFile);

    //////////UTILITIES FUNCTIONS////////////////////////////////////////////////////////////
	void Concatenate_Descriptors();
    void read_Descriptors_from_File(const char * outputFile);
	void write_Descriptors_to_File(const char * outputFile);
	void Compare_Descriptors(const char * train_path);

    void Compute_Distance(const char * path_to_descriptor);
	void write_Distances_to_File(const char * outputFile);
    void Find_Good_Matches(vector<vector<DMatch> > matches, int i);
	void localizeInImage(const std::vector<DMatch>& good_matches,
			const std::vector<KeyPoint>& keypoints_object,
			const std::vector<KeyPoint>& keypoints_scene, const Mat& img_object,
			const Mat& img_matches);
	Mat Otsu_Threshold(Mat img);
    ///////////HOG//////////////////////////
	void Histogram_of_Gradients(int train_test);
	void Gradient_Images(string objectInputFile);
	void Gradient_Color();
	void Compute_Logo_HOG(String LogoPath);
	void Compare_Images(String object_file_path);
	void Compute_Block(int start_cols, int start_rows);
	void Compare_Concatenated_Histograms();
	Mat Vector_to_Matrix( vector<vector<vector<double> > > vector);
	void draw_Matches(int min_test);
	void draw_Matches_HOG(int min_test, int min_train);
    ///////CLASSIFICATION PROCESS///////////
	void Compare_Descriptors();
    void Compare_Set_Descriptors(const char * path_to_descriptor);
};


bool error_handling(std::fstream & file){
	bool error = false;

	if(file.is_open()){ cout<<"The file is opened"<<endl;}

	else if(file.good()){ cout<<"The stream doesn't present any problem"<<endl;}

	else if(file.fail()){cout<<"Its fails to open....and this is why its failing"<<endl;
		string s;
		if(file.bad()){cout<<"The stream is in a bad state"<<endl;}
		else{cout<<"It fails for other reasons"<<endl;} error = true;
		file.clear();
	}
	else{cout<<"This is the end of the file"<<endl;}

	return error;
}


void SIFT_SURF_HOG::read_Descriptors_from_File(const char * File){

	ifstream file(File,ios_base::out);
	int row = 0;
	int col = 0;
    std::string line;
    if(file.is_open()){

    while(std::getline(file,line)){
        std::stringstream stream(line);
        double x;
        if(row<40){
             col = 0;  // reset column counter
             while (stream>>x) {
                 Logo_Descriptor_Sift_Surf.at<float>(row,col) = x;
                 col++;
             }
        }
        if(row<50){
            col = 0;  // reset column counter
            while (stream>>x) {
                Logo_Descriptor_Kaze.at<float>(row-40,col) = x;
                col++;
            }
        }
        if(row<60){
            col = 0;  // reset column counter
            while (stream>>x) {
                Logo_Descriptor_Brisk.at<float>(row-50,col) = x;
                col++;
            }
        }

     row++;
    }
    descriptors_test.push_back(Logo_Descriptor_Sift_Surf);
    descriptors_test.push_back(Logo_Descriptor_Kaze);
    descriptors_test.push_back(Logo_Descriptor_Brisk);
  }
}

void SIFT_SURF_HOG::write_Descriptors_to_File(const char * outputFile){

	  puts(outputFile);
	  bool fails = false;
	  std::fstream fout(outputFile,ios_base::out);
	  fails = error_handling(fout);

	  if(!fout.good())
	    {
	        cout<<"File Not Opened"<<endl;  return;
	    }
	  else{
        for(int i=0;i<descriptors_train.size();i++){
            for(int j=0; j<descriptors_train[i].rows; j++)
            {
                for(int k=0; k<descriptors_train[i].cols; k++)
                {
                    fout<<descriptors_train[i].at<float>(j,k)<<"\t";
                }
                fout<<endl;
            }
        }

      }
	   descriptors_sift.resize(0);
	   descriptors_surf.resize(0);
       descriptors_brisk.resize(0);
       descriptors_kaze.resize(0);
       descriptors_train.resize(0);
       centers.resize(0);

	    fout.close();
}

//void SIFT_SURF_HOG::Brute_Force_Matcher(){

	//		BruteForceMatcher<Distance  > matcher_brute;
	//		vector<DMatch> matches_brute;
	//		matcher_brute.match(centers,Logo_Descriptor, matches_brute, noArray());
//}

void SIFT_SURF_HOG::Concatenate_Descriptors(){

	Mat descriptors_scene;
    Mat centers_concat;
	vconcat(descriptors_surf, descriptors_sift,descriptors_scene);

    Mat bestLabels_sift_surf;
    Mat bestLabels_brisk;
    Mat bestLabels_kaze;

	int attempts = 20;
    int clusts_sift_surf = 40;
    int clusts_brisk = 10;
    int clusts_kaze = 10;
	double eps = 0.001;

	if(descriptors_scene.rows < 100){
		descriptors_sift.resize(0);
		descriptors_surf.resize(0);
		return;
	}
	cout<<"CLUSTERING........."<<endl;

    double compactness_sift_surf = kmeans(descriptors_scene, clusts_sift_surf, bestLabels_sift_surf, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, attempts, eps),
                                            attempts, KMEANS_RANDOM_CENTERS, centers_sift_surf);

    cout<<"CLUSTER FOR SIFT AND SURF DONE"<<endl;
    double compactness_kaze = kmeans(descriptors_kaze, clusts_kaze, bestLabels_kaze, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, attempts, eps),
                                      attempts, KMEANS_RANDOM_CENTERS, centers_kaze);

    cout<<"CLUSTER FOR KAZE DONE"<<endl;
    double compactness_brisk = kmeans(descriptors_brisk, clusts_brisk, bestLabels_brisk, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, attempts, eps),
                                          attempts, KMEANS_RANDOM_CENTERS, centers_brisk);
    cout<<"CLUSTERED DONE"<<endl;


    descriptors_train.push_back(centers_sift_surf);
    descriptors_train.push_back(centers_kaze);
    descriptors_train.push_back(centers_brisk);

    cout<<"The size of the concatenated centers is:"<<descriptors_train.size()<<endl;
    waitKey(0);
}

void SIFT_SURF_HOG::write_Distances_to_File(const char * outputFile){
	  bool fails = false;
	  fstream fout;
	  fout.open(outputFile,ios_base::in);
	  cout<<"WRITING DISTANCES TO:"<<outputFile<<endl;
	  fails = error_handling(fout);
	  if(fout.is_open())
	    {
	        cout<<"File Not Opened"<<endl;  return;
	    }
	  else{
	    for(int i=0; i<distances.size(); i++)
	    {
	       fout<<distances[i].distance<<"\n";
	    }
	    fout<<endl;
	  }
	  distances.resize(0);
	  fout.close();
}


void SIFT_SURF_HOG::Find_Good_Matches( vector<vector<cv::DMatch> > matches,int i){

    for (int k = 0; k < std::min(descriptors_test[i].rows - 1, (int)matches.size()); k++)
    {
        if ( (matches[k][0].distance < 0.70*(matches[k][1].distance)) &&
                ((int)matches[k].size() <= 2 && (int)matches[k].size()>0) )
        {
          good_matches[i].push_back( matches[k][0] );
        }
    }
}

void SIFT_SURF_HOG::Compute_Distance(const char * path_to_descriptor){

    float dist=0;
    for(int i=0;i<good_matches.size();i++){
        for(int j=0;j<good_matches[i].size();j++){
            dist += good_matches[i][j].distance;
        }
    }
    dist = dist/good_matches.size();
    Match_Logo match;
    match.distance = dist;
    match.path_train = path_to_descriptor;
    match_logo.push_back(match);
}

void SIFT_SURF_HOG::Compare_Descriptors(const char * Descriptor_Path){

    FlannBasedMatcher matcher;
    vector<vector<cv::DMatch> > matches_sift_surf;
    vector<vector<cv::DMatch> > matches_kaze;
    vector<vector<cv::DMatch> > matches_brisk;

    matcher.knnMatch( centers_sift_surf, descriptors_test[0], matches_sift_surf,2);
    matcher.knnMatch(centers_kaze, descriptors_test[1], matches_kaze,2);
    matcher.knnMatch(centers_brisk, descriptors_test[2],matches_brisk,2);

    Find_Good_Matches(matches_sift_surf,0);
    Find_Good_Matches(matches_kaze,1);
    Find_Good_Matches(matches_brisk,2);

    Compute_Distance(Descriptor_Path);
	}

void SIFT_SURF_HOG::Compute_CPU_Surf_Descriptors(string sceneInputFile)
{

	int minHessian=100;
	FILE* pfile;

	img_object = imread( sceneInputFile, IMREAD_GRAYSCALE );

	if( !img_object.data ) {
		std::cout<< "Error reading  train image." << std::endl;
		return;
	}
	vector<KeyPoint> keypoints_object, keypoints_scene; // keypoints
	Ptr<SURF> surf = SURF::create(minHessian,5,4, true, false);
	surf->detectAndCompute( img_object, noArray(), keypoints_scene, descriptors_surf );


	if ( descriptors_surf.empty() )
	   cvError(0,"MatchFinder","SURF descriptor empty",__FILE__,__LINE__);

	cout<<"NUMBER OF SURF DESCRIPTORS"<<descriptors_surf.rows<<endl;


}
void SIFT_SURF_HOG::SURF_processWithGpu(string objectInputFile, string sceneInputFile, string outputFile)
{
	int minHessian=400;
	// Load the image from the disk
	Mat img_object = imread( objectInputFile, IMREAD_GRAYSCALE ); // surf works only with grayscale images
	Mat img_scene = imread( sceneInputFile, IMREAD_GRAYSCALE );
	if( !img_object.data || !img_scene.data ) {
		std::cout<< "Error reading images." << std::endl;
		return;
	}

	// Copy the image into GPU memory
	cuda::GpuMat img_object_Gpu( img_object );
	cuda::GpuMat img_scene_Gpu( img_scene );

	// Start the timer - the time moving data between GPU and CPU is added
//	GpuTimer timer;
//	timer.Start();

	cuda::GpuMat keypoints_scene_Gpu, keypoints_object_Gpu; // keypoints
	cuda::GpuMat descriptors_scene_Gpu, descriptors_object_Gpu; // descriptors (features)

	//-- Steps 1 + 2, detect the keypoints and compute descriptors, both in one method
	cuda::SURF_CUDA surf( minHessian);
	surf( img_object_Gpu, cuda::GpuMat(), keypoints_object_Gpu, descriptors_object_Gpu );
	surf( img_scene_Gpu, cuda::GpuMat(), keypoints_scene_Gpu, descriptors_scene_Gpu );
	//cout << "FOUND " << keypoints_object_Gpu.cols << " keypoints on object image" << endl;
	//cout << "Found " << keypoints_scene_Gpu.cols << " keypoints on scene image" << endl;

	//-- Step 3: Matching descriptor vectors using BruteForceMatcher
	//Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher();
	vector< vector< DMatch> > matches;
	//matcher->knnMatch(descriptors_object_Gpu, descriptors_scene_Gpu, matches, 2);

	// Downloading results  Gpu -> Cpu
	vector< KeyPoint > keypoints_scene, keypoints_object;
	//vector< float> descriptors_scene, descriptors_object;
	surf.downloadKeypoints(keypoints_scene_Gpu, keypoints_scene);
	surf.downloadKeypoints(keypoints_object_Gpu, keypoints_object);
	//surf.downloadDescriptors(descriptors_scene_Gpu, descriptors_scene);
	//surf.downloadDescriptors(descriptors_object_Gpu, descriptors_object);

//	timer.Stop();
//	printf( "Method processImage() ran in: %f msecs.\n", timer.Elapsed() );

	//-- Step 4: Select only goot matches
	//vector<Point2f> obj, scene;
	std::vector< DMatch > good_matches;
	for (int k = 0; k < std::min(keypoints_object.size()-1, matches.size()); k++)
	{
		if ( (matches[k][0].distance < 0.6*(matches[k][1].distance)) &&
				((int)matches[k].size() <= 2 && (int)matches[k].size()>0) )
		{
			// take the first result only if its distance is smaller than 0.6*second_best_dist
			// that means this descriptor is ignored if the second distance is bigger or of similar
			good_matches.push_back(matches[k][0]);
		}
	}

	//-- Step 5: Draw lines between the good matching points
	Mat img_matches;
	drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
			good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::DEFAULT );

	//-- Step 6: Localize the object inside the scene image with a square
	localizeInImage( good_matches, keypoints_object, keypoints_scene, img_object, img_matches );

	//-- Step 7: Show/save matches
	//imshow("Good Matches & Object detection", img_matches);
	//waitKey(0);
	imwrite(outputFile, img_matches);

	//-- Step 8: Release objects from the GPU memory
	surf.releaseMemory();
//	matcher.release();
	img_object_Gpu.release();
	img_scene_Gpu.release();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////LOCALIZATION OF THE OBJECT////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//It searches for the right position, orientation and scale of the object in the scene based on the good_matches.

void SIFT_SURF_HOG::localizeInImage(const std::vector<DMatch>& good_matches,
		const std::vector<KeyPoint>& keypoints_object,
		const std::vector<KeyPoint>& keypoints_scene, const Mat& img_object,
		const Mat& img_matches)
{
	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	for (int i = 0; i < good_matches.size(); i++) {
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	try {
		Mat H = findHomography(obj, scene, RANSAC);
		//-- Get the corners from the image_1 ( the object to be "detected" )
		std::vector<Point2f> obj_corners(4);
		obj_corners[0] = cvPoint(0, 0);
		obj_corners[1] = cvPoint(img_object.cols, 0);
		obj_corners[2] = cvPoint(img_object.cols, img_object.rows);
		obj_corners[3] = cvPoint(0, img_object.rows);
		std::vector<Point2f> scene_corners(4);

		perspectiveTransform(obj_corners, scene_corners, H);
		// Draw lines between the corners (the mapped object in the scene - image_2 )
		line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0),
				scene_corners[1] + Point2f(img_object.cols, 0),
				Scalar(255, 0, 0), 4);
		line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0),
				scene_corners[2] + Point2f(img_object.cols, 0),
				Scalar(255, 0, 0), 4);
		line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0),
				scene_corners[3] + Point2f(img_object.cols, 0),
				Scalar(255, 0, 0), 4);
		line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0),
				scene_corners[0] + Point2f(img_object.cols, 0),
				Scalar(255, 0, 0), 4);
	} catch (Exception& e) {}

}

////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////SIFT//////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

void show_KeyPoints(vector<KeyPoint> KPoints, Mat img){

	for(int i=0; i<KPoints.size();i++){
		circle( img, Point( KPoints[i].pt.x,KPoints[i].pt.y), 2,  Scalar(0,0,255), 2, 8, 0 );
	}
	imshow("KEYPOINTS ON IMAGE",img);
}

void SIFT_SURF_HOG::Compute_CPU_Sift_Descriptors(string objectInputFile)
{
	Mat img1_c=imread(objectInputFile,CV_LOAD_IMAGE_COLOR);
	Mat img1;
	cv::cvtColor(img1_c,img1,CV_BGR2GRAY);

	Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(2000,7);
	xfeatures2d::SiftDescriptorExtractor extractor;

	sift->detectAndCompute(img1, noArray(),kpoints_test, descriptors_sift);
	//sift->detectAndCompute(img2, noArray(),key_points_2,descriptors_2);

	if (descriptors_sift.empty())
	   cvError(0,"MatchFinder","SIFT 1st descriptor empty",__FILE__,__LINE__);

	cout<<"NUMBER OF SIFT DESCRIPTORS"<<descriptors_sift.rows<<endl;



//
//	std::vector< DMatch > good_matches;
//	for (int k = 0; k < std::min(descriptors_2.rows - 1, (int)matches.size()); k++)
//	{
//	if ( (matches[k][0].distance < 0.60*(matches[k][1].distance)) &&
//					((int)matches[k].size() <= 2 && (int)matches[k].size()>0) )
//	{
//		// take the first result only if its distance is smaller than 0.6*second_best_dist
//		// that means this descriptor is ignored if the second distance is bigger or of similar
//		good_matches.push_back( matches[k][0] );
//		}
//	}
//	//-- Step 5: Draw lines between the good matching points
//	Mat img_matches;
//
//	drawMatches( img1, key_points_1, img2, key_points_2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
//	vector<char>(), DrawMatchesFlags::DEFAULT );
//
//	//-- Step 6: Localize the object inside the scene image with a square
//	localizeInImage( good_matches,key_points_1, key_points_2, img_object, img_matches );
//	//-- Step 7: Show/save matches
//	//imshow("Good Matches & Object detection", img_matches);
//	//waitKey(0);
//	imshow("IMG_MATCHES FOR SIFT METHOD",img_matches);
//	cv::resize(img_matches,img_matches,Size(1200,720));
//	imshow("DOWNSAMPLED MATCHING IMAGE FOR SIFT",img_matches);
//	waitKey(0);
}

void SIFT_SURF_HOG::SIFT_processWithGpu(string objectInputFile, string sceneInputFile, string outputFile)
{
	int minHessian=100;
	// Load the image from the disk
	Mat img_object = imread( objectInputFile, IMREAD_GRAYSCALE ); // surf works only with grayscale images
	Mat img_scene = imread( sceneInputFile, IMREAD_GRAYSCALE );
	if( !img_object.data || !img_scene.data ) {
		std::cout<< "Error reading images." << std::endl;
		return;
	}

	// Copy the image into GPU memory
	cuda::GpuMat img_object_Gpu( img_object );
	cuda::GpuMat img_scene_Gpu( img_scene );

	// Start the timer - the time moving data between GPU and CPU is added
//	GpuTimer timer;
//	timer.Start();

	cuda::GpuMat keypoints_scene_Gpu, keypoints_object_Gpu; // keypoints
	cuda::GpuMat descriptors_scene_Gpu, descriptors_object_Gpu; // descriptors (features)

	//-- Steps 1 + 2, detect the keypoints and compute descriptors, both in one method
	cuda::SURF_CUDA surf( minHessian);
	surf( img_object_Gpu, cuda::GpuMat(), keypoints_object_Gpu, descriptors_object_Gpu );
	surf( img_scene_Gpu, cuda::GpuMat(), keypoints_scene_Gpu, descriptors_scene_Gpu );
	//cout << "FOUND " << keypoints_object_Gpu.cols << " keypoints on object image" << endl;
	//cout << "Found " << keypoints_scene_Gpu.cols << " keypoints on scene image" << endl;

	//-- Step 3: Matching descriptor vectors using BruteForceMatcher
	//Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher();
	vector< vector< DMatch> > matches;
	//matcher->knnMatch(descriptors_object_Gpu, descriptors_scene_Gpu, matches, 2);

	// Downloading results  Gpu -> Cpu
	vector< KeyPoint > keypoints_scene, keypoints_object;
	//vector< float> descriptors_scene, descriptors_object;
	surf.downloadKeypoints(keypoints_scene_Gpu, keypoints_scene);
	surf.downloadKeypoints(keypoints_object_Gpu, keypoints_object);
	//surf.downloadDescriptors(descriptors_scene_Gpu, descriptors_scene);
	//surf.downloadDescriptors(descriptors_object_Gpu, descriptors_object);

//	timer.Stop();
//	printf( "Method processImage() ran in: %f msecs.\n", timer.Elapsed() );

	//-- Step 4: Select only goot matches
	//vector<Point2f> obj, scene;
	std::vector< DMatch > good_matches;
	for (int k = 0; k < std::min(keypoints_object.size()-1, matches.size()); k++)
	{
		if ( (matches[k][0].distance < 0.6*(matches[k][1].distance)) &&
				((int)matches[k].size() <= 2 && (int)matches[k].size()>0) )
		{
			// take the first result only if its distance is smaller than 0.6*second_best_dist
			// that means this descriptor is ignored if the second distance is bigger or of similar
			good_matches.push_back(matches[k][0]);
		}
	}

	//-- Step 5: Draw lines between the good matching points
	Mat img_matches;
	drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
			good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::DEFAULT );

	//-- Step 6: Localize the object inside the scene image with a square
	localizeInImage( good_matches, keypoints_object, keypoints_scene, img_object, img_matches );

	//-- Step 7: Show/save matches
	//imshow("Good Matches & Object detection", img_matches);
	//waitKey(0);
	imwrite(outputFile, img_matches);

	//-- Step 8: Release objects from the GPU memory
	surf.releaseMemory();
//	matcher.release();
	img_object_Gpu.release();
	img_scene_Gpu.release();
}
void Compute_CPU_Agast_Descriptors(string objectInputFile, string sceneInputFile, string outputFile){

}


void SIFT_SURF_HOG::Compute_CPU_Akaze_Descriptors(string objectInputFile){

    img_object = imread(objectInputFile,CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(img_object,img_gray,CV_32FC1);
    vector<KeyPoint> kpoints;

    Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    akaze->detect(img_gray, kpoints, noArray());
    akaze->compute(img_gray, kpoints, descriptors_akaze);
}

void SIFT_SURF_HOG::Compute_CPU_Brisk_Descriptors(string objectInputFile){
    img_object = imread(objectInputFile,CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(img_object, img_gray, CV_32FC1);
    vector<KeyPoint> kpoints;

    Ptr<cv::BRISK> brisk = cv::BRISK::create();
    brisk->detect(img_gray, kpoints, noArray());
    brisk->compute(img_gray, kpoints, descriptors_brisk);
    cout<<"The size of the brisk descriptor is:"<<descriptors_brisk.size()<<endl;
    descriptors_brisk.convertTo(descriptors_brisk,CV_32FC1);
}

//void SIFT_SURF_HOG::Compute_CPU_Fast_Descriptors(string objectInputFile){
//    img_object = imread(objectInputFile,CV_LOAD_IMAGE_COLOR);
//    cv::cvtColor(img_object, img_gray, CV_32FC1);
//    vector<KeyPoint> kpoints;

//    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create();
//    fast->detect(img_gray, kpoints, noArray());
//    fast->compute(img_gray,kpoints, descriptors_fast);

//}

//void SIFT_SURF_HOG::Compute_CPU_GFTT_Descriptors(string objectInputFile){

//    img_object = imread(objectInputFile,CV_LOAD_IMAGE_COLOR);
//    cv::cvtColor(img_object, img_gray, CV_32FC1);
//    vector<KeyPoint> kpoints;

//    Ptr<GoodFeaturesToTrackDetector> gftt = GoodFeaturesToTrackDetector::create();
//    gftt->detect(img_gray, kpoints, noArray());
//    gftt->compute(img_gray, kpoints, descriptors_gftt);
//}

void SIFT_SURF_HOG::Compute_CPU_Kaze_Descriptors(string objectInputFile){

    img_object = imread(objectInputFile,CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(img_object, img_gray, CV_32FC1);
    vector<KeyPoint> kpoints;

    Ptr<cv::KAZE> kaze = cv::KAZE::create();
    kaze->detect(img_gray, kpoints, noArray());
    kaze->compute(img_gray, kpoints, descriptors_kaze);
    cout<<"The size of the kaze descriptor is:"<<descriptors_kaze.size()<<endl;
    cout<<"The type of the kaze descriptor is:"<<descriptors_kaze.type()<<endl;

}

void SIFT_SURF_HOG::Compute_CPU_Mser_Descriptors(string objectInputFile){

    img_object = imread(objectInputFile,CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(img_object, img_gray, CV_32FC1);
    vector<KeyPoint> kpoints;

    Ptr<cv::MSER> mser = cv::MSER::create();
    mser->detect(img_gray, kpoints, noArray());
    mser->compute(img_gray, kpoints, descriptors_mser);
}

void SIFT_SURF_HOG::Compute_CPU_SimpleBlobDetector_Descriptors(string objectInputFile){

        img_object = imread(objectInputFile,CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(img_object, img_gray, CV_32FC1);
    vector<KeyPoint> kpoints;

    Ptr<cv::SimpleBlobDetector> simple_blob_detector = cv::SimpleBlobDetector::create();
    simple_blob_detector->detect(img_gray, kpoints, noArray());
    simple_blob_detector->compute(img_gray, kpoints, descriptors_simple_blob_detector);

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////LOCALIZATION OF THE OBJECT////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////HISTOGRAM OF GRADIENTS////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SIFT_SURF_HOG::Gradient_Images(string objectInputFile){


	Mat mag, angle;
	double max_gx=0,max_gy=0;
	double min_gx=0,min_gy=0;
	int num_elements=0;

	img_object = imread(objectInputFile, CV_LOAD_IMAGE_COLOR );
	img_object.convertTo(img_object,CV_32FC3);
	split(img_object,channels_object);

	num_elements = sizeof(channels_object)/sizeof(channels_object[0]);
	for(int j=0;j<num_elements;j++){

		Sobel(channels_object[j], gx[j],CV_32F, 1,0,1);
		Sobel(channels_object[j], gy[j],CV_32F, 0,1,1);
	}
	Gradient_Color();
}


void SIFT_SURF_HOG::Gradient_Color(){

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
	cv::cartToPolar(GX, GY, magnit, angle, 1);
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


void SIFT_SURF_HOG::Compute_Block(int init_cols, int init_rows){

	float item=0; float weight_one=0; float weight_two=0;
	float start_rows=0;
	vector<double> bins =vector<double>(9);
	 vector<vector<double> >(4,vector<double>(9));
	float start_cols=0;
	start_cols = init_cols;
	start_rows = init_rows;
	vector<vector<double> > concatenated_restored;
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
			}while(start_cols < init_cols + 14);
			start_rows += 8;
			start_cols = init_cols;
			}while(start_rows < init_rows + 14);
			if(concatenated_bins.size() == 8){
				concatenated_restored =  vector<vector<double> >(4,vector<double>(9));
				concatenated_restored.assign(concatenated_bins.begin(),concatenated_bins.begin()+4);
				concatenated_restored = Calculate_norm(concatenated_restored);
				concatenated_histogram.push_back(concatenated_restored);
				concatenated_restored.assign(concatenated_bins.begin()+4,concatenated_bins.end());
				concatenated_restored = Calculate_norm(concatenated_restored);
				concatenated_histogram.push_back(concatenated_restored);
				concatenated_bins.resize(0);
				return;
			}
			concatenated_bins = Calculate_norm(concatenated_bins);
			concatenated_histogram.push_back(concatenated_bins);
			concatenated_bins.resize(0);
}

void SIFT_SURF_HOG::Histogram_of_Gradients(int train_test){

	double normalized_value=0.0;
	float item=0; float weight_one=0; float weight_two=0;
	int init_cols=0,init_rows=0;

	do{
		do{
			KeyPoint point = KeyPoint(init_cols,init_rows,0,0,0,0);
			kpoints.push_back(point);
			//(Point2f(init_cols, init_rows));
			Compute_Block(init_cols, init_rows);
			init_cols += 8;
			}while(init_cols < magnit.cols -15);
			init_rows += 8;
			init_cols=0;
			}while(init_rows < magnit.rows - 15);

			if(train_test == 0){
				concatenated_histogram_test = concatenated_histogram;
				kpoints_test = kpoints;
				cout<<"SIZE KPOINTS TEST IMAGE:"<<kpoints_test.size()<<endl;
			}

			else if(train_test == 1){
				concatenated_histogram_train = concatenated_histogram;
				kpoints_train = kpoints;
				cout<<"SIZE KPOINTS TRAIN IMAGE:"<<kpoints_train.size()<<endl;
			}
			concatenated_histogram.resize(0);
			kpoints.resize(0);
}

void SIFT_SURF_HOG::draw_Matches(int min_test){

	int cells_per_row=img_object.cols/4;
	int row = min_test/cells_per_row;
	int number = min_test - row*cells_per_row;
	number *= 4;
	row *= 4;

	Point p1 = Point(row,number);
	Point p2= Point(row+8,number+8);
	rectangle(img_object ,p1,p2, Scalar( 255, 0 , 0 ),  2, 4 );
	imshow("MATCHING HOG",img_object);
	waitKey(1000);
}


Mat SIFT_SURF_HOG::Vector_to_Matrix( vector<vector<vector<double> > > vector){

	Mat matrix = Mat(vector.size(),36,CV_32FC1,Scalar::all(0));

	for(int i=0;i<vector.size();i++){
		for(int j=0; j<vector[i].size();j++){
			for(int k=0;k<vector[i][j].size();k++){
				float y1 = (j*4) + k;
				matrix.at<float>(i,y1)=vector[i][j][k];
				}
		}
	}
	return matrix;
}
void SIFT_SURF_HOG::Compare_Concatenated_Histograms(){

	float diff=0;
	float min_test=0;
	float min_train=0;
	float value=0;
	float min=10000;
	Mat bestLabels_scene;

	points_train = Vector_to_Matrix(concatenated_histogram_train);
	points_test = Vector_to_Matrix(concatenated_histogram_test);

	cout<<"1"<<endl;
	FlannBasedMatcher matcher; // FLANN - Fast Library for Approximate Nearest Neighbors
	vector<vector<DMatch> > matches;
	matcher.knnMatch(points_train, points_test, matches, 2 ); // find the best 2 matches of each descriptor

	cout<<"matches size:"<<matches.size()<<endl;
	std::vector<DMatch> good_matches;
	for (int k = 0; k < std::min(points_train.rows - 1, (int)matches.size()); k++)
	{
		if ( (matches[k][0].distance < 0.6*(matches[k][1].distance)) &&
				((int)matches[k].size() <= 2 && (int)matches[k].size()>0) ){
			// take the first result only if its distance is smaller than 0.6*second_best_dist
			// that means this descriptor is ignored if the second distance is bigger or of similar
			good_matches.push_back( matches[k][0] );
		}
	}
	cout<<"Lets go to draw the matches"<<endl;
	Mat img_matches;

	drawMatches( img_object, points_test, img_scene, points_train,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::DEFAULT );

	cout<<"4"<<endl;
	localizeInImage( good_matches, points_test, points_train, img_object, img_matches );

	imshow("IMG_MATCHES FOR HOG",img_matches);
	cv::resize(img_matches,img_matches,Size(1200,720));
	imshow("DOWNSAMPLED MATCHING IMAGE FOR HOG",img_matches);

}


void SIFT_SURF_HOG::Compute_Logo_HOG(String LogoPath){
	Gradient_Images(LogoPath);
	Histogram_of_Gradients(1);
}

void SIFT_SURF_HOG::Compare_Images(String object_file_path){
	Gradient_Images(object_file_path);
	Histogram_of_Gradients(0);
	Compare_Concatenated_Histograms();
}




//	Mat centers_train = Mat(100,36,CV_32FC1);
//	Mat centers_test = Mat(100,36,CV_32FC1);
//	int attempts = 20;
//	int clusts = 80;
//	double eps = 0.001;
//
//	cout<<"BEFORE THE KMEANS CLUSTERING PROCESS FOR THE OBJECT"<<endl;
//
//	double compactness_train = kmeans(points_train, clusts, bestLabels_object, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, attempts, eps),
//										attempts, KMEANS_RANDOM_CENTERS, centers_train);
//
//	cout<<"BEFORE THE KMEANS CLUSTERING PROCESS FOR THE SCENE"<<endl;
//	double compactness_test = kmeans(points_test, clusts, bestLabels_scene, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, attempts, eps),
//	  	        						attempts, KMEANS_RANDOM_CENTERS, centers_test);
//
//
//
//
//	cout<<"BEST_LABELS_OBJECT SIZE:"<<bestLabels_object.size()<<endl;
//	cout<<"BEST_LABELS_SCENE SIZE:"<<bestLabels_scene.size()<<endl;
//	waitKey(10);


//	Mat img_scene = imread(Desc[good_matches[0].trainIdx].path, CV_LOAD_IMAGE_COLOR);
//	imshow("IMG_SCENE",img_scene);
//	waitKey(1000);
//	cout<<"IMG READ"<<endl;
//	Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(2000,7);
//	xfeatures2d::SiftDescriptorExtractor extractor;
//	vector<KeyPoint> kpoints_scene;
//	sift->detect(img_scene, kpoints_scene, noArray());
//
//
//	Mat img_matches;
//	drawMatches( img_object, kpoints_test, img_scene, kpoints_scene, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
//																					vector<char>(), DrawMatchesFlags::DEFAULT );
//
//	//-- Step 6: Localize the object inside the scene image with a square
//	localizeInImage( good_matches, kpoints_test, kpoints_scene, img_object, img_matches );
//
//	imshow("MATCHES",img_matches);
//	cv::resize(img_matches,img_matches,Size(1200,720));
//	imshow("LOGO FINAL OUTPUT",img_matches);
