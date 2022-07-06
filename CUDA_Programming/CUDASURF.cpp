
#include <iostream>
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
//#include "opencv2/cudalegacy.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudastereo.hpp"
#include "opencv2/cudawarping.hpp"
//#include "opencv2/cudev.hpp"
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
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect/objdetect.hpp"

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
#include "SURFKNNFLANN.h"

using namespace Eigen;
using namespace cv;
using namespace std;
using namespace cv::cuda;


using namespace std;
using namespace cv;
using namespace cv::cuda;


int main(int argc, char* argv[])
{

    GpuMat img1, img2;
    img1.upload(imread("/home/xisco/Escritorio/CVISION/PROJECTS/LOGO_DETECTION/MESSI_1.jpg", IMREAD_GRAYSCALE));
    img2.upload(imread("/home/xisco/Escritorio/CVISION/PROJECTS/LOGO_DETECTION/MESSI_2.jpg", IMREAD_GRAYSCALE));

    cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    SURF_CUDA surf;

    // detecting keypoints & computing descriptors
    GpuMat keypoints1GPU, keypoints2GPU;
    GpuMat descriptors1GPU, descriptors2GPU;

    surf(img1, GpuMat(), keypoints1GPU, descriptors1GPU);
    surf(img2, GpuMat(), keypoints2GPU, descriptors2GPU);

    cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
    cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;

    // matching descriptors
    Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
    vector<DMatch> matches;
    matcher->match(descriptors1GPU, descriptors2GPU, matches);

    // downloading results
    vector<KeyPoint> keypoints1, keypoints2;
    vector<float> descriptors1, descriptors2;
    surf.downloadKeypoints(keypoints1GPU, keypoints1);
    surf.downloadKeypoints(keypoints2GPU, keypoints2);
    surf.downloadDescriptors(descriptors1GPU, descriptors1);
    surf.downloadDescriptors(descriptors2GPU, descriptors2);

    // drawing the results
    Mat img_matches;
    drawMatches(Mat(img1), keypoints1, Mat(img2), keypoints2, matches, img_matches);

    namedWindow("matches", 0);
    imshow("matches", img_matches);
    waitKey(0);

    return 0;
}

