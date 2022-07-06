
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "stdio.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include "CVision.h"
#include "SIFT.h"
#include "SURF.h"
#include <stdio.h>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <time.h>
#include <vector>



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


using namespace Eigen;

#undef _GLIBCXX_DEBUG
using namespace cv;
using namespace std;



int main(){


	//SURF det_surf;
	SIFT det_sift;
	vector<Mat> img;
	Mat input;
	Mat gray_img;
	char path[200];

	//LOAD IMAGES OR VIDEO.

	input = imread("/home/xisco/Escritorio/CVISION/PROJECTS/LOGO_DETECTION/messi.jpg");
	cvtColor(input,gray_img,CV_BGR2GRAY);
	img.push_back(gray_img);

	//CALL METHOD.

	det_sift.loadImages(img);
	//det_surf.loadImages(img);
	det_sift.Sift_Detection();
	//det_surf.Surf_Detection_Method();
}

