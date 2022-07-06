//////////////////////////////////////
///////////////////////////////////////
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
///////////////////////////////////////////////
//////////////////////CUDA/////////////////////
///////////////////////////////////////////////
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
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

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
#include <iostream>
#include <glob.h>
#include <vector>
#include <string>
#include <iostream>
//#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <dirent.h>
#include "TypesClassification.h"


using namespace Eigen;
using namespace cv;
using namespace std;
using namespace cv::cuda;
using namespace cv::xfeatures2d;
//using namespace boost::filesystem;
//namespace fs = boost::filesystem;

#undef _GLIBCXX_DEBUG

//struct recursive_directory_range {
//    typedef recursive_directory_iterator iterator;
//    recursive_directory_range(path p) : p_(p) {}
//
//    iterator begin() { return recursive_directory_iterator(p_); }
//    iterator end() { return recursive_directory_iterator(); }
//
//    path p_;
//};

 void open(string path, vector<string> & files ) {

	 std::vector <std::string> result;
	 dirent* de;
	 DIR* dp;
	 string point =".";
	 string two_points = "..";
	 errno = 0;
	  dp = opendir( path.empty() ? "." : path.c_str() );
	  if (dp)
	    {
	    while (true)
	      {
	      errno = 0;
	      de = readdir( dp );
	      if (de == NULL) {break;}
	      else if( point.compare(de->d_name) && two_points.compare(de->d_name)){
	    	  files.push_back( std::string( de->d_name ));
	      }
	      }
	    closedir( dp );
	    sort( files.begin(), files.end() );
	 }

 }



//    DIR*    dir;
//    dirent* pdir;
//    string point =".";
//    string two_points = "..";
//    dir = opendir(path.c_str());
//
//    while (readdir(dir)) {
//    	pdir=readdir(dir);
//    	if( point.compare(pdir->d_name) && two_points.compare(pdir->d_name)){
//			files.push_back(pdir->d_name);
//    	}
//    }
//}

vector<string> globVector(const string& pattern){
    glob_t glob_result;
    glob(pattern.c_str(),GLOB_TILDE,NULL,&glob_result);
    vector<string> files;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        files.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return files;
}



template <typename T>
string NumberToString ( T Number )
{
	stringstream ss;
	ss << Number;
	return ss.str();
}

bool error_handling(std::fstream * file){

	bool error = false;
	if(file->is_open()){ cout<<"The file is opened"<<endl;}

	else if(file->good()){ cout<<"The stream doesn't present any problem"<<endl;}

	else if(file->fail()){cout<<"Its fails to open....and this is why its failing"<<endl;
		string s;
		if(file->bad()){cout<<"The stream is in a bad state"<<endl;}
		else{cout<<"It fails for other reasons"<<endl;} error = true;}

	else{cout<<"This is the end of the file"<<endl;}

	return error;
}

void Display_Options(){

	cout<<"---------- I/O OPERATIONS ----------"<<endl;
	cout<<"...PLEASE SELECT A STREAM MODE...:\n"<<endl;
	cout<<"...............READ_FILE............[0]"<<endl;
	cout<<"..............WRITE_FILE...........[1]"<<endl;
	cout<<".............BINARY_MODE...........[2]"<<endl;
	cout<<"...TRUNCATE THE FILE TO LENGTH 0...[3]"<<endl;
	cout<<"...............APPEND..............[4]"<<endl;
	cout<<".......OPEN AND SEEK TO THE END....[5]"<<endl;

}

void Implement_Options(std::fstream & file, int option, const char * Train_Path ){

	switch(option){
			case 0:file.open(Train_Path,ios_base::in);break;
			case 1:file.open(Train_Path, ios_base::out);break;
			case 2:file.open(Train_Path, ios_base::binary);break;
			case 3:file.open(Train_Path, ios_base::trunc);break;
			case 4:file.open(Train_Path, ios_base::app);break;
			case 5:file.open(Train_Path, ios_base::ate);break;
			default:return;
	}
}


void Clear_File(std::fstream file){

		while(error_handling(&file)){
			file.clear();
			cout<<"FILE CLEARED. WE ARE READY TO GO"<<endl;
		}
}

void Input_Output_Streams(SIFT_SURF_HOG &Logo_Detection, const char * Train_Path , const char * path_to_write, const char * name_class){
	string LogoPath = "";
	string name = "";
    string Path_to_Logo = "/home/xisco89/Escritorio/CVISION/PROJECTS/LOGO_DETECTION/TRAIN_LOGOS/FlickrLogos-v2/";
	string previous_name = "";

	//OUTPUT VARIABLES
	int row=0,col=0;
	int it=0;
	//SELECT THE STEAM MODE VARIABLES
	//ERROR HANDLING VARIABLES
	bool initialized = false;
	int option=0;
	//STREAM HANDLING VARIABLES
	std::fstream file;
	std::fstream file_name(name_class, ios_base::in);
	Display_Options();
	cin >> option;
	waitKey();
	Implement_Options(file, option, Train_Path);

	std::string line_name;
	std::string line_path; //LINE TO RECEIVE THE STREM INPUT/OUTPUT

	if(file.is_open()){
		while(getline(file,line_path)) {  // read each line from the file
			std::istringstream stream(line_path);
			char x;
			getline(file_name, line_name);
			std:istringstream stream_name(line_name);
			char x_name;
			while (stream >> x) {  // keep trying to read ints until there are no more
				LogoPath += x;
			}
			while( stream_name >> x_name){
				if(x_name == ','){break;}
				name += x_name;
			}
			if(!initialized){
				previous_name = name;
				initialized = true;
			}
			if(name != previous_name){it=0;initialized = false;}
			else{it++;}

            string write_train = &path_to_write[0];
            write_train += name + '/' + NumberToString(it) + ".txt";
            cout<<"WRITE_TRAIN:"<<write_train<<endl;
            string logo_path = Path_to_Logo + LogoPath;
			cout<<"LOGO_PATH:"<<logo_path<<endl;

			Logo_Detection.Compute_CPU_Sift_Descriptors(logo_path);
			cout<<"SIFT DESCRIPTORS COMPUTED"<<endl;
			Logo_Detection.Compute_CPU_Surf_Descriptors(logo_path);
            cout<<"SURF DESCRIPTORS COMPUTED"<<endl;
            Logo_Detection.Compute_CPU_Kaze_Descriptors(logo_path);
            cout<<"KAZE DESCRIPTORS COMPUTED"<<endl;
            Logo_Detection.Compute_CPU_Brisk_Descriptors(logo_path);
            cout<<"BRISK DESCRIPTORS COMPUTED"<<endl;
            Logo_Detection.Concatenate_Descriptors();
            cout<<"ALL DESCRIPTORS CONCATENATED"<<endl;
			Logo_Detection.write_Descriptors_to_File(&write_train[0]);
            cout<<"DESCRIPTORS HAVE BEEN WRITTEN IN "<<&write_train[0]<<endl;
            LogoPath = "";
			name = "";
		}
	}
	else{
	    perror("error while opening file");
	}
	file.close();
}


void Train_System( SIFT_SURF_HOG & Logo_Detection,const char * Train_Path, const char * path_to_write, const char * name_class){
	Input_Output_Streams(Logo_Detection,Train_Path,path_to_write, name_class);
}


void Test_System(SIFT_SURF_HOG & Logo_Detection, const char * Test_Folder, const char * Descriptors_Folder){

	std::fstream file;file.open(Test_Folder,ios_base::in);std::string line;

    string Path_Test_Image = &Test_Folder[0];string test_image;string Path;
    string write_distances = "/home/xisco89/CVISION/PROJECTS/LOGO_DETECTION/TRAIN_LOGOS/TEST_RESULTS/DISTANCES";
    string test_logo = "/home/xisco89/Escritorio/CVISION/PROJECTS/LOGO_DETECTION/TRAIN_LOGOS/FlickrLogos-v2/";
	string distance_path;
	int it=0;

	if(file.is_open()){
		while(std::getline(file,line)){
			cout<<"\nNEW TEST IMAGE"<<endl;
			it++;
			distance_path = write_distances + NumberToString(it) + ".txt";
			std::stringstream stream(line);
			char x;
			string test_image;
			while(stream >> x){
				test_image += x;
			}
			string logo_test = test_logo + test_image;
			cout<<"LOGO_TEST_PATH"<<logo_test<<endl;
			Logo_Detection.Compute_CPU_Sift_Descriptors(logo_test);
			Logo_Detection.Compute_CPU_Surf_Descriptors(logo_test);
            Logo_Detection.Compute_CPU_Kaze_Descriptors(logo_test);
            cout<<"KAZE DESCRIPTORS COMPUTED"<<endl;
            Logo_Detection.Compute_CPU_Brisk_Descriptors(logo_test);
            cout<<"BRISK DESCRIPTORS COMPUTED"<<endl;
			Logo_Detection.Concatenate_Descriptors();
            cout<<"DESCRIPTORS COMPUTED AND CONCATENATED"<<endl;
			//COMPARE TO OTHERS AND COMPUTE SIMILARITY SCORE
			vector<string> paths;
			open(Descriptors_Folder,paths);
			cout<<"PATHS SIZE:"<<paths.size()<<endl;
			for(int i=0; i<paths.size();i++){
				string path_to_descriptor = Descriptors_Folder + paths[i];
				cout<<"PATH_TO_DESCRIPTOR"<<path_to_descriptor<<endl;
                Logo_Detection.read_Descriptors_from_File(&path_to_descriptor[0]);
                Logo_Detection.Compare_Descriptors(&path_to_descriptor[0]);
            }
            cout<<"ALL DESCRIPTORS READ"<<endl;
        }
	}
}


int main(){

	char Path_Img_Training[200]; 	//PATH FROM WHERE TO EXTRACT SUBPATH TO TRAIN IMAGES FOR CLASSES
	char path_to_descriptors[200]; 	 	//PATH TO WRITE THE DESCRIPTORS FOR EACH CLASS
	char Train_Descriptors[200]; 	//PATH TO SAVE THE FILES WITH DESCRIPTORS FOR EACH CLASS
	char name_class[200];
	char Path_Img_Testing[200];
	char extract_Images[200];
	SIFT_SURF_HOG Logo_Detection;

    sprintf(Path_Img_Training,"/home/xisco89/Escritorio/CVISION/PROJECTS/LOGO_DETECTION/TRAIN_LOGOS/FlickrLogos-v2/trainvalset.relpaths.txt");
    sprintf(path_to_descriptors,"/home/xisco89/Escritorio/CVISION/PROJECTS/LOGO_DETECTION/TRAIN_LOGOS/TRAINED_CLASSES/");
    sprintf(Train_Descriptors, "/home/xisco89/Escritorio/CVISION/PROJECTS/LOGO_DETECTION/TRAIN_LOGOS/FlickrLogos-v2/");
    sprintf(name_class,"/home/xisco89/Escritorio/CVISION/PROJECTS/LOGO_DETECTION/TRAIN_LOGOS/FlickrLogos-v2/trainset.txt");
    sprintf(Path_Img_Testing, "/home/xisco89/Escritorio/CVISION/PROJECTS/LOGO_DETECTION/TRAIN_LOGOS/FlickrLogos-v2/trainset.relpaths.txt");

	cout<<"WHAT DO YOU WANT TO DO : TRAIN[0] OR TEST[1]"<<endl;
	int option=0;
	cin >> option;
	waitKey();
	switch(option){
		case 0:Train_System(Logo_Detection,Path_Img_Training,path_to_descriptors, name_class);break;
        case 1:Test_System(Logo_Detection, Path_Img_Testing, path_to_descriptors);break;
	}
}

