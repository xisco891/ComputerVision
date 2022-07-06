#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <iostream>
#include <iomanip>

//#include "calcSiftSurf.h"

#include "stdio.h"
#include "opticalflow.h"
//#include <algorithm>
//#include <array>

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

		char path2[200];
		char filename[100];
		char filemin[100];
		char file[100];
		char file1[100];

		int probes=0;
		int l=0;
		double min_aux;
		vector<double> vect;

				while(l<7){
					opticalFlow opt;
					sprintf(path2,"../PROYECTO/MOTION/TRAINING_DATA/visualwords_walking_%i_file",l);

					for(int probes=0;probes<7;probes++){
						sprintf(filename,"/home/usr/PROYECTO/MOTION/PROBES_DATA/visualwords_walking_%i_file",probes);
						cout<<"probes:"<<probes;
						cout<<""<<endl;
						waitKey(2000);
						opt.compare_visualwords_2(path2,filename);
						//opt.draw_visualwords();
						waitKey();

						vect.push_back(opt.min_sum);



					}
					l++;
					cout<<"l"<<l;
					cout<<""<<endl;
					waitKey(2000);
				}
				opticalFlow opt1;
				sprintf(filemin,"/home/usr/PROYECTO/MOTION/COMPARE_VISUAL/compare_visual_words_file_SILHOUETTE_3_6_12");
				opt1.writeFile_min(filemin,vect);


		}



