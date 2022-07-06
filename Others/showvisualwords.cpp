
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
//#include <cv.h>
//#include <highgui.h>
//#include "calcSiftSurf.h"
#include "opticalflow.h"
#include "stdio.h"
//#include <algorithm>
//#include <array>
#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

Ptr<BackgroundSubtractor> pMOG2;
Mat fgMaskMOG2;


#undef _GLIBCXX_DEBUG
#include <stdio.h>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <vector>

using namespace cv;
using namespace std;

void shiftImages(vector<Mat>& vimg,int sn)
{
	for(int i=0;i<sn;i++)
	{
		vimg[i]=vimg[i+sn].clone();
	}
}
int main(int argc, char* argv[]){

	int imgnum=10;
	vector<Mat> imgs_final(imgnum);
	vector<Mat> img(10);


	int k=0;
	int flag=0;
	int m=0;
	int num_images=0;
	char path[200];
	vector<vector<float> > results;
	char path2[200];
	Mat mask;

	char filecovariance[100];
	char filename[100];
	char file[100];
	char filename2[100];
	char filename_tracks[100];
	char filecov[100];

	int video_train=0;
	int probes=0;
	int training=0;
	int start=5;
	int h=0;
	int o;
	int framenum=10;
	double min_aux;
	int cont=0;

	while(training<8){

			opticalFlow opt;
			cout<<"tracks.size: "<<opt.tracks.size();
			cout<<""<<endl;
			waitKey(500);
			for(int i=start;i<start+3;i++){
					int flag=0;
					int step=0;
					int end=0;
					h=i;

					if (i == 14) {
                                            cont++;
								}

			sprintf(path,"/home/usr/PROYECTO/MOTION/VIDEO/%03d.avi",h);
          					puts(path);
					VideoCapture cap(path);

					if ( !cap.isOpened() )
					{
						cout << "Cannot open the video file" << endl;
						return -1;
					}

					double fps = cap.get(CV_CAP_PROP_FPS);
					cout << "Frame per seconds : " << fps << endl;
					Mat frame;

					int frameNumbers = cap.get(CV_CAP_PROP_FRAME_COUNT);



					while(1)
					{

						for(int j=0;j<framenum-step;j++)
						{
							bool bSuccess = cap.read(frame); // read a new frame from video

							if (!bSuccess || flag>=frameNumbers)
							{

								end++;
								break;
							}



						//	GaussianBlur(frame,frame, Size(3,3), 0,0);
							img[step+j]=frame.clone();
							flag++;
						}
						if(end==1){
							break;
						}

						for (int j=0;j<framenum;j++)
						{

							cvtColor(img[j],imgs_final[j] , CV_BGR2GRAY);
							namedWindow("IMAGEN");

						}

						opt.loadImages(imgs_final);
						opt.denseflow_2_1();
						opt.denseflow_2_2(2);
						step=5;
						shiftImages(img,step);

					}


				if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
						{
							cout << "esc key is pressed by user" << endl;
							break;
							return 0;
						}


			}

			opt.KmeansClust(80);
			opt.remove_clusters_seeds();
			sprintf(filename,"/home/usr/PROYECTO/MOTION/PROBES_DATA/visualwords_walking_%i_file",probes);
			opt.writetoFile(filename);
			cout<<"FINISHED"<<endl;
			waitKey(4000);


			opt.num_points=0;
			start+=10;
			training++;
			probes++;
	}
	cout<<"PROBES FINISHED"<<endl;

}






