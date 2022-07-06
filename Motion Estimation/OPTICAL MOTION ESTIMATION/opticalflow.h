
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/core/types_c.h"
#include "opencv2/core/version.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include  <opencv2/opencv.hpp>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"



#include <sstream>
#include <fstream>
#include <iostream>
#include "stdio.h"
#include <stdio.h>
#include "VISUAL.h"
#include <vector>
#include <math.h>
#include <string>
#include <sstream>
#include <iterator>
#include <cmath>
#include <eigen3/Eigen/Dense>

#include <vector>

using namespace Eigen;
using namespace cv;
using namespace std;

#undef _GLIBCXX_DEBUG



template<typename T>
int minV(vector<T> V)
{
	if (V.size()<1)
		return -1;

	int inm=0;
	T mv=V[0];

	for(int i=0;i<V.size();i++)
	{
		if (V[i]<=mv)
		{
			mv=V[i];
			inm=i;
		}
	}
	return inm;
}

class opticalFlow{
public:
	Mat I1;
	Mat I2;

	vector<Mat> u1;
	vector<Mat> v1;
	vector<Mat> I;

	vector<vector<KeyPoint> >keypoints;


	vector<vector<Point2f> > points;
	vector< unsigned char> status;

	int dense_count;

	Point pt1;
	Point pt2;

	Mat mask;

	Mat fgMaskMOG2;
	Ptr<BackgroundSubtractor> pMOG2;


	int count;


	vector<VectorXd > tracks_aux;
	vector<VectorXd > tracks_aux_2;

	float min;
	MatrixXd cov;
	vector<MatrixXd > covariances;
	vector<MatrixXd > covariances2;
	MatrixXd distances;

	float min_sum;
	vector<float> vect_aux;
	vector<float> vect2_aux;

	int num_maps;
//	float average;
	float sum_final;


//	float min;
//	float max;


	Mat subtract;

	int num_points;
	int num_points2;
	int num_points_total;

//	Point2f  point1;
//	Point2f  point2;
//	vector<Point2f>  point_1[9];
//	vector<Point2f>   point_2[9];


	vector<int> min_pos_vect;
	vector<float> vect_min;
	vector<double > vect;

	vector<vector<int> > clusters;
	vector<VectorXd > seeds;
	vector<VectorXd > tracks;

	//-------------------VENTANA ARTIFICAL---------------------------------------
	vector<Point > skel;
	vector<vector<int> > clusters_skel;
	vector<Point> seeds_skel;
	vector<VectorXd> final;
	Point2f desplaz;
	int flows=0;

	//--------------------------------------------------------------------------------
	Mat depth;


//	vector<Point2f> corners_first_frame;
//	vector<Point2f>  corners_second_frame[9];




	opticalFlow(vector<Mat>& I_i);
	opticalFlow(){num_points=0;num_points2=0;};
	opticalFlow(vector<vector<Point2f> > &points);


	void sparseOpticalflow();
	void drawFlow(Mat& flow, int step);
	void calcSift();
	void findCorners();
	void denseflow(int resolution);

	//-------------VENTANA ARTIFICIAL----------------------------------

	void skeleton_box();
	void skeleton_tracking(int );
	void skeleton_flow(int,int);
	void KmeansClust_skeleton(int num_clusters);
	void draw_skeleton();
	void points_flow(int);
	//---------------DEPTH---------------------------------
	void depth_boxing(Mat depth);
	void depth_tracking(int res);

	//-----------------------------------------------------------------

	void writeTracks(char * filename);
	void loadImages(vector<Mat>& I_i);

	void denseflow_2_1();
	void colortrack_flow_2_1();

	void denseflow_2_2(int res);
	void colortrack_flow_2_2(int res);
	void KmeansClust(int num_clusters);
	void remove_clusters_seeds();
	void draw_visualwords();
	void writetoFile(char * filename);

	void covariance();
	void read_covariances();
	void writetoFile_covariance(char * filename);

	void readFile(char * filename);
	void distances_covariances(char * filename, char * filename2);
	void compare_visualwords_2(char * filename,char * filename2 );
	void writeFile_min(char * filemin, vector<double> result);


//	void read_visualwords (char * filename);
//	void read_visualwords_2 (char * filename);

protected:
	bool findIndex(vector<int> seedinx,int inx);

};

void opticalFlow::loadImages(vector<Mat>& I_i){
	I.resize(I_i.size());
	points.resize(I_i.size());
	for(int i=0;i<I_i.size();i++)
	{
		I[i]=I_i[i].clone();
//		imshow("IMAGENS",I[i]);
//		waitKey();
	}
	u1.resize(I_i.size()-1);
	v1.resize(I_i.size()-1);

}

opticalFlow::opticalFlow(vector<Mat>& I_i){
	I.resize(I_i.size());
	points.resize(I_i.size());
	for(int i=0;i<I_i.size();i++)
	{
		I[i]=I_i[i].clone();
	}
	u1.resize(I_i.size()-1);
	v1.resize(I_i.size()-1);
	num_points=0;
	clusters.clear();


}



void opticalFlow::findCorners()
{

	int maxCorners=500;
	double qualityLevel=0.01;
	double minDistance=10;


	for(int i=0; i<I.size()-1;i++){
	goodFeaturesToTrack(I[i], points[i],  maxCorners, 0.01, 10, Mat(),  3, 0, 0.04);
	goodFeaturesToTrack(I[i+1], points[i+1],  maxCorners, 0.01, 10, Mat(),  3, 0, 0.04);

	}
}


void opticalFlow::sparseOpticalflow()
{


	char file4[100];

	Mat err;

	Size winSize=Size(21,21);
	int maxLevel=3;

	CvTermCriteria criteria= cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3 );

	int flags=0;
	double minEigThreshold=1e-4;


	int num_points=0;

	cout<<"2.1"<<endl;

	for(int i=0; i<I.size()-1;i++){
		cout<<"2.12"<<endl;

		calcOpticalFlowPyrLK(I[0],I[i],points[i] ,points[i+1], status, err, winSize,maxLevel,criteria, flags, minEigThreshold);

		for(int j=0;j<status.size();j++){
			if (status[j]==0){
				points[i+1][j]=Point2f(-10,-10);
			}

		}
		cout<<"2.1.3"<<endl;

	}

	cout<<"2.2"<<endl;


	for(int i=0;i<points.size()-1;i++){
		for(int l=0;l<points[i+1].size();l++)
		{
			if(points[i+1][l].x<0)
				continue;
			line(I[I.size()-1],points[i][l],points[i+1][l],Scalar(0,0,255),2);
		}
	}
	cout<<"3"<<endl;

	namedWindow("OPTICAL FLOW");
	imshow("OPTICAL FLOW",I[I.size()-1]);

	waitKey(1000);

	sprintf(file4,"../FLOW/FOTO%03d",count);
	imwrite(file4,I[I.size()-1]);

	cout<<"4"<<endl;
	count++;

}



bool opticalFlow::findIndex(vector<int> seedinx,int inx)
{
	for (int i=0;i<seedinx.size();i++)
		if (seedinx[i]==inx)
			return true;

	return false;
}


void opticalFlow::KmeansClust(int num_clusters)
{

	cout<<"0"<<endl;
			cout<<""<<endl;

	cout<<"TRACKS_SIZE:"<<tracks.size();
	cout<<""<<endl;
	cout<<"NUM_POINTS:"<<num_points;
	cout<<""<<endl;



	tracks.resize(num_points);

//	cout<<"0.1"<<endl;
//			cout<<""<<endl;

	int elem=0;

	clusters.resize(num_clusters);
	seeds.resize(num_clusters);

//	cout<<"0.2"<<endl;
//	cout<<""<<endl;
//	waitKey(20);


	int size=0;
	vector<int> seedinx;
	int j=0;

//	cout<<"1"<<endl;
//	cout<<""<<endl;


	while (j<num_clusters)
	{
		int inx=rand()%(num_points-1);
		if (!findIndex(seedinx,inx)){
			seeds[j]=tracks[inx];
		}
		j++;

	}
	//waitKey(10000);


//	cout<<"2"<<endl;
//			cout<<""<<endl;


	for (int it=0;it<20;it++)
	{

		waitKey(1000);
		for (int i=0;i<tracks.size();i++)
		{
			vector<double> dists(seeds.size());
			for (int j=0;j<seeds.size();j++)
			{
				VectorXd diff=tracks[i]-seeds[j];
				dists[j]=(diff.transpose()*diff)(0);

			}

			int inm=minV(dists);


			clusters[inm].push_back(i);
		}

//		cout<<"3"<<endl;
//		cout<<""<<endl;

		for (int i=0;i<clusters.size();i++)
		{
			VectorXd mean;
			mean=seeds[i]*0;


			if (clusters[i].size()==0 && it==19)
			{

				cout<<"clusters with size 0:"<<i;
				cout<<""<<endl;
				waitKey(20);
				vect.resize(elem+1);
				vect[elem]=i;
				elem++;
			}

			for (int j=0;j<clusters[i].size();j++)
			{

				mean+=tracks[clusters[i][j]];

			}

			mean/=clusters[i].size();
			seeds[i]=mean;


		}

	}

}






void opticalFlow::remove_clusters_seeds(){
	int elemento=0;

	for(int i=0;i<vect.size();i++)
	{
		elemento=vect[i];
		cout<<"SEEDS SIZE BEFORE RESIZING"<<seeds.size();
		cout<<"\nCluster with size 0: "<<elemento;

		cout<<"\n"<<endl;
		for(int j=elemento;j<seeds.size()-1;j++){
			seeds[j]=seeds[j+1];
			clusters[j]=clusters[j+1];
		}
		for(int m=0;m<vect.size();m++){
			vect[m]=vect[m]-1;
		}

		clusters.resize(clusters.size()-1);
		cout<<"NUMBER CLUSTERS AFTER"<<clusters.size();
		cout<<"\n"<<endl;


		seeds.resize(seeds.size()-1);
		cout<<"SEEDS SIZE AFTER"<<seeds.size();
		cout<<"\n"<<endl;
	}


}

void opticalFlow::writetoFile(char * filename)
{
	FILE* pfile = fopen(filename, "w");

	for (int i=0;i<seeds.size();i++)
	{
		for(int j=0;j<seeds[i].size();j++){
			fprintf(pfile,"%f ",seeds[i](j));

		}

		fprintf(pfile,"\n");
	}

	fclose(pfile);


}




void opticalFlow::writeFile_min(char * filemin, vector<double> result)
{

	FILE* pfile = fopen(filemin, "w");

	for(int h=0;h<result.size();h++) {
		cout<<"result["<<h;
		cout<<"]="<<result[h];
		cout<<""<<endl;
		waitKey(1000);
		fprintf(pfile,"%f ",result[h]);

		if((h%6)== 0){
			fprintf(pfile,"\n\n");
		}
	}

	fclose(pfile);
}





//void opticalFlow::writeTracks(char * filename)
//{
//	char c[50]="AVERAGE";
//
//	FILE* pfile = fopen(filename, "w");
//	float size=vect2_aux.size();
//	float aver=sum_final/size;
//	for(int h=0;h<vect2_aux.size();h++) {
//		fprintf(pfile,"%f ",vect2_aux[h]);
//	}
//	fprintf(pfile,"\n\n ");
//	fprintf(pfile,"%f ",min);
//	fprintf(pfile,"\n");
//	fprintf(pfile,"%s ",c);
//	fprintf(pfile,"%f ",aver);
//	fprintf(pfile,"\n");
//	fprintf(pfile,"%f ",max);
//
//	fclose(pfile);
//}


void opticalFlow::compare_visualwords_2(char * filename,char * filename2)
{

	Mat img1(860,860, CV_8UC3, Scalar(0,0,0));
	int end1=0;

	int h=img1.cols/2;
	int l=img1.rows/2;

	char file5[100];
	int m=0;
	int res=0;

	min_pos_vect.resize(80);
	double rest(0.0);

	int end2=0;

	double rest_sum=0;
	double min=1000;

	vector<double> resta(80,0);
	vector<double> minimum(80,0);

	vector<double> vect2;
	vector<vector<double> > vect_aux;
	vector<float> vect_aux2;

	int k=0;
	int i=0;
	min_sum=0;
	double min_pos=0;
	vector<Point2f> traj(8);
	vector<Point2f> traj2(8);




	ifstream theStream1(filename);
	if( ! theStream1 ){
		cerr << "file1.in\n";
	}

	while(true)
	{
		ifstream theStream2(filename2);

		if( ! theStream2 ){
			cerr << "file2.in\n";
		}



		string line1;
		getline(theStream1, line1);

		if (line1.empty()){
			cout<<"hallo!"<<endl;
			end1++;
		//	getchar();
			break;
		}

		if(end1>0){
			cout<<"1"<<endl;
			break;
		}

		tracks_aux.resize(80);
		tracks_aux_2.resize(80);


		istringstream myStream1( line1 );
		istream_iterator<float> aux(myStream1), eof;
		vector<float> numbers1(aux, eof);

		tracks_aux_2[i].resize(numbers1.size());
		for(int g=0;g<numbers1.size();g++)
		{
			double dValue3(0.0);
			dValue3 = static_cast<double>(numbers1[g]);
			tracks_aux_2[i](g)=(double)numbers1[g];
		}


		while(k<80)
		{


			string line2;
			getline(theStream2, line2);

			if (line2.empty()){
				cout<<"hallo2!"<<endl;
//				getchar();
				end2++;
				break;
			}


			if(end2>0){
				break;
			}
			cout<<"1"<<endl;

			istringstream myStream2( line2 );
			istream_iterator<float> aux2(myStream2), eof;
			vector<float> numbers2(aux2, eof);
			resta[k]=0;

			tracks_aux[k].resize(numbers2.size());
			for(int j=0;j<numbers1.size();j++)
			{
				double dValue1(0.0);
				double dValue2(0.0);
				dValue1 = static_cast<double>(numbers1[j]);
				dValue2 = static_cast<double>(numbers2[j]);
//				cout<<"PROBES["<<k;
//				cout<<"]["<<j;
//				cout<<"]="<<dValue2;
//				cout<<"TRAINED["<<i;
//				cout<<"]["<<j;
//				cout<<"]="<<dValue1;
//				cout<<""<<endl;

				tracks_aux[k](j)=(double)numbers2[j];

				rest=dValue1-dValue2;
				resta[k]+=abs(rest);

			}

			if(resta[k]<min){
				min=resta[k];
				min_pos=k;

			}
			k++;
		}
		end2=0;

		minimum[i]=min;
		min_pos_vect[i]=min_pos;
		int f=min_pos_vect[i];
		cout<<"f:"<<f;
		//waitKey(1000);
		min_sum+=minimum[i];

		k=0;
		min=1000;

//		traj[m]=Point(h,l/4 +res);
//		traj2[m]=Point(h,l/4 + res);
//		for(int j=0;j<tracks_aux[i].size();j+=2){
//
//					cout<<"2"<<endl;
//					cout<<"i:"<<i;
//					cout<<""<<endl;
//
//					double dValue1(0.0);
//					dValue1 = static_cast<double>(tracks_aux[f](j));
//					double dValue2(0.0);
//					dValue2 = static_cast<double>(tracks_aux[f](j+1));
//
//					double dValue3(0.0);
//					dValue3 = static_cast<double>(tracks_aux_2[i](j));
//					double dValue4(0.0);
//					dValue4 = static_cast<double>(tracks_aux_2[i](j+1));
//
//					dValue1=dValue1*10;
//					dValue2=dValue2*10;
//					dValue3=dValue3*10;
//					dValue4=dValue4*10;
//
//					traj2[m+1]=traj2[m]+Point2f(dValue1,dValue2);
//					line(img1,traj2[m],traj2[m+1],Scalar(0,0,255),1);
//
//
//					traj[m+1]=traj[m]+Point2f(dValue3,dValue4);
//					line(img1,traj[m],traj[m+1],Scalar(0,255,255),1);
//
//
//
//					cout<<"traj2:"<<traj2[m+1];
//					cout<<"\ntraj:"<<traj[m+1];
//					cout<<""<<endl;
//
//					m++;
//
//
//		}

		m=0;
		res+=5;
		i++;
	}
	vect_min.push_back(min_sum);
//	cout<<"vect_min.size:"<<vect_min.size();
//	cout<<""<<endl;
//
//	cout<<"min_sum"<<min_sum;
//	cout<<""<<endl;
//	waitKey(3000);
	sprintf(file5,"../PROYECTO/MOTION/Walk_min%03d.jpg",count);
	imwrite(file5, img1);
	count++;
	cout<<"FINAL"<<endl;
	waitKey(2000);

}


void opticalFlow::depth_boxing(Mat depth){

	Mat dilatada;
	float flow;
	char file4[100];


	Mat subtract1;
	vector<vector<Point> > contours;
	int largest_area=0;
	int largest_contour_index=0;
	int largest_contour_index_2=0;
	int cont_areas=0;
	Rect bounding_rect;
	int MAX_KERNEL_LENGTH=6;

	Mat I0,I0f;
	int thresh=100;
	Mat I1,I1f;
	//depth.convertTo(depth,CV_8UC1);
	imshow("DEPTH",depth);

//	threshold(depth,depth,100,0, CV_THRESH_BINARY);

	//	adaptiveThreshold(subtract1,subtract1,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,7,4);
	//	bitwise_not(subtract1,subtract1);
	threshold(depth,depth,0,255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	imshow("SUBTRACT_TRAS_FILTRADO",depth);
	//waitKey();

	Mat dst(depth.rows,depth.cols,CV_32FC1,Scalar::all(0));
	dst.convertTo(dst, CV_8UC1);

	Mat dst2(depth.rows,depth.cols,CV_32FC1,Scalar::all(0));
	dst2.convertTo(dst2, CV_8UC1);

	Mat contoured=depth.clone();
	vector<Vec4i> hierarchy;

	//	erode(contoured,contoured,Mat());
	//	dilate(contoured,contoured,Mat());
	//	//bitwise_not(contoured,contoured);
	//Canny( contoured, canny_output, thresh, thresh*2, 3 );
	//imshow("CANNY_OUTPUT",canny_output);

	findContours( contoured, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0,0)); // Find the contours in the imag
	vector<vector<Point> >hull( contours.size() );
	vector<vector<Point> >convex( contours.size() );

	for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
	{
		//if(contourArea(contours[i])>=5000){
			double a=contourArea( contours[i],false);  //  Find the area of contour
			convexHull( Mat(contours[i]), hull[i], false );
			//convexityDefects( Mat(contours[i]), hull[i],convex[i]);
			if(a>largest_area){
				largest_contour_index_2=largest_contour_index;
				largest_area=a;
				largest_contour_index=i;//Store the index of largest contour
				bounding_rect=boundingRect(hull[i]); 	   // Find the bounding rectangle for biggest contour
				cont_areas++;

			}
		}
	//}
	Scalar color( 255,255,255);
	//drawContours( dst2, contours,largest_contour_index, color, CV_FILLED, 8, hierarchy );

	drawContours( dst, contours,largest_contour_index, color, 1, 8, hierarchy );
	drawContours( dst, contours,largest_contour_index_2, color, 1, 8, hierarchy );

	//imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst);
	//				 waitKey(500);
	//drawContours( dst2, contours,largest_contour_index_2, color, 1, 8, hierarchy );
	//imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst);

	drawContours( dst2, contours,largest_contour_index, color, CV_FILLED, 8, hierarchy );

	//				 imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst2);
	//				 waitKey(500);
	drawContours( dst2, contours,largest_contour_index_2, color, CV_FILLED, 8, hierarchy );
	//				 imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst2);
	imshow("dst2:",dst2);

	dst2.convertTo(dst2, CV_32FC1);
	imshow("dst1:",dst);
	int h1=dst2.rows;
	int h2=0;
	int a1=dst2.cols;
	int a2=0;
	int wpix=0;


	//waitKey();


	//	cout<<"dst.rows:"<<dst.rows;
	//	cout<<"\ndst.cols:"<<dst.cols;


	for(int k=0;k<dst2.cols;k++){
		for(int l=0;l<dst2.rows;l++){

			if(dst2.at<float>(l,k)>0){

				if(l<h1){
					h1=l;
				}
				if(l>h2){
					h2=l;
				}
				if(k<a1){
					a1=k;
				}
				if(k>a2){
					a2=k;
				}
			}
			//	else{
			//		dst2.at<float>(l,k)=0;
			//	}
		}
	}
	a1-=5;
	h1-=5;
	a2+=5;
	h2+=5;

	Point p1(a1,h1);
	Point p2(a2,h2);


	//imwrite("/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst);

	//waitKey(5000);

	subtract=dst2.clone();
	//
	//
	rectangle(dst2,p1,p2, Scalar( 255, 0 , 0 ),  2, 4 );
	//	circle(dst,p1, 1, Scalar( 80,255,255 ),6,4,0);
	//	circle(dst,p2, 1, Scalar( 80, 255, 255 ),6,4,0);
	//
	//	imwrite("/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst2);
	//	waitKey(500);
	count++;
	pt1=p1;
	pt2=p2;

	imshow("FINAL_IMAGE",dst2);
	waitKey(200);

}

void opticalFlow::depth_tracking(int res){

	Point ptx1, ptx2;
	ptx1=pt1;
	ptx2=pt2;

	int cont=0;
//	cout<<"PT1:"<<ptx1;
//	cout<<"\nPT2:"<<ptx2;
//	cout<<""<<endl;
//	waitKey(200);
	Mat puntos=subtract.clone();
	cvtColor(puntos,puntos, CV_GRAY2BGR);


	for(int k=ptx2.x;k>ptx1.x-1;k-=res)
	{
		for(int l=ptx2.y;l>ptx1.y-1;l-=res)
		{
			if(subtract.at<float>(l,k)>0){
				cout<<"NUM_POINTS:"<<num_points;
				cout<<""<<endl;
				Point punto(k,l);
				skel.push_back(punto);
//				circle(puntos, punto,2, Scalar(),CV_FILLED,8,0);
//				imshow("PUNTOS",puntos);
//				waitKey(200);
//				cout<<"VECTOR SAVED:"<<skel.at(num_points);
//				cout<<""<<endl;
				num_points++;
			}

		}
	}
return;
}



void opticalFlow::skeleton_box(){

	Mat dilatada;
	float flow;
	char file4[100];


	Mat subtract1;
	vector<vector<Point> > contours;
	int largest_area=0;
	int largest_contour_index=0;
	int largest_contour_index_2=0;
	int cont_areas=0;
	Rect bounding_rect;
	int MAX_KERNEL_LENGTH=6;

	Mat I0,I0f;
	int thresh=100;
	Mat I1,I1f;
	Mat avg;
	Mat smoothed_background;
	Mat grey_background;
//	I[1].convertTo(I1f,CV_32FC1);
	//Mat background=imread("../PROYECTO/MOTION/BACKGROUND_CASA.jpg");
	Mat background=imread("../PROYECTO/MOTION/VIDEO/BACKGROUND_SAVED/BACKGROUND.jpg");

	GaussianBlur(background,smoothed_background, Size(3,3), 0,0);
//	for(int j=1;j<MAX_KERNEL_LENGTH;j=j+2){
//		bilateralFilter(background,smoothed_background,j,j*2,j/2);
//	}

	cvtColor(smoothed_background, grey_background, CV_BGR2GRAY);
	//imshow("background",grey_background);
	grey_background.convertTo(grey_background,CV_32FC1);
	I[0].convertTo(I0f,CV_32FC1);
	absdiff(grey_background,I0f,subtract1);
//	absdiff(I1f,I0f,subtract1);
	subtract1.convertTo(subtract1,CV_8UC1);
	imshow("SUBTRACT1",subtract1);

	threshold(subtract1,subtract1,40,255, CV_THRESH_BINARY);

//	adaptiveThreshold(subtract1,subtract1,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,7,4);
//	bitwise_not(subtract1,subtract1);

	//threshold(subtract1,subtract1,0,255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	imshow("SUBTRACT_TRAS_FILTRADO",subtract1);

	Mat dst(subtract1.rows,subtract1.cols,CV_32FC1,Scalar::all(0));

	dst.convertTo(dst, CV_8UC1);
	Mat dst2(subtract1.rows,subtract1.cols,CV_32FC1,Scalar::all(0));
	dst2.convertTo(dst2, CV_8UC1);

	Mat contoured=subtract1.clone();
	vector<Vec4i> hierarchy;

	//	erode(contoured,contoured,Mat());
	//	dilate(contoured,contoured,Mat());
	//	//bitwise_not(contoured,contoured);
	//Canny( contoured, canny_output, thresh, thresh*2, 3 );
	//imshow("CANNY_OUTPUT",canny_output);

	findContours( contoured, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0,0)); // Find the contours in the imag
	vector<vector<Point> >hull( contours.size() );
	vector<vector<Point> >convex( contours.size() );

	for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
	{
		//if(contourArea(contours[i])>=5000){
			double a=contourArea( contours[i],false);  //  Find the area of contour
			convexHull( Mat(contours[i]), hull[i], false );
			//convexityDefects( Mat(contours[i]), hull[i],convex[i]);
			if(a>largest_area){
				largest_contour_index_2=largest_contour_index;
				largest_area=a;
				largest_contour_index=i;//Store the index of largest contour
				bounding_rect=boundingRect(hull[i]); 	   // Find the bounding rectangle for biggest contour
				cont_areas++;

			}
		}
	//}
	Scalar color( 255,255,255);
	//drawContours( dst2, contours,largest_contour_index, color, CV_FILLED, 8, hierarchy );

	drawContours( dst, contours,largest_contour_index, color, 1, 8, hierarchy );
	drawContours( dst, contours,largest_contour_index_2, color, 1, 8, hierarchy );

	//imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst);
	//				 waitKey(500);
	//drawContours( dst2, contours,largest_contour_index_2, color, 1, 8, hierarchy );
	//imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst);

	drawContours( dst2, contours,largest_contour_index, color, CV_FILLED, 8, hierarchy );
	//				 imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst2);
	//				 waitKey(500);
	drawContours( dst2, contours,largest_contour_index_2, color, CV_FILLED, 8, hierarchy );
	//				 imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst2);

	dst2.convertTo(dst2, CV_32FC1);
	//dst2=subtract1.clone();


imshow("DST1:",dst);
imshow("DST2:",dst2);


	int h1=dst2.rows;
	int h2=0;
	int a1=dst2.cols;
	int a2=0;
	int wpix=0;
	//
	//	cout<<"dst.rows:"<<dst.rows;
	//	cout<<"\ndst.cols:"<<dst.cols;



	for(int k=0;k<dst2.cols;k++){
		for(int l=0;l<dst2.rows;l++){

			if(dst2.at<float>(l,k)>0){

				if(l<h1){
					h1=l;
				}
				if(l>h2){
					h2=l;
				}
				if(k<a1){
					a1=k;
				}
				if(k>a2){
					a2=k;
				}
			}
			//	else{
			//		dst2.at<float>(l,k)=0;
			//	}
		}
	}
	a1-=5;
	h1-=5;
	a2+=5;
	h2+=5;

	Point p1(a1,h1);
	Point p2(a2,h2);


	//imwrite("/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst);

	//waitKey(5000);

	subtract=dst2.clone();
	//
	//
	rectangle(dst2,p1,p2, Scalar( 255, 0 , 0 ),  2, 4 );
	//	circle(dst,p1, 1, Scalar( 80,255,255 ),6,4,0);
	//	circle(dst,p2, 1, Scalar( 80, 255, 255 ),6,4,0);
	//
	//	imwrite("/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst2);
	//	waitKey(500);
	count++;
	pt1=p1;
	pt2=p2;

	imshow("FINAL_IMAGE",dst2);
	waitKey(200);

}


void opticalFlow::skeleton_tracking(int res){

	Point ptx1, ptx2;
	ptx1=pt1;
	ptx2=pt2;

	int cont=0;
	num_points=0;
	skel.resize(0);
//	cout<<"PT1:"<<ptx1;
//	cout<<"\nPT2:"<<ptx2;
//	cout<<""<<endl;
//	waitKey(200);
	Mat puntos=subtract.clone();
	puntos.convertTo(puntos,CV_32FC1);
	cvtColor(puntos,puntos, CV_GRAY2BGR);

	for(int k=ptx2.x;k>ptx1.x-1;k-=res)
	{
		for(int l=ptx2.y;l>ptx1.y-1;l-=res)
		{
			if(subtract.at<float>(l,k)>0){
//				cout<<"NUM_POINTS:"<<num_points;
//				cout<<""<<endl;
				Point punto(k,l);
				skel.push_back(punto);
				num_points++;
//				circle(puntos, punto,2, Scalar(204,0,0),CV_FILLED,8,0);
//				imshow("PUNTOS",puntos);
				//waitKey(20);
//				cout<<"VECTOR SAVED:"<<skel.at(num_points);
//				cout<<""<<endl;
			}

		}
	}
return;
}


void opticalFlow::points_flow(int res){

	Point ptx1, ptx2;
	ptx1=pt1;
	ptx2=pt2;
	float mean_a=0,mean_b=0;
	Mat uv;
	Mat channels[2];
	float sum=0;
	float aux=1000;
	float aux2=0;
	char file3[100];
	int flows=0;

	float a=0;	float b=0;

	for(int i=0;i<I.size()-1;i++){
		calcOpticalFlowFarneback (I[i], I[i+1], uv, 0.5, 6 ,12, 3, 5, 1.2, 0);
		split(uv,channels);
		u1[i]=channels[0].clone();
		v1[i]=channels[1].clone();
	}
	Mat Ic;
	vector<Point2f> point;
	int m=0;
	cvtColor(I[I.size()-1], Ic,CV_GRAY2BGR);
	imshow("IC",Ic);
	//num_maps=I.size()/10;
	vector<Point2f> traj(I.size());
	int k=0,l=0;
	for(int i=0;i<seeds_skel.size();i++){
		k=seeds_skel[i].x;
		l=seeds_skel[i].y;
		traj[m]=Point2f(k,l);
		for(int j=m;j<m+4;j++){
			traj[j+1]=traj[j]+Point2f(u1[j].at<float>(l,k),v1[j].at<float>(l,k));
			line(Ic,traj[j],traj[j+1],Scalar(0,0,255),0.05);
			a+=abs(u1[j].at<float>(traj[j]));
			b+=abs(v1[j].at<float>(traj[j]));
		}
		flows++;
		mean_a+=a/5;
		mean_b+=b/5;
		 namedWindow("FLOW");
		 imshow("FLOW",Ic);

	}
	mean_a=mean_a/flows;
	mean_b=mean_b/flows;
	Point2f mean(mean_a,mean_b);
	desplaz=mean;
	return;
}

void opticalFlow::skeleton_flow(int res, int step){

	Point ptx1, ptx2;
	ptx1=pt1;
	ptx2=pt2;
	float mean_a=0,mean_b=0;
	Mat uv;
	Mat channels[2];
	float sum=0;
	float aux=1000;
	float aux2=0;
	char file3[100];
//	int flows=0;

	float a=0;
	float b=0;

	for(int i=0;i<I.size()-1;i++){

		//calcOpticalFlowFarneback (I[i], I[i+1], uv, 0.5, 2 ,3, 3, 5, 1.2, 0);
		calcOpticalFlowFarneback (I[i], I[i+1], uv, 0.5, 6 ,24, 3, 5, 1.2, 0);
		//calcOpticalFlowFarneback (I[i], I[i+1], uv, 0.5, 2 ,3, 3, 5, 1.1, 0);
		split(uv,channels);
		u1[i]=channels[0].clone();
		v1[i]=channels[1].clone();

//		prev – first 8-bit single-channel input image.
//		next – second input image of the same size and the same type as prev.
//		flow – computed flow image that has the same size as prev and type CV_32FC2.
//		pyr_scale – parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
//		levels – number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
//		winsize – averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
//		iterations – number of iterations the algorithm does at each pyramid level.
//		poly_n – size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
//		poly_sigma – standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
//		flags –

//		IplImage lpl_u=u1[i];
//		IplImage lpl_v=v1[i];
//
//		OF_visualization of_visualization;
//		of_visualization.drawColorField(&lpl_u, &lpl_u);

}

	Mat Ic;
	vector<Point2f> point;
	int m=0;
	cvtColor(I[I.size()-1], Ic,CV_GRAY2BGR);
	imshow("IC",Ic);
	//num_maps=I.size()/10;
	vector<Point2f> traj(I.size());
	for(int k=ptx2.x;k>ptx1.x-1;k-=res)
	{
		for(int l=ptx2.y;l>ptx1.y-1;l-=res)
		{
			if(subtract.at<float>(l,k)>0){
				a=0;
				b=0;
				traj[m]=Point2f(k,l);
				for(int j=m;j<m+(step-1);j++){
					traj[j+1]=traj[j]+Point2f(u1[j].at<float>(l,k),v1[j].at<float>(l,k));
					line(Ic,traj[j],traj[j+1],Scalar(0,0,255),0.05);
					a+=abs(u1[j].at<float>(traj[j]));
					b+=abs(v1[j].at<float>(traj[j]));

				}
				flows++;
				mean_a+=a/5;
				mean_b+=b/5;

			}


		}


		 namedWindow("FLOW");
		 imshow("FLOW",Ic);
	//	 waitKey(70);
	}
	cout<<"\nPREVIOUS_mean_a:"<<mean_a;
	cout<<"\nPREVIOUS_mean_b:"<<mean_b;
	mean_a=mean_a/flows;
	mean_b=mean_b/flows;
	cout<<"\nmean_a:"<<mean_a;
	cout<<"\nmean_b:"<<mean_b;
	waitKey(1000);
	Point2f mean(mean_a,mean_b);
	desplaz+=mean;
	cout<<"desplaz:"<<desplaz<<endl;


	return;
}

void opticalFlow::KmeansClust_skeleton(int num_clusters)
{
	cout<<"SKEL_SIZE:"<<skel.size();
	cout<<""<<endl;
	cout<<"NUM_POINTS:"<<num_points;
	cout<<""<<endl;


	skel.resize(num_points);
	int elem=0;

	clusters_skel.resize(num_clusters);
	seeds_skel.resize(num_clusters);

	int size=0;
	vector<int> seedinx;
	int j=0;


	while (j<num_clusters)
	{
		int inx=rand()%(num_points-1);
		//int inx2=rand()%(num_points-1);

		if (!findIndex(seedinx,inx)){
			seeds_skel[j]=skel[inx];
		}
		j++;

	}
	for (int it=0;it<20;it++)
	{

		//waitKey(300);
		for (int i=0;i<skel.size();i++)
		{
			vector<float> dists(seeds_skel.size());
			for (int j=0;j<seeds_skel.size();j++)
			{
				Point diff=skel[i]-seeds_skel[j];
				dists[j]=sqrt(diff.x*diff.x + diff.y*diff.y);

			}
			int inm=minV(dists);
			clusters_skel[inm].push_back(i);
		}

		for (int i=0;i<clusters_skel.size();i++)
		{
			Point mean;
			mean=seeds_skel[i]*0;

			for (int j=0;j<clusters_skel[i].size();j++)
			{
				mean+=skel[clusters_skel[i][j]];
			}
			int num=clusters_skel[i].size();
			mean/=num;
			//cout<<"mean:"<<mean<<endl;
			seeds_skel[i]=mean;

		}
	}
}

void opticalFlow::draw_skeleton(){
	Point center;
	Mat skeleton=subtract.clone();
	cvtColor(skeleton,skeleton, CV_GRAY2BGR);
	cout<<"DRAWING SKELETON POINTS"<<endl;
	for(int i=0;i<seeds_skel.size();i++){
		Scalar color = Scalar( 204,0,0 );
		center.x=seeds_skel[i].x;
		center.y=seeds_skel[i].y;
		cout<<"CENTER:"<<center<<endl;
		circle(skeleton, center,4, color,CV_FILLED,8,0);
//		cout<<"point_["<<i<<"]:"<<i<<endl;
		imshow("skeleton",skeleton);
	//	waitKey(200);
		}

}


void opticalFlow::denseflow_2_1(){

	Mat dilatada;
	float flow;
	char file4[100];


	Mat subtract1;
	vector<vector<Point> > contours;
	int largest_area=0;
	int largest_contour_index=0;
	int largest_contour_index_2=0;
	int cont_areas=0;
	Rect bounding_rect;
	int MAX_KERNEL_LENGTH=6;

	Mat I0,I0f;
	int thresh=100;
	Mat I1,I1f;
	Mat avg;
	Mat smoothed_background;
	Mat grey_background;
//	I[1].convertTo(I1f,CV_32FC1);
	Mat background=imread("../PROYECTO/MOTION/BACKGROUND_CASA.jpg");
	//Mat background=imread("../PROYECTO/MOTION/VIDEO/BACKGROUND_SAVED/BACKGROUND.jpg");

	//GaussianBlur(background,smoothed_background, Size(3,3), 0,0);
	for(int j=1;j<MAX_KERNEL_LENGTH;j=j+2){
		bilateralFilter(background,smoothed_background,j,j*2,j/2);
	}

	cvtColor(smoothed_background, grey_background, CV_BGR2GRAY);
	//imshow("background",grey_background);
	grey_background.convertTo(grey_background,CV_32FC1);
	I[0].convertTo(I0f,CV_32FC1);
	absdiff(grey_background,I0f,subtract1);
//	absdiff(I1f,I0f,subtract1);
	subtract1.convertTo(subtract1,CV_8UC1);
	imshow("SUBTRACT1",subtract1);

	threshold(subtract1,subtract1,40,255, CV_THRESH_BINARY);

//	adaptiveThreshold(subtract1,subtract1,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,7,4);
//	bitwise_not(subtract1,subtract1);

	//threshold(subtract1,subtract1,0,255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	imshow("SUBTRACT_TRAS_FILTRADO",subtract1);

	Mat dst(subtract1.rows,subtract1.cols,CV_32FC1,Scalar::all(0));

	dst.convertTo(dst, CV_8UC1);
	Mat dst2(subtract1.rows,subtract1.cols,CV_32FC1,Scalar::all(0));
	dst2.convertTo(dst2, CV_8UC1);

	Mat contoured=subtract1.clone();
	vector<Vec4i> hierarchy;

	//	erode(contoured,contoured,Mat());
	//	dilate(contoured,contoured,Mat());
	//	//bitwise_not(contoured,contoured);
	//Canny( contoured, canny_output, thresh, thresh*2, 3 );
	//imshow("CANNY_OUTPUT",canny_output);

	findContours( contoured, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0,0)); // Find the contours in the imag
	vector<vector<Point> >hull( contours.size() );
	vector<vector<Point> >convex( contours.size() );

	for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
	{
		//if(contourArea(contours[i])>=5000){
			double a=contourArea( contours[i],false);  //  Find the area of contour
			convexHull( Mat(contours[i]), hull[i], false );
			//convexityDefects( Mat(contours[i]), hull[i],convex[i]);
			if(a>largest_area){
				largest_contour_index_2=largest_contour_index;
				largest_area=a;
				largest_contour_index=i;//Store the index of largest contour
				bounding_rect=boundingRect(hull[i]); 	   // Find the bounding rectangle for biggest contour
				cont_areas++;

			}
		}
	//}
	Scalar color( 255,255,255);
	//drawContours( dst2, contours,largest_contour_index, color, CV_FILLED, 8, hierarchy );

	drawContours( dst, contours,largest_contour_index, color, 1, 8, hierarchy );
	drawContours( dst, contours,largest_contour_index_2, color, 1, 8, hierarchy );

	//imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst);
	//				 waitKey(500);
	//drawContours( dst2, contours,largest_contour_index_2, color, 1, 8, hierarchy );
	//imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst);

	drawContours( dst2, contours,largest_contour_index, color, CV_FILLED, 8, hierarchy );
	//				 imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst2);
	//				 waitKey(500);
	drawContours( dst2, contours,largest_contour_index_2, color, CV_FILLED, 8, hierarchy );
	//				 imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst2);

	dst2.convertTo(dst2, CV_32FC1);
	//dst2=subtract1.clone();


imshow("DST1:",dst);
imshow("DST2:",dst2);


	int h1=dst2.rows;
	int h2=0;
	int a1=dst2.cols;
	int a2=0;
	int wpix=0;
	//
	//	cout<<"dst.rows:"<<dst.rows;
	//	cout<<"\ndst.cols:"<<dst.cols;



	for(int k=0;k<dst2.cols;k++){
		for(int l=0;l<dst2.rows;l++){

			if(dst2.at<float>(l,k)>0){

				if(l<h1){
					h1=l;
				}
				if(l>h2){
					h2=l;
				}
				if(k<a1){
					a1=k;
				}
				if(k>a2){
					a2=k;
				}
			}
			//	else{
			//		dst2.at<float>(l,k)=0;
			//	}
		}
	}
	a1-=5;
	h1-=5;
	a2+=5;
	h2+=5;

	Point p1(a1,h1);
	Point p2(a2,h2);


	//imwrite("/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst);

	//waitKey(5000);

	subtract=dst2.clone();
	//
	//
	rectangle(dst2,p1,p2, Scalar( 255, 0 , 0 ),  2, 4 );
	//	circle(dst,p1, 1, Scalar( 80,255,255 ),6,4,0);
	//	circle(dst,p2, 1, Scalar( 80, 255, 255 ),6,4,0);
	//
	//	imwrite("/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", dst2);
	//	waitKey(500);
	count++;
	pt1=p1;
	pt2=p2;

	cout<<"PT1:"<<pt1;
	cout<<"\nPT2:"<<pt2;
	cout<<""<<endl;
	waitKey(2000);

	imshow("FINAL_IMAGE",dst2);
	waitKey(200);

}

//	absdiff(I1f,I0f,subtract1);
//	Mat dst(subtract1.rows,subtract1.cols,CV_32FC1,Scalar::all(0));
//
//	imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", subtract1);
//	waitKey(1000);




//	cvtColor(avg, background, CV_BGR2GRAY);
//
//	avg.convertTo(avg,CV_32FC1);
	//			Mat background=imread("../PROYECTO/MOTION/VIDEO/BACKGROUND_SAVED/BACKGROUND_2.jpg");
	//			cvtColor(background, background, CV_BGR2GRAY);
	////			waitKey(1000);
	////			imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", background);
	//			absdiff(background,I[0],subtract1);
	//			imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", subtract1);
	//			waitKey(1000);
	//	threshold(subtract1,subtract1,0,255, CV_THRESH_BINARY | CV_THRESH_OTSU);
//			imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", subtract1);
//			waitKey(500);


	 ////	 if(cont_areas >> 1 ){
	////	 drawContours( dst, contours,largest_contour_index_2, color, CV_FILLED, 8, hierarchy );
	////	 }

//GaussianBlur(background,background, Size(3,3), 0,0);


//	imwrite( "/home/usr/PROYECTO/MOTION/BACKGROUND_IMAGE.jpg", background);
//	imwrite( "/home/usr/PROYECTO/MOTION/I0_IMAGE.jpg",I0f);



//	imshow("I0f",I0f);
//	imshow("I1f",I1f);

//absdiff(I1f,I0f,subtract1);





//threshold(subtract1,subtract1, 4, 255, CV_THRESH_BINARY);






//	imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", subtract1);



//FILTRADO ADAPTATIVO
//subtract1.convertTo(subtract1,CV_8UC1);
//adaptiveThreshold(subtract1,subtract1,255,CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY,7,4);
//bitwise_not(subtract1,subtract1);

//OTSU--------------------
//	threshold(subtract1,subtract1,0,255, CV_THRESH_BINARY | CV_THRESH_OTSU);
//	imwrite( "/home/usr/PROYECTO/MOTION/BINARIZED_IMAGE.jpg", subtract1);
//	imshow("SUBTRACT1",subtract1);



//	dst.convertTo(dst,CV_32FC1);

//dst2=subtract1.clone();



//
//void opticalFlow::colortrack_flow_2_1(){
//
//	Mat dilatada;
//	uchar flow;
//	char file4[100];
//	Mat subtract1;
//
////	int erosion_size=6;
////	Mat element = getStructuringElement(cv::MORPH_CROSS,
////	              Size(2 * erosion_size + 1, 2 * erosion_size + 1),
////	              Point(erosion_size, erosion_size) );
//
//	Mat I0f,I9f;
//
//	I[0].convertTo(I0f,CV_32FC1);
//	I[9].convertTo(I9f,CV_32FC1);
//
//    //imwrite( "../IMAGE1.jpg", I0f);
//    //imwrite( "../IMAGE2.jpg", I1f);
//
//    //imshow("IMAGE_0", I0f);
//    //imshow("IMAGE_1", I1f);
//
//
//	absdiff(I9f,I0f,subtract1);
//	threshold(subtract1,subtract1,400,500,4);
//	imwrite( "/home/usr/PROYECTO/MOTION/IMAGEN_FILTRADA.jpg", subtract1);
//
//
//	int h1=subtract1.rows;
//	int h2=0;
//	int a1=subtract1.cols;
//	int a2=0;
//	int wpix=0;
//
//
//	for(int k=0;k<subtract1.cols;k++){
//		for(int l=0;l<subtract1.rows;l++){
//
//			if(subtract1.at<float>(l,k)>80){
//
//				if(l<h1){
//					h1=l;
//				}
//				if(l>h2){
//					h2=l;
//				}
//				if(k<a1){
//					a1=k;
//				}
//				if(k>a2){
//					a2=k;
//				}
//
//			}
//			else
//				subtract1.at<float>(l,k)=0;
//
//		}
//	}
//
//	a1-=5;
//	h1-=5;
//	a2+=5;
//	h2+=5;
//
//	Point p1(a1,h1);
//	Point p2(a2,h2);
//
//	dilate(subtract1,subtract, Mat(),Point(-1,-1),2,1,1);
//	imwrite( "/home/usr/PROYECTO/MOTION/IMAGEN_DILATADA.jpg", subtract);
//	dilate(subtract,subtract, Mat(),Point(-1,-1),2,1,1);
//	imwrite( "/home/usr/PROYECTO/MOTION/IMAGEN_DILATADA_2.jpg", subtract);
////	dilate(subtract,subtract, Mat(),Point(-1,-1),2,1,1);
////	imwrite( "/home/usr/PROYECTO/MOTION/IMAGEN_DILATADA_3.jpg", subtract);
//
//
//	rectangle(subtract1,p1,p2, Scalar( 255, 0 , 0 ),  2, 4 );
//	imwrite( "/home/usr/PROYECTO/MOTION/rectangled.jpg", subtract1);
//KmeansClust_skeleton
//
//
////	imshow("RECTANGLED IMAGE", subtract/255.0);
////	waitKey(50);
////	sprintf(file4,"../FOTO%03d.jpg",count);
////	imwrite(file4,subtract1);
//
//	count++;
//	pt1=p1;
//	pt2=p2;
//
//
//}






void opticalFlow::denseflow_2_2(int res){

	Point ptx1, ptx2;
	ptx1=pt1;
	ptx2=pt2;
	Mat uv;
	Mat channels[2];
	float sum=0;
	float aux=1000;
	float aux2=0;
	char file3[100];


	for(int i=0;i<I.size()-1;i++){

//		calcOpticalFlowFarneback (I[i], I[i+1], uv, 0.5, 2 ,3, 3, 5, 1.2, 0);
		calcOpticalFlowFarneback (I[i], I[i+1], uv, 0.5, 6 ,12, 3, 5, 1.2, 0);
//      calcOpticalFlowFarneback (I[i], I[i+1], uv, 0.5, 2 ,3, 3, 5, 1.1, 0);


//		prev – first 8-bit single-channel input image.
//		next – second input image of the same size and the same type as prev.
//		flow – computed flow image that has the same size as prev and type CV_32FC2.
//		pyr_scale – parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
//		levels – number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
//		winsize – averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
//		iterations – number of iterations the algorithm does at each pyramid level.
//		poly_n – size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
//		poly_sigma – standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
//		flags –
		split(uv,channels);
		u1[i]=channels[0].clone();
		v1[i]=channels[1].clone();
//		IplImage lpl_u=u1[i];
//		IplImage lpl_v=v1[i];
//
//		OF_visualization of_visualization;
//		of_visualization.drawColorField(&lpl_u, &lpl_u);

}

	Mat Ic;
	vector<Point2f> point;
	int m=0;
	cvtColor(I[I.size()-1], Ic,CV_GRAY2BGR);
	num_maps=I.size()/10;
	vector<Point2f> traj(I.size());



	for(int k=ptx2.x;k>ptx1.x-1;k-=res)
	{
		for(int l=ptx2.y;l>ptx1.y-1;l-=res)
		{


		if(subtract.at<float>(l,k)>0){

			VectorXd tv(2*u1.size());
			tracks.push_back(tv);
			traj[m]=Point2f(k,l);
			//vect2_aux.resize(num_points);

			//circle(Ic, Point(k,l), 1, Scalar( 255, 255 , 255 ),1,4,0);

			for(int j=m;j<m+9;j++)
			{
				traj[j+1]=traj[j]+Point2f(u1[j].at<float>(l,k),v1[j].at<float>(l,k));
				line(Ic,traj[j],traj[j+1],Scalar(0,0,255),0.05);


				float a=u1[j].at<float>(traj[j]);
				float b=v1[j].at<float>(traj[j]);

				sum+=sqrt((a*a)+(b*b));

				tracks[num_points](2*j)=u1[j].at<float>(traj[j]);
				tracks[num_points](2*j+1)=v1[j].at<float>(traj[j]);

//				imshow("IMAGEN_TRACK",Ic);
//				waitKey();
//				cout<<"TRACK_NUM["<<num_points;
//				cout<<"]["<<2*j;
//				cout<<"] y TRACK_NUM["<<num_points;
//				cout<<"]["<<2*j+1;
//				cout<<"] Trazandose y guardandose"<<endl;

			}

			if(sum<1){
				for(int j=0;j<tracks.size()-1;j++){
//					cout<<"\ntracks["<<j;
//					cout<<"]size()="<<tracks[j].size();
					for(int l=0;l<tracks[j].size();l++){
						double f=tracks[j+1][l];
						tracks[j][l]=f;
					}
				}
				num_points=num_points-1;
				tracks.resize(tracks.size()-1);
			}

			sum=0;
			num_points++;

		}

		}
//		 rectangle(Ic,ptx1,ptx2, Scalar( 255, 0 , 0 ),  2, 4 );
		 namedWindow("FLOW");
		 imshow("FLOW",Ic);
//		 imwrite( "/home/usr/PROYECTO/MOTION/SUBTRACTED_IMAGE.jpg", Ic);

		 waitKey(70);
//		out_capture.write(Ic);
	}

//	out_capture.release();
	return;

}


//void opticalFlow::colortrack_flow_2_2(int res){
//	Point ptx1, ptx2;
//	ptx1=pt1;
//	ptx2=pt2;
//	Mat uv;
//	Mat channels[2];
//	float sum=0;
//	float aux=1000;
//	float aux2=0;
//	char file3[100];
//
////VideoWriter out_capture("/home/usr/PROYECTO/MOTION/video.avi", CV_FOURCC('M','J','P','G'), 25, I[0].size());
//
//	for(int i=0;i<I.size()-1;i++){
//
////		calcOpticalFlowFarneback (I[i], I[i+1], uv, 0.5, 2 ,3, 3, 5, 1.2, 0);
//		calcOpticalFlowFarneback (I[i], I[i+1], uv, 0.5, 6 ,12, 3, 5, 1.2, 0);
////      calcOpticalFlowFarneback (I[i], I[i+1], uv, 0.5, 2 ,3, 3, 5, 1.1, 0);
////		Parameters:
////		prev – first 8-bit single-channel input image.
////		next – second input image of the same size and the same type as prev.
////		flow – computed flow image that has the same size as prev and type CV_32FC2.
////		pyr_scale – parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
////		levels – number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
////		winsize – averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
////		iterations – number of iterations the algorithm does at each pyramid level.
////		poly_n – size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
////		poly_sigma – standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
////		flags –
//		split(uv,channels);
//		u1[i]=channels[0].clone();
//		v1[i]=channels[1].clone();
////		IplImage lpl_u=u1[i];
////		IplImage lpl_v=v1[i];
////
////		OF_visualization of_visualization;
////		of_visualization.drawColorField(&lpl_u, &lpl_u);
//
//}
//
//	Mat Ic;
//	vector<Point2f> point;
//	int m=0;
//	cvtColor(I[I.size()-1], Ic,CV_GRAY2BGR);
//	num_maps=I.size()/10;
//	vector<Point2f> traj(I.size());
//
//	for(int k=ptx1.x;k<ptx2.x+1;k+=res)
//	{
//		for(int l=ptx1.y;l<ptx2.y+1;l+=res)
//		{
//
//		if(subtract.at<float>(l,k)>100){
//
//			VectorXd tv(2*u1.size());
//			tracks.push_back(tv);
//			traj[m]=Point2f(k,l);
//			//vect2_aux.resize(num_points);
//
//			circle(Ic, Point(k,l), 1, Scalar( 255, 255 , 255 ),1,8,0);
//
//			for(int j=m;j<m+9;j++)
//			{
//				traj[j+1]=traj[j]+Point2f(u1[j].at<float>(l,k),v1[j].at<float>(l,k));
//				line(Ic,traj[j],traj[j+1],Scalar(0,0,255),0.05);
//
//
//				float a=u1[j].at<float>(traj[j]);
//				float b=v1[j].at<float>(traj[j]);
//
//				sum+=sqrt((a*a)+(b*b));
//
//				tracks[num_points](2*j)=u1[j].at<float>(traj[j]);
//				tracks[num_points](2*j+1)=v1[j].at<float>(traj[j]);
////				imshow("IMAGEN_TRACK",Ic);
////				waitKey();
////				cout<<"TRACK_NUM["<<num_points;
////				cout<<"]["<<2*j;
////				cout<<"] y TRACK_NUM["<<num_points;
////				cout<<"]["<<2*j+1;
////				cout<<"] Trazandose y guardandose"<<endl;
//
//			}
//
//			if(sum<1){
//				for(int j=0;j<tracks.size()-1;j++){
////					cout<<"\ntracks["<<j;
////					cout<<"]size()="<<tracks[j].size();
//					for(int l=0;l<tracks[j].size();l++){
//						double f=tracks[j+1][l];
//						tracks[j][l]=f;
//					}
//				}
//				num_points=num_points-1;
//				tracks.resize(tracks.size()-1);
//			}
//			if(sum >= 1){
//			//vect2_aux.push_back(sum);
//			}
//			sum=0;
//			num_points++;
////
////		    rectangle(Ic,ptx1,ptx2, Scalar( 255, 0 , 0 ),  2, 4 );
////		    circle(Ic, ptx1, 1, Scalar( 255, 0 , 0 ),1,8,0);
////		    circle(Ic, ptx2, 1, Scalar( 255, 0 , 0 ),1,8,0);
////		    namedWindow("FLOW");
//		    imshow("FLOW",Ic);
//		    waitKey(10);
//	//    out_capture.write(Ic);
//		}
//		}
//	}
//
////	cout<<"FIN_2.2"<<endl;
//
////	sprintf(file3,"../FLOW/FOTO%03d.jpg",count);
////	imwrite(file3,Ic);
////	count++;
//}
//

