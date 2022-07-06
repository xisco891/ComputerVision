#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/video/background_segm.hpp>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iostream>
#include "opticalflow.h"
#include "stdio.h"
#include <eigen3/Eigen/Dense>



#include <stdlib.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include "SDL/SDL.h"
#include <SDL/SDL_image.h>
#include <GL/glu.h>


using namespace Eigen;
#include "opencv2/objdetect/objdetect.hpp"




#undef _GLIBCXX_DEBUG
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <time.h>
//#include "opticalmotion.h"
#include <iostream>

#include <sstream>
#include <string>
#include <iostream>
#include "opencv2/objdetect.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>



#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/video/background_segm.hpp>


#include <sstream>
#include <iostream>
#include <fstream>
#include <iostream>
//#include "calcSiftSurf.h"

#include "stdio.h"
#include <eigen3/Eigen/Dense>
using namespace Eigen;
#undef _GLIBCXX_DEBUG
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "opticalmotion.h"

#include <iostream>


using namespace cv;
using namespace std;

//void detectAndDisplay( Mat frame, CascadeClassifier& face_cascade, CascadeClassifier& eyes_cascade );
//initial min and max HSV filter values.
//these will be changed using trackbars
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;
//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS=50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 20*20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;
//names that will appear at the top of each window
const string windowName = "Original Image";
const string windowName1 = "HSV Image";
const string windowName2 = "Thresholded Image";
const string windowName3 = "After Morphological Operations";
const string trackbarWindowName = "Trackbars";

void on_trackbar( int, void* )
{//This function gets called whenever a
	// trackbar position is changed

}


string intToString(int number){


	std::stringstream ss;
	ss << number;
	return ss.str();
}
static void help()
{
        cout << "\nThis program demonstrates usage of depth sensors (Kinect, XtionPRO,...).\n"
                        "The user gets some of the supported output images.\n"
            "\nAll supported output map types:\n"
            "1.) Data given from depth generator\n"
            "   CAP_OPENNI_DEPTH_MAP            - depth values in mm (CV_16UC1)\n"
            "   CAP_OPENNI_POINT_CLOUD_MAP      - XYZ in meters (CV_32FC3)\n"
            "   CAP_OPENNI_DISPARITY_MAP        - disparity in pixels (CV_8UC1)\n"
            "   CAP_OPENNI_DISPARITY_MAP_32F    - disparity in pixels (CV_32FC1)\n"
            "   CAP_OPENNI_VALID_DEPTH_MASK     - mask of valid pixels (not ocluded, not shaded etc.) (CV_8UC1)\n"
            "2.) Data given from RGB image generator\n"
            "   CAP_OPENNI_BGR_IMAGE            - color image (CV_8UC3)\n"
            "   CAP_OPENNI_GRAY_IMAGE           - gray image (CV_8UC1)\n"
         << endl;
}


static void colorizeDisparity( const Mat& gray, Mat& rgb, double maxDisp=-1.f, float S=1.f, float V=1.f )
{
    CV_Assert( !gray.empty() );
    CV_Assert( gray.type() == CV_8UC1 );

    if( maxDisp <= 0 )
    {
        maxDisp = 0;
        minMaxLoc( gray, 0, &maxDisp );

    }
    rgb.create( gray.size(), CV_8UC3 );
    rgb = Scalar::all(0);
    if( maxDisp < 1 )
        return;

    for( int y = 0; y < gray.rows; y++ )
    {
        for( int x = 0; x < gray.cols; x++ )
        {
            float d = gray.at<float>(y,x);
            unsigned int H = ((float)maxDisp - d) * 240 / (float)maxDisp;
//            cout<<"Distances:"<<H<<endl;
//            waitKey(200);
            unsigned int hi = (H/60) % 6;
            float f = H/60.f - H/60;
            float p = V * (1 - S);
            float q = V * (1 - f * S);
            float t = V * (1 - (1 - f) * S);


            Point3f res;

            if( hi == 0 ) //R = V,  G = t,  B = p
                res = Point3f( p, t, V );
            if( hi == 1 ) // R = q, G = V,  B = p
                res = Point3f( p, V, q );
            if( hi == 2 ) // R = p, G = V,  B = t
                res = Point3f( t, V, p );
            if( hi == 3 ) // R = p, G = q,  B = V
                res = Point3f( V, q, p );
            if( hi == 4 ) // R = t, G = p,  B = V
                res = Point3f( V, p, t );
            if( hi == 5 ) // R = V, G = p,  B = q
                res = Point3f( q, p, V );

            float b = (float)(std::max(0.f, std::min (res.x, 1.f)) * 255.f);
            float g = (float)(std::max(0.f, std::min (res.y, 1.f)) * 255.f);
            float r = (float)(std::max(0.f, std::min (res.z, 1.f)) * 255.f);
//            cout<<"Distances:"<<H<<endl;
//            waitKey(200);

            rgb.at<Point3_<float> >(y,x) = Point3_<float>(b, g, r);

        }

    }
}

static float getMaxDisparity( VideoCapture& capture )
{
    const int minDistance = 400; // mm
    float b = (float)capture.get( CAP_OPENNI_DEPTH_GENERATOR_BASELINE );
    float F = (float)capture.get( CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH ); // pixels
    return b * F / minDistance;
}

static void printCommandLineParams()
{
    cout << "-cd       Colorized disparity? (0 or 1; 1 by default) Ignored if disparity map is not selected to show." << endl;
    cout << "-fmd      Fixed max disparity? (0 or 1; 0 by default) Ignored if disparity map is not colorized (-cd 0)." << endl;
    cout << "-mode     image mode: resolution and fps, supported three values:  0 - CAP_OPENNI_VGA_30HZ, 1 - CAP_OPENNI_SXGA_15HZ," << endl;
    cout << "          2 - CAP_OPENNI_SXGA_30HZ (0 by default). Ignored if rgb image or gray image are not selected to show." << endl;
    cout << "-m        Mask to set which output images are need. It is a string of size 5. Each element of this is '0' or '1' and" << endl;
    cout << "          determine: is depth map, disparity map, valid pixels mask, rgb image, gray image need or not (correspondently)?" << endl ;
    cout << "          By default -m 01010 i.e. disparity map and rgb image will be shown." << endl ;
    cout << "-r        Filename of .oni video file. The data will grabbed from it." << endl ;
}

static void parseCommandLine( int argc, char* argv[], bool& isColorizeDisp, bool& isFixedMaxDisp, int& imageMode, bool retrievedImageFlags[],
                       string& filename, bool& isFileReading )
{
    // set defaut values
    isColorizeDisp = true;
    isFixedMaxDisp = false;
    imageMode = 0;

    retrievedImageFlags[0] = false;
    retrievedImageFlags[1] = true;
    retrievedImageFlags[2] = true;
    retrievedImageFlags[3] = true;
    retrievedImageFlags[4] = true;

    filename.clear();
    isFileReading = false;

    if( argc == 1 )
    {
        help();
    }
    else
    {
        for( int i = 1; i < argc; i++ )
        {
            if( !strcmp( argv[i], "--help" ) || !strcmp( argv[i], "-h" ) )
            {
                printCommandLineParams();
                exit(0);
            }
            else if( !strcmp( argv[i], "-cd" ) )
            {
                isColorizeDisp = atoi(argv[++i]) == 0 ? false : true;
            }
            else if( !strcmp( argv[i], "-fmd" ) )
            {
                isFixedMaxDisp = atoi(argv[++i]) == 0 ? false : true;
            }
            else if( !strcmp( argv[i], "-mode" ) )
            {
                imageMode = atoi(argv[++i]);
            }
            else if( !strcmp( argv[i], "-m" ) )
            {
                string mask( argv[++i] );
                if( mask.size() != 5)
                    CV_Error( Error::StsBadArg, "Incorrect length of -m argument string" );
                int val = atoi(mask.c_str());

                int l = 100000, r = 10000, sum = 0;
                for( int j = 0; j < 5; j++ )
                {
                    retrievedImageFlags[j] = ((val % l) / r ) == 0 ? false : true;
                    l /= 10; r /= 10;
                    if( retrievedImageFlags[j] ) sum++;
                }

                if( sum == 0 )
                {
                    cout << "No one output image is selected." << endl;
                    exit(0);
                }
            }
            else if( !strcmp( argv[i], "-r" ) )
            {
                filename = argv[++i];
                isFileReading = true;
            }
            else
            {
                cout << "Unsupported command line argument: " << argv[i] << "." << endl;
                exit(-1);
            }
        }
    }
}

void drawObject(int x, int y,Mat &frame){

	//use some of the openCV drawing functions to draw crosshairs
	//on your tracked image!

    //UPDATE:JUNE 18TH, 2013
    //added 'if' and 'else' statements to prevent
    //memory errors from writing off the screen (ie. (-25,-25) is not within the window!)

	circle(frame,Point(x,y),20,Scalar(0,255,0),2);
    if(y-25>0)
    line(frame,Point(x,y),Point(x,y-25),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(x,0),Scalar(0,255,0),2);
    if(y+25<FRAME_HEIGHT)
    line(frame,Point(x,y),Point(x,y+25),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(x,FRAME_HEIGHT),Scalar(0,255,0),2);
    if(x-25>0)
    line(frame,Point(x,y),Point(x-25,y),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(0,y),Scalar(0,255,0),2);
    if(x+25<FRAME_WIDTH)
    line(frame,Point(x,y),Point(x+25,y),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(FRAME_WIDTH,y),Scalar(0,255,0),2);

	putText(frame,intToString(x)+","+intToString(y),Point(x,y+30),1,1,Scalar(0,255,0),2);

}
void createTrackbars(){
	//create window for trackbars


    namedWindow(trackbarWindowName,0);
	//create memory to store trackbar name on window
	char TrackbarName[50];

//	sprintf( TrackbarName, "H_MIN", H_MIN);
//	sprintf( TrackbarName, "H_MAX", H_MAX);
//	sprintf( TrackbarName, "S_MIN", S_MIN);
//	sprintf( TrackbarName, "S_MAX", S_MAX);
//	sprintf( TrackbarName, "V_MIN", V_MIN);
//	sprintf( TrackbarName, "V_MAX", V_MAX);

	//create trackbars and insert them into window
	//3 parametretrievedImageFlagsers are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
	//the max value the trackbar can move (eg. H_HIGH),
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
	//                       retrievedImageFlags           ---->    ---->     ---->
    createTrackbar( "H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar );
    createTrackbar( "H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar );
    createTrackbar( "S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar );
    createTrackbar( "S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar );
    createTrackbar( "V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar );
    createTrackbar( "V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar );


}

void getDepthData(Mat depthMat,int& count_time_in,vector<double>& time_1, bool& dentro){

	depthMat.convertTo(depthMat,CV_32FC1);
	for( int x = 0; x < depthMat.cols; x++ )
	{
		for( int y = 0; y < depthMat.rows; y++ )
		{
			if((depthMat.at<float>(y,x)/1000)<0.70 && (depthMat.at<float>(y,x)/1000)>0.35){
				time_1[count_time_in]=getTickCount();
				count_time_in++;
				dentro=true;
				return;
			}
		}
	}
	count_time_in=0;

	cout<<"No Hay Nadie alrededor\n"<<endl;

}

void shiftImages(vector<Mat>& vimg,int sn)
{
	for(int i=0;i<sn;i++)
	{
		vimg[i]=vimg[i+sn].clone();
	}
}


/** @function detectAndDisplay */
//void detectAndDisplay( Mat frame, CascadeClassifier& face_cascade, CascadeClassifier& eyes_cascade )
//{
//    std::vector<Rect> faces;
//    Mat frame_gray;
//
//    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
//    equalizeHist( frame_gray, frame_gray );
//
//    //-- Detect faces
//    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
//
//    for ( size_t i = 0; i < faces.size(); i++ )
//    {
//        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
//        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
//
//        Mat faceROI = frame_gray( faces[i] );
//        std::vector<Rect> eyes;
//
//        //-- In each face, detect eyes
//        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
//
//        for ( size_t j = 0; j < eyes.size(); j++ )
//        {
//            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
//            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
//            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
//        }
//    }
//    //-- Show what you got
//    imshow("WINDOW", frame );
//}


void crear_ventana (Mat land, Mat& ventana,Point p1,Point p2,Point2f desplaz){

	float dx=0,dy=0;
	for(int i=p1.x;i<p2.x;i++){
		for(int j=p1.y;j<p2.y;j++){
			dx=(i+8*(int)desplaz.x);
			dy=(j+8*(int)desplaz.y);
//			cout<<"dx:"<<desplaz.x;
//			cout<<"dy:"<<desplaz.y;
//			waitKey(10);
			ventana.ptr<Point3_<float> >(j,i)[0]=land.ptr<Point3_<float> >(dy,dx)[0];
			ventana.ptr<Point3_<float> >(j,i)[1]=land.ptr<Point3_<float> >(dy,dx)[1];
			ventana.ptr<Point3_<float> >(j,i)[2]=land.ptr<Point3_<float> >(dy,dx)[2];

		}
	}
	//rectangle(ventana,p1,p2, Scalar( 255, 0 , 0 ),  2, 4 );
	ventana.convertTo(ventana,CV_8UC1);
	//land.convertTo(land,CV_8UC1);
	imshow("VENTANA_ACTUALIZADA",ventana);

}


//void drawCube() {
//	/*
//     * Draw the cube. A cube consists of six quads, with four coordinates (glVertex3f)
//     * per quad.
//     *
//     */
//	cout<<"DIBUJANDO"<<endl;
//	glBegin(GL_QUADS);
//      /* Front Face */
//      glTexCoord2f( 0.0f, 1.0f ); glVertex3f( -1.0f, -1.0f, 1.0f );
//      glTexCoord2f( 1.0f, 1.0f ); glVertex3f(  1.0f, -1.0f, 1.0f );
//      glTexCoord2f( 1.0f, 0.0f ); glVertex3f(  1.0f,  1.0f, 1.0f );
//      glTexCoord2f( 0.0f, 0.0f ); glVertex3f( -1.0f,  1.0f, 1.0f );
//
//      /* Back Face */
//      glTexCoord2f( 0.0f, 0.0f ); glVertex3f( -1.0f, -1.0f, -1.0f );
//      glTexCoord2f( 0.0f, 1.0f ); glVertex3f( -1.0f,  1.0f, -1.0f );
//      glTexCoord2f( 1.0f, 1.0f ); glVertex3f(  1.0f,  1.0f, -1.0f );
//      glTexCoord2f( 1.0f, 0.0f ); glVertex3f(  1.0f, -1.0f, -1.0f );
//
//      /* Top Face */
//      glTexCoord2f( 1.0f, 1.0f ); glVertex3f( -1.0f,  1.0f, -1.0f );
//      glTexCoord2f( 1.0f, 0.0f ); glVertex3f( -1.0f,  1.0f,  1.0f );
//      glTexCoord2f( 0.0f, 0.0f ); glVertex3f(  1.0f,  1.0f,  1.0f );
//      glTexCoord2f( 0.0f, 1.0f ); glVertex3f(  1.0f,  1.0f, -1.0f );
//
//      /* Bottom Face */
//      /* Top Right Of The Texture and Quad */
//      glTexCoord2f( 0.0f, 1.0f ); glVertex3f( -1.0f, -1.0f, -1.0f );
//      glTexCoord2f( 1.0f, 1.0f ); glVertex3f(  1.0f, -1.0f, -1.0f );
//      glTexCoord2f( 1.0f, 0.0f ); glVertex3f(  1.0f, -1.0f,  1.0f );
//      glTexCoord2f( 0.0f, 0.0f ); glVertex3f( -1.0f, -1.0f,  1.0f );
//
//      /* Right face */
//      glTexCoord2f( 0.0f, 0.0f ); glVertex3f( 1.0f, -1.0f, -1.0f );
//      glTexCoord2f( 0.0f, 1.0f ); glVertex3f( 1.0f,  1.0f, -1.0f );
//      glTexCoord2f( 1.0f, 1.0f ); glVertex3f( 1.0f,  1.0f,  1.0f );
//      glTexCoord2f( 1.0f, 0.0f ); glVertex3f( 1.0f, -1.0f,  1.0f );
//
//      /* Left Face */
//      glTexCoord2f( 1.0f, 0.0f ); glVertex3f( -1.0f, -1.0f, -1.0f );
//      glTexCoord2f( 0.0f, 0.0f ); glVertex3f( -1.0f, -1.0f,  1.0f );
//      glTexCoord2f( 0.0f, 1.0f ); glVertex3f( -1.0f,  1.0f,  1.0f );
//      glTexCoord2f( 1.0f, 1.0f ); glVertex3f( -1.0f,  1.0f, -1.0f );
//    glEnd();
//    cout<<"FINAL"<<endl;
//}
//
//void drawSquare() {
//
//
//    glBegin(GL_QUADS);
//      glTexCoord2f( 0.0f, 1.0f ); glVertex3f( -2.0f, -2.0f, 2.0f );
//      glTexCoord2f( 1.0f, 1.0f ); glVertex3f(  2.0f, -2.0f, 2.0f );
//      glTexCoord2f( 1.0f, 0.0f ); glVertex3f(  2.0f,  2.0f, 2.0f );
//      glTexCoord2f( 0.0f, 0.0f ); glVertex3f( -2.0f,  2.0f, 2.0f );
//    glEnd( );
//}

int main( int argc, char* argv[] )
{
	opticalFlow opt;
	int imgnum = 10;
	int back=0;
	vector<Mat> imgs_aux(imgnum);
	vector<Mat> imgs_final(imgnum);
	vector<Mat> imgs_flow(2);
	vector<Mat> imgs_depth(2);
	int count_sec=0;

	int step = 0;
	int MAX_KERNEL_LENGTH=6;
	int count_time_in=0;
	vector<double> time(100);
	vector<double> time_1(100);
	double time_start=0;
	double time_now=0;
	double time_2=0;
	double time_period=0;


	//PRUEBAS DE OPENGL-------------------------------------
//	Mat imagen;
//	drawCube();
//	cout<<"CUBO DIBUJADO"<<endl;
//	imshow("",imagen);
//	waitKey();

	//------------------------------------------------------


	bool isColorizeDisp, isFixedMaxDisp;
	bool dentro;
	int imageMode;
    bool retrievedImageFlags[5];
    string filename;
    bool isVideoReading;
    parseCommandLine( argc, argv, isColorizeDisp, isFixedMaxDisp, imageMode, retrievedImageFlags, filename, isVideoReading );
    bool trackObjects = true;
    bool useMorphOps = true;
	Mat threshold;
	int x=0, y=0;
	int i=0;

	//----------INICIALIZACION VENTANA------------------------//
	Mat land=imread("/home/usr/PROYECTO/KINECT/LANDSCAPE.jpeg");
	Size size(1280,640);
	Mat land2;
	resize(land,land2, size);
	land2.convertTo(land2,CV_8UC3);
	Mat pared(land2.rows,land2.cols,CV_8UC3, Scalar( 192, 224 , 224 ));
	//pared.convertTo(pared,CV_8UC1);
	imshow("PARED",pared);
	//imshow("LANDSCAPE_REDUCIDO :)",land2);

	Point p1,p2,p3,p4;
	int despl_x=0,despl_y=0;

	//VENTANA 1
	p1.x=(size.width)/4-150;
	p1.y=(size.height)/2-250;
	p2.x=(size.width)/4+150;
	p2.y=(size.height)/2+100;
	//rectangle(land2,p1,p2, Scalar( 255, 0 , 0 ),  2, 4 );
	imshow("VENTANA1",land2);
	//VENTANA2
	p3.x=(size.width-(size.width)/4)-150;
	p3.y=(size.height-(size.height)/2)-250;
	p4.x=(size.width-(size.width)/4)+150;
	p4.y=(size.height-(size.height)/2)+100;
	//rectangle(land2,p3,p4, Scalar( 255, 0 , 0 ),  2, 4 );
	imshow("VENTANA_1Y2",land2);
	imshow("VENTANA_FINAL",pared);//FACE DETECTION VARIABLES


	//---------STREAMING CAPTURE------------------------//

    cout << "Device opening ..." << endl;
    VideoCapture capture;
    if( isVideoReading )
        capture.open( filename );
    else
    {
        capture.open( CAP_OPENNI2 );
        cout<<"Hola"<<endl;
        if( !capture.isOpened() )
            capture.open( CAP_OPENNI );
        	cout<<"Hola2"<<endl;
    }

    cout << "done." << endl;

    if( !capture.isOpened() )
    {
        cout << "Can not open a capture object." << endl;
        return -1;
    }
    if( !isVideoReading )
    {
        bool modeRes=false;
        switch ( imageMode )
        {
            case 0:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_VGA_30HZ );
                break;
            case 1:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_SXGA_15HZ );
                break;
            case 2:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_SXGA_30HZ );
                break;
                //The following modes are only supported by the Xtion Pro Live
            case 3:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_QVGA_30HZ );
                break;
            case 4:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_QVGA_60HZ );
                break;
            default:
                CV_Error( Error::StsBadArg, "Unsupported image mode property.\n");
        }
        if (!modeRes)
            cout << "\nThis image mode is not supported by the device, the default value (CV_CAP_OPENNI_SXGA_15HZ) will be used.\n" << endl;
    }

    // Print some avalible device settings.
    cout << "\nDepth generator output mode:" << endl <<
            "FRAME_WIDTH      " << capture.get( CAP_PROP_FRAME_WIDTH ) << endl <<
            "FRAME_HEIGHT     " << capture.get( CAP_PROP_FRAME_HEIGHT ) << endl <<
            "FRAME_MAX_DEPTH  " << capture.get( CAP_PROP_OPENNI_FRAME_MAX_DEPTH ) << " mm" << endl <<
            "FPS              " << capture.get( CAP_PROP_FPS ) << endl <<
            "REGISTRATION     " << capture.get( CAP_PROP_OPENNI_REGISTRATION ) << endl;
    if( capture.get( CAP_OPENNI_IMAGE_GENERATOR_PRESENT ) )
    {
        cout <<
            "\nImage generator output mode:" << endl <<
            "FRAME_WIDTH   " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FRAME_WIDTH ) << endl <<
            "FRAME_HEIGHT  " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FRAME_HEIGHT ) << endl <<
            "FPS           " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FPS ) << endl;
    }
    else
    {
        cout << "\nDevice doesn't contain image generator." << endl;
        if (!retrievedImageFlags[0] && !retrievedImageFlags[1] && !retrievedImageFlags[2])
            return 0;
    }
    double fps=capture.get( CAP_PROP_FPS );

    for(;;)
    {
        Mat depthMap;
        Mat validDepthMap;
        Mat disparityMap;
        Mat bgrImage;
        Mat grayImage;
        Mat smoothed;
        Mat grayIm;
        Mat HSV;



        if( !capture.grab() )
        {
            cout << "Can not grab images." << endl;
            return -1;
        }
        else
        {
        	if( retrievedImageFlags[0] && capture.retrieve( depthMap, CAP_OPENNI_DEPTH_MAP ) )
        	{
//        		crear_ventana(land2,pared,p1,p2,despl_x,despl_y);
//        		crear_ventana(land2,pared,p3,p4,despl_x,despl_y);
//        		const float scaleFactor = 0.05f;
//        		Mat show;
//        		getDepthData(depthMap,count_time_in,time_1,dentro);
//        		cout<<"HOLA"<<endl;
//        		while(dentro){
//        			cout<<"OBJECTO_EN_ZONA DE INTERES"<<endl;
//        			time_start=time_1[0];
//        			time_now=getTickCount();
//        			time_period=(time_now-time_start)/getTickFrequency();
//        			cout<<"Time:"<<time_period;
//        			cout<<""<<endl;
//        			if(time_period>2){
//        				cout<<"+2_SEGUNDOS"<<endl;
//        				cout<<"\n LANZANDO VIDEO---------------"<<endl;
//        				cout<<"\n time_start:"<<time_start<<endl;
//        				cout<<"\n time_final:"<<time_now<<endl;
//        				cout<<"TIEMPO EXACTO DELANTE DEL ORDENADOR"<<time_period;
//        				cout<<"\n"<<endl;
//        				//waitKey(2000);
//        				dentro=false;
//        				count_time_in=0;
//        			}
//
//        			else{
//        				dentro=false;
//        			}
//        		}
//        		cout<<"HOLA2"<<endl;
//        		depthMap.convertTo( show, CV_8UC1, scaleFactor );
//        		imshow( "depth map", show );

        	}

        	if( retrievedImageFlags[1] && capture.retrieve( disparityMap, CAP_OPENNI_DISPARITY_MAP ) )
        	{
        		//FORMATO CV_8UC1
        	//	imshow("Disparity map",disparityMap);
          		//opt.depth_boxing(disparityMap);
//          		opt.skeleton_tracking(3);
//          		opt.KmeansClust_skeleton(8);
//          		opt.draw_skeleton();
//
//          		opt.num_points=0;


//        		if( isColorizeDisp )
//        		{
//        			Mat colorDisparityMap;
//
//        			colorizeDisparity( disparityMap, colorDisparityMap, isFixedMaxDisp
//        					? getMaxDisparity(capture) : -1 );
//        			Mat validColorDisparityMap;
//        			colorDisparityMap.copyTo( validColorDisparityMap, disparityMap != 0 );
//        			imshow( "colorized disparity map", validColorDisparityMap );
//
//                }
//                else
//                {
//                    imshow( "original disparity map", disparityMap );
//                }
            }

//            if( retrievedImageFlags[2] && capture.retrieve( validDepthMap, CAP_OPENNI_VALID_DEPTH_MASK ) )
//                imshow( "valid depth mask", validDepthMap );


            if( retrievedImageFlags[3] && capture.retrieve( bgrImage, CAP_OPENNI_BGR_IMAGE ) )

            //GaussianBlur(bgrImage,smoothed, Size(3,3), 0,0);

            for(int j=1;j<MAX_KERNEL_LENGTH;j=j+2){
            	bilateralFilter(bgrImage,smoothed,j,j*2,j/2);
            }

            cvtColor(smoothed,grayImage, CV_BGR2GRAY);
        	imgs_flow[step]=grayImage.clone();
        	imgs_depth[step]=disparityMap.clone();
        	cout<<"i:"<<i<<endl;
        	i++;
        	step++;

        	if(step==2){
        		count_sec++;
        		cout<<"HOLA"<<endl;
        		opt.loadImages(imgs_flow);
        		cout<<"HOLA2"<<endl;
        		opt.depth_boxing(imgs_depth[0]);
        		cout<<"HOLA3"<<endl;
        		opt.skeleton_flow(10,step);
        		cout<<"HOLA4"<<endl;
        		crear_ventana(land2,pared,p1,p2,opt.desplaz);
        	    crear_ventana(land2,pared,p3,p4,opt.desplaz);
        	    if(count_sec==10){
        	    	opt.desplaz=Point(0,0);
        	    	cout<<"SECOND"<<endl;
        	    	waitKey(1000);
        	    	count_sec=0;
        	    }

        	    step=0;

        		//        		opt.skeleton_tracking(4);
        		//        		opt.KmeansClust_skeleton(6);
        		//opt.draw_skeleton();
        		//opt.points_flow(2);
        		//        		shiftImages(imgs_flow, step);
        		//        		shiftImages(imgs_depth,step);

        	}







//        	if( retrievedImageFlags[4] && capture.retrieve( grayIm, CAP_OPENNI_GRAY_IMAGE ) ){
//        		//   	imshow("grayImage:",grayIm);
//        	}
        }

        if( waitKey( 30 ) >= 0 )
        	break;
    }

    return 0;
}
