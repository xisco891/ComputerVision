//--------------OpenCV-----------------//
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
#include "opencv2/objdetect.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/video/background_segm.hpp>
#include "opencv2/contrib/contrib.hpp"

//-----------------------------------------------------//
#include <sstream>
#include <iostream>
#include <fstream>
#include <iostream>
#include "opticalflow.h"
#include "stdio.h"
#include <eigen3/Eigen/Dense>
using namespace Eigen;
//------OpenGL----------------------------------//
#include <stdlib.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include "SDL/SDL.h"
#include <SDL/SDL_image.h>
#include <GL/glu.h>
//----------------------------------------------//
#undef _GLIBCXX_DEBUG
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <string>
#include "stdio.h"

using namespace Eigen;

#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "opticalmotion.h"


using namespace cv;
using namespace std;



static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int main( int argc, char* argv[] ){
	Mat frame(640,480,CV_8UC1,Scalar::all(0));
	bool isVideoReading;
	VideoCapture cap;
	cap.open(0);
	if( !cap.isOpened() ){
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	5 eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

		// Check for valid command line arguments, print usage
	    // if no arguments were given.
	    if (argc < 2) {
	        cout << "usage: " << argv[0] << " <csv.ext> <output_folder> " << endl;

	        /* Used for rotating the cube */
	        GLfloat rotationXaxis;
	        GLfloat rotationYaxis;
	        GLfloat rotationZaxis;

	        /* Storage For One Texture ( NEW ) */
	        GLuint texture[1];

	        /* Release resources and quit  */
	        void Quit( int returnCode ) {
	            SDL_Quit( );
	            exit( returnCode );
	        }

	        /* Loads in a bitmap as a GL texture */
	        int LoadGLTextures( ) {
	            int Status = FALSE;

	            /* Create storage space for the texture */
	            SDL_Surface *TextureImage[1];

	            /* Load The Bitmap into Memory */
	            if ((TextureImage[0] = SDL_LoadBMP("home/usr/Escritorio/cubo.bmp"))) {
	        	    Status = TRUE;
	        	    glGenTextures( 1, &texture[0] );
	        	    glBindTexture( GL_TEXTURE_2D, texture[0] );
	        	    glTexImage2D( GL_TEXTURE_2D, 0, 3, TextureImage[0]->w,
	        			  TextureImage[0]->h, 0, GL_BGR,
	        			  GL_UNSIGNED_BYTE, TextureImage[0]->pixels );
	        	    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	        	    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	                }

	            /* Free up some memory */
	            if ( TextureImage[0] )
	        	    SDL_FreeSurface( TextureImage[0] );

	            return Status;
	        }

	        /* function to reset our viewport after a window resize */
	        int resizeWindow( int width, int height ) {
	            /* Height / width ration */
	            GLfloat ratio;

	            /* Protect against a divide by zero */
	            if ( height == 0 )
	        	height = 1;

	            ratio = ( GLfloat )width / ( GLfloat )height;

	            /* Setup our viewport. */
	            glViewport( 0, 0, ( GLint )width, ( GLint )height );

	            /*
	             * change to the projection matrix and set
	             * our viewing volume.
	             */
	            glMatrixMode( GL_PROJECTION );
	            glLoadIdentity( );

	            /* Set our perspective */
	            gluPerspective( 45.0f, ratio, 0.1f, 100.0f );

	            /* Make sure we're chaning the model view and not the projection */
	            glMatrixMode( GL_MODELVIEW );

	            /* Reset The View */
	            glLoadIdentity( );

	            return( TRUE );
	        }

	        /* function to handle key press events */
	        void handleKeyPress( SDL_keysym *keysym )
	        {
	            switch ( keysym->sym )
	        	{
	         	case SDLK_ESCAPE:
	        	    /* ESC key was pressed */
	        	    Quit( 0 );
	        	    break;
	        	case SDLK_F1:
	        	    /* F1 key was pressed
	        	     * this toggles fullscreen mode
	        	     */
	        	    SDL_WM_ToggleFullScreen( surface );
	        	    break;
	        	default:
	        	    break;
	        	}

	            return;
	        }

	        /* OpenGL initialization function */
	        int initGL()
	        {

	            /* Load in the texture */
	            if ( !LoadGLTextures( ) )
	        	return FALSE;

	            /* Enable Texture Mapping ( NEW ) */
	            glEnable( GL_TEXTURE_2D );

	            /* Enable smooth shading */
	            glShadeModel( GL_SMOOTH );

	            /* Set the background black */
	            glClearColor( 0.0f, 0.0f, 0.0f, 0.5f );

	            /* Depth buffer setup */
	            glClearDepth( 1.0f );

	            /* Enables Depth Testing */
	            glEnable( GL_DEPTH_TEST );

	            /* The Type Of Depth Test To Do */
	            glDepthFunc( GL_LEQUAL );

	            /* Really Nice Perspective Calculations */
	            glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );

	            return( TRUE );
	        }

	        ///* Here goes our drawing code */
	        int drawGLScene()
	        {
	            /* Clear The Screen And The Depth Buffer */
	            glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	            /* Move Into The Screen 5 Units */
	            glLoadIdentity( );
	            glTranslatef( 0.0f, 0.0f, -5.0f );

	            glRotatef( rotationXaxis, 1.0f, 0.0f, 0.0f); /* Rotate On The X Axis */
	            glRotatef( rotationYaxis, 0.0f, 1.0f, 0.0f); /* Rotate On The Y Axis */
	            glRotatef( rotationZaxis, 0.0f, 0.0f, 1.0f); /* Rotate On The Z Axis */

	            /* Select Our Texture */
	            glBindTexture( GL_TEXTURE_2D, texture[0] );

	            /*
	             * Draw the cube. A cube consists of six quads, with four coordinates (glVertex3f)
	             * per quad.
	             *
	             */
	            glBegin(GL_QUADS);
	              /* Front Face */
	              glTexCoord2f( 0.0f, 1.0f ); glVertex3f( -1.0f, -1.0f, 1.0f );
	              glTexCoord2f( 1.0f, 1.0f ); glVertex3f(  1.0f, -1.0f, 1.0f );
	              glTexCoord2f( 1.0f, 0.0f ); glVertex3f(  1.0f,  1.0f, 1.0f );
	              glTexCoord2f( 0.0f, 0.0f ); glVertex3f( -1.0f,  1.0f, 1.0f );

	              /* Back Face */
	              glTexCoord2f( 0.0f, 0.0f ); glVertex3f( -1.0f, -1.0f, -1.0f );
	              glTexCoord2f( 0.0f, 1.0f ); glVertex3f( -1.0f,  1.0f, -1.0f );
	              glTexCoord2f( 1.0f, 1.0f ); glVertex3f(  1.0f,  1.0f, -1.0f );
	              glTexCoord2f( 1.0f, 0.0f ); glVertex3f(  1.0f, -1.0f, -1.0f );

	              /* Top Face */
	              glTexCoord2f( 1.0f, 1.0f ); glVertex3f( -1.0f,  1.0f, -1.0f );
	              glTexCoord2f( 1.0f, 0.0f ); glVertex3f( -1.0f,  1.0f,  1.0f );
	              glTexCoord2f( 0.0f, 0.0f ); glVertex3f(  1.0f,  1.0f,  1.0f );
	              glTexCoord2f( 0.0f, 1.0f ); glVertex3f(  1.0f,  1.0f, -1.0f );

	              /* Bottom Face */
	              /* Top Right Of The Texture and Quad */
	              glTexCoord2f( 0.0f, 1.0f ); glVertex3f( -1.0f, -1.0f, -1.0f );
	              glTexCoord2f( 1.0f, 1.0f ); glVertex3f(  1.0f, -1.0f, -1.0f );
	              glTexCoord2f( 1.0f, 0.0f ); glVertex3f(  1.0f, -1.0f,  1.0f );
	              glTexCoord2f( 0.0f, 0.0f ); glVertex3f( -1.0f, -1.0f,  1.0f );

	              /* Right face */
	              glTexCoord2f( 0.0f, 0.0f ); glVertex3f( 1.0f, -1.0f, -1.0f );
	              glTexCoord2f( 0.0f, 1.0f ); glVertex3f( 1.0f,  1.0f, -1.0f );
	              glTexCoord2f( 1.0f, 1.0f ); glVertex3f( 1.0f,  1.0f,  1.0f );
	              glTexCoord2f( 1.0f, 0.0f ); glVertex3f( 1.0f, -1.0f,  1.0f );

	              /* Left Face */
	              glTexCoord2f( 1.0f, 0.0f ); glVertex3f( -1.0f, -1.0f, -1.0f );
	              glTexCoord2f( 0.0f, 0.0f ); glVertex3f( -1.0f, -1.0f,  1.0f );
	              glTexCoord2f( 0.0f, 1.0f ); glVertex3f( -1.0f,  1.0f,  1.0f );
	              glTexCoord2f( 1.0f, 1.0f ); glVertex3f( -1.0f,  1.0f, -1.0f );
	            glEnd( );

	            /* Draw it to the screen */
	            SDL_GL_SwapBuffers( );

	          //  /* Rotate Cube */
	            rotationXaxis += 0.3f;
	            rotationYaxis += 0.2f;

	            return( TRUE );
	        }

	        //-------------------------------------------MAIN----------------------------------------------//

	        /* https://tutorialsplay.com/opengl/ */
	        int main( int argc, char **argv )
	        {
	            /* Flags to pass to SDL_SetVideoMode */
	            int videoFlags;
	            /* main loop variable */
	            int done = FALSE;
	            /* used to collect events */
	            SDL_Event event;
	            /* this holds some info about our display */
	            const SDL_VideoInfo *videoInfo;
	            /* whether or not the window is active */
	            int isActive = TRUE;

	            /* initialize SDL */
	            if ( SDL_Init( SDL_INIT_VIDEO ) < 0 )
	        	{
	        	    fprintf( stderr, "Video initialization failed: %sn",
	        		     SDL_GetError( ) );
	        	    Quit( 1 );
	        	}

	            /* Fetch the video info */
	            videoInfo = SDL_GetVideoInfo( );

	            if ( !videoInfo )
	        	{
	        	    fprintf( stderr, "Video query failed: %sn",
	        		     SDL_GetError( ) );
	        	    Quit( 1 );
	        	}

	            /* the flags to pass to SDL_SetVideoMode */
	            videoFlags  = SDL_OPENGL;          /* Enable OpenGL in SDL */
	            videoFlags |= SDL_GL_DOUBLEBUFFER; /* Enable double buffering */
	            videoFlags |= SDL_HWPALETTE;       /* Store the palette in hardware */
	            videoFlags |= SDL_RESIZABLE;       /* Enable window resizing */

	            /* This checks to see if surfaces can be stored in memory */
	            if ( videoInfo->hw_available )
	        	videoFlags |= SDL_HWSURFACE;
	            else
	        	videoFlags |= SDL_SWSURFACE;

	            /* This checks if hardware blits can be done */
	            if ( videoInfo->blit_hw )
	        	videoFlags |= SDL_HWACCEL;

	            /* Sets up OpenGL double buffering */
	            SDL_GL_SetAttribute( SDL_GL_DOUBLEBUFFER, 1 );

	            /* get a SDL surface */
	            surface = SDL_SetVideoMode( SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_BPP,
	        				videoFlags );

	            /* Verify there is a surface */
	            if ( !surface )
	        	{
	        	    fprintf( stderr,  "Video mode set failed: %sn", SDL_GetError( ) );
	        	    Quit( 1 );
	        	}

	            /* initialize OpenGL */
	            initGL( );

	            /* resize the initial window */
	            resizeWindow( SCREEN_WIDTH, SCREEN_HEIGHT );

	            /* wait for events */
	            while ( !done )
	        	{
	        	    /* handle the events in the queue */

	        	    while ( SDL_PollEvent( &event ) )
	        		{
	        		    switch( event.type )
	        			{
	        			case SDL_ACTIVEEVENT:
	        			    if ( event.active.gain == 0 )
	        				isActive = FALSE;
	        			    else
	        				isActive = TRUE;
	        			    break;
	        			case SDL_VIDEORESIZE:
	        			    /* handle resize event */
	        			    surface = SDL_SetVideoMode( event.resize.w, event.resize.h,16, videoFlags );
	        			    if ( !surface )
	        				{
	        				    fprintf( stderr, "Could not get a surface after resize: %sn", SDL_GetError( ) );
	        				    Quit( 1 );
	        				}
	        			    resizeWindow( event.resize.w, event.resize.h );
	        			    break;
	        			case SDL_KEYDOWN:
	        			    /* handle key presses */
	        			    handleKeyPress( &event.key.keysym );

	        			    break;
	        			case SDL_QUIT:
	        			    /* handle quit requests */
	        			    done = TRUE;
	        			    break;
	        			default:
	        			    break;
	        			}
	        		}

	        	    /* draw the scene */
	        	    if ( isActive )
	        		drawGLScene( );
	        	}

	            /* clean ourselves up and exit */
	            Quit( 0 );

	            /* Should never get here */
	            return( 0 );
	        }
      exit(1);
	    }
	    string output_folder = ".";
	    if (argc == 3) {
	        output_folder = string(argv[2]);
	    }
	    // Get the path to your CSV.
	    string fn_csv = string(argv[1]);
	    // These vectors hold the images and corresponding labels.
	    vector<Mat> images;
	    vector<int> labels;
	    // Read in the data. This can fail if no valid
	    // input filename is given.
	    try {
	        read_csv(fn_csv, images, labels);
	    } catch (cv::Exception& e) {
	        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
	        // nothing more we can do
	        exit(1);
	    }
	    // Quit if there are not enough images for this demo.
	    if(images.size() <= 1) {
	        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
	        CV_Error(CV_StsError, error_message);
	    }
	    // Get the height from the first image. We'll need this
	    // later in code to reshape the images to their original
	    // size:
	    int height = images[0].rows;
	    // The following lines simply get the last images from
	    // your dataset and remove it from the vector. This is
	    // done, so that the training data (which we learn the
	    // cv::FaceRecognizer on) and the test data we test
	    // the model with, do not overlap.
	    Mat testSample = images[images.size() - 1];
	    int testLabel = labels[labels.size() - 1];
	    images.pop_back();
	    labels.pop_back();
	    // The following lines create an Fisherfaces model for
	    // face recognition and train it with the images and
	    // labels read from the given CSV file.
	    // If you just want to keep 10 Fisherfaces, then call
	    // the factory method like this:
	    //
	    //      cv::createFisherFaceRecognizer(10);
	    //
	    // However it is not useful to discard Fisherfaces! Please
	    // always try to use _all_ available Fisherfaces for
	    // classification.
	    //
	    // If you want to create a FaceRecognizer with a
	    // confidence threshold (e.g. 123.0) and use _all_
	    // Fisherfaces, then call it with:
	    //
	    //      cv::createFisherFaceRecognizer(0, 123.0);
	    //
	    Ptr<FaceRecognizer> model =  createFisherFaceRecognizer();
	    model->train(images, labels);
	    int predictedLabel = model->predict(testSample);
	       //
	       // To get the confidence of a prediction call the model with:
	       //
	       //      int predictedLabel = -1;
	       //      double confidence = 0.0;
	       //      model->predict(testSample, predictedLabel, confidence);
	       //
	       string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	       cout << result_message << endl;
	       // Here is how to get the eigenvalues of this Eigenfaces model:
	       Mat eigenvalues = model->eigenvalues;
	       // And we can do the same to display the Eigenvectors (read Eigenfaces):
	       Mat W = model->getMat("eigenvectors");
	       // Get the sample mean from the training data
	       Mat mean = model->getMat("mean");
	       // Display or save:
	       if(argc == 2) {
	           imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
	       } else {
	           imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
	       }
	       // Display or save the first, at most 16 Fisherfaces:
	       for (int i = 0; i < min(16, W.cols); i++) {
	           string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
	           cout << msg << endl;
	           // get eigenvector #i
	           Mat ev = W.col(i).clone();
	           // Reshape to original size & normalize to [0...255] for imshow.
	           Mat grayscale = norm_0_255(ev.reshape(1, height));
	           // Show the image & apply a Bone colormap for better sensing.
	           Mat cgrayscale;
	           applyColorMap(grayscale, cgrayscale, COLORMAP_BONE);
	           // Display or save:
	           if(argc == 2) {
	               imshow(format("fisherface_%d", i), cgrayscale);
	           } else {
	               imwrite(format("%s/fisherface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
	           }
	       }
	       // Display or save the image reconstruction at some predefined steps:
	       for(int num_component = 0; num_component < min(16, W.cols); num_component++) {
	           // Slice the Fisherface from the model:
	           Mat ev = W.col(num_component);
	           Mat projection = subspaceProject(ev, mean, images[0].reshape(1,1));
	           Mat reconstruction = subspaceReconstruct(ev, mean, projection);
	           // Normalize the result:
	           reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
	           // Display or save:
	           if(argc == 2) {
	               imshow(format("fisherface_reconstruction_%d", num_component), reconstruction);
	           } else {
	               imwrite(format("%s/fisherface_reconstruction_%d.png", output_folder.c_str(), num_component), reconstruction);
	           }
	       }
	       // Display if we are not writing to an output folder:
	       if(argc == 2) {
	           waitKey(0);
	       }
	       return 0;


	for(;;){
		cap.read(frame);
		//---FILTRADO-----//
		//GaussianBlur(frame,frame, Size(7,7), 0,0);
		//----------------//
	 	imshow("LIVE_VIDEO",frame);




		if(waitKey(30) >= 0) break;

	}



}

