#include <iostream>
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
#include "stdio.h"
#include <eigen3/Eigen/Dense>


#ifdef WIN32
    #define WIN32_LEAN_AND_MEAN 1
    #define NOMINMAX 1
    #include <windows.h>
#endif
#if defined(_WIN64)
    #include <windows.h>
#endif

#if defined(__APPLE__)
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
#else
    #include <GL/gl.h>
    #include <GL/glu.h>
#endif
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/video/tracking.hpp"
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

#include "opencv2/core/core.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

const int win_width = 800;
const int win_height = 640;

struct DrawData
{
    ogl::Arrays arr;
    ogl::Texture2D tex;
    ogl::Buffer indices;
};

void draw(void* userdata);

void draw(void* userdata)
{
    DrawData* data = static_cast<DrawData*>(userdata);

    glRotated(0.6, 0, 1, 0);

    ogl::render(data->arr, data->indices, ogl::TRIANGLES);
}

int main(int argc, char* argv[])
{
//    string filename;
//    if (argc < 2)
//    {
//        cout << "Usage: " << argv[0] << " image" << endl;
//        filename = "../data/lena.jpg";
//    }
//    else
//        filename = argv[1];

    Mat img = imread("/home/usr/Escritorio/PHOTOS/lena.jpg");
    if (img.empty())
    {
        cerr << "Can't open image " << endl;
        return -1;
    }

    namedWindow("OpenGL", WINDOW_OPENGL);
    waitKey();
    resizeWindow("OpenGL", win_width, win_height);

    Mat_<Vec2f> vertex(1, 4);
    vertex << Vec2f(-1, 1), Vec2f(-1, -1), Vec2f(1, -1), Vec2f(1, 1);

    Mat_<Vec2f> texCoords(1, 4);
    texCoords << Vec2f(0, 0), Vec2f(0, 1), Vec2f(1, 1), Vec2f(1, 0);

    Mat_<int> indices(1, 6);
    indices << 0, 1, 2, 2, 3, 0;

    DrawData data;

    data.arr.setVertexArray(vertex);
    data.arr.setTexCoordArray(texCoords);
    data.indices.copyFrom(indices);
    data.tex.copyFrom(img);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)win_width / win_height, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0, 0, 3, 0, 0, 0, 0, 1, 0);

    glEnable(GL_TEXTURE_2D);
    data.tex.bind();

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvi(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glDisable(GL_CULL_FACE);

    setOpenGlDrawCallback("OpenGL", draw, &data);

    for (;;)
    {
        updateWindow("OpenGL");
        int key = waitKey(40);
        if ((key & 0xff) == 27)
            break;
    }

    setOpenGlDrawCallback("OpenGL", 0, 0);
    destroyAllWindows();

    return 0;
}
