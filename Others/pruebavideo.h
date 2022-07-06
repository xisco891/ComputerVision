#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <iostream>
#undef _GLIBCXX_DEBUG



using namespace cv;
using namespace std;

void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
	                    double scale, const Scalar& color);

