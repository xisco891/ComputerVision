/*
 /*
 * OF_visualization.h
 *
 *  Created on: May 27, 2013
 *      Author: mahmoud
 */

#ifndef OF_VISUALIZATION_H_
#define OF_VISUALIZATION_H_
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
class OF_visualization
{
public:
//	Draw a vector field based on horizontal and vertical flow fields
	void drawVectorsField(IplImage* u, IplImage* v, int xSpace = 5, int ySpace = 5, float cutoff = 1, int multiplier = 5, CvScalar color = CV_RGB(255,200,100))
	{
		int x, y;
		CvPoint p0 = cvPoint(0,0);
		CvPoint p1 = cvPoint(0,0);
		IplImage* imgMotion = cvCreateImage(cvSize(u->width, u->height), 8, 3);cvZero(imgMotion);
		float deltaX, deltaY, angle, hyp;

		for(y = ySpace; y < u->height; y+= ySpace )
		{
			for(x = xSpace; x < u->width; x+= xSpace )
			{
				p0.x = x;
				p0.y = y;
				deltaX = *((float*)(u->imageData + y*u->widthStep)+x);
				deltaY = -(*((float*)(v->imageData + y*v->widthStep)+x));
				angle = atan2(deltaY, deltaX);
				hyp = sqrt(deltaX*deltaX + deltaY*deltaY);
				if(hyp > cutoff)
				{
					p1.x = p0.x + cvRound(multiplier*hyp*cos(angle));
					p1.y = p0.y + cvRound(multiplier*hyp*sin(angle));
					cvLine( imgMotion, p0, p1, color,1, CV_AA, 0);

					p0.x = p1.x + cvRound(3*cos(angle-M_PI + M_PI/4));
					p0.y = p1.y + cvRound(3*sin(angle-M_PI + M_PI/4));
					cvLine( imgMotion, p0, p1, color,1, CV_AA, 0);

					p0.x = p1.x + cvRound(3*cos(angle-M_PI - M_PI/4));
					p0.y = p1.y + cvRound(3*sin(angle-M_PI - M_PI/4));
					cvLine( imgMotion, p0, p1, color,1, CV_AA, 0);
				}
			}
		}
		cvShowImage("Motion Field", imgMotion);
		cvWaitKey(0);
		cvReleaseImage(&imgMotion);
	}

	// Draws a color field representation of the flow field
	void drawColorField(IplImage* imgU, IplImage* imgV)
	{
		IplImage* imgColor    = cvCreateImage( cvSize(imgU->width, imgU->height), 8, 3 );
		IplImage* imgColorHSV = cvCreateImage( cvSize(imgColor->width, imgColor->height), IPL_DEPTH_32F, 3 );

		cvZero(imgColorHSV);

		float max_s = 0;
		float *hsv_ptr, *u_ptr, *v_ptr;
		float *color_ptr;
		float angle;
		float h,s,v;
		float r,g,b;
		float deltaX, deltaY;
		int x, y;
		// Generate hsv image
		for(y = 0; y < imgColorHSV->height; y++ )
		{
			hsv_ptr = (float*)(imgColorHSV->imageData + y*imgColorHSV->widthStep);
			u_ptr = (float*)(imgU->imageData + y*imgU->widthStep);
			v_ptr = (float*)(imgV->imageData + y*imgV->widthStep);

			for(x = 0; x < imgColorHSV->width; x++)
			{
				deltaX = u_ptr[x];
				deltaY = v_ptr[x];
				angle = atan2(deltaY,deltaX);

				if(angle < 0)
					angle += 2*M_PI;

				hsv_ptr[3*x] = angle * 180 / M_PI;
				hsv_ptr[3*x+1] = sqrt(deltaX*deltaX + deltaY*deltaY);
				hsv_ptr[3*x+2] = 0.9;

				if(hsv_ptr[3*x+1] > max_s)
					max_s = hsv_ptr[3*x+1];
			}
		}
		// Generate color image
		for(y = 0; y < imgColor->height; y++ )
		{
			hsv_ptr = (float*)(imgColorHSV->imageData + y*imgColorHSV->widthStep);
			color_ptr = (float*)(imgColor->imageData + y*imgColor->widthStep);

			for(x = 0; x < imgColor->width; x++)
			{
				h = hsv_ptr[3*x];
				s = hsv_ptr[3*x+1] / max_s;
				v = hsv_ptr[3*x+2];

				hsv2rgb(h, s, v, r, g, b);

				color_ptr[3*x] = b;
				color_ptr[3*x+1] = g;
				color_ptr[3*x+2] = r;
			}
		}
		drawLegendHSV(imgColor, 25, 28, 28);
		cvShowImage("Color Field", imgColor);
		cvWaitKey(0);
		cvReleaseImage(&imgColorHSV);
	}

	void hsv2rgb(float h, float s, float v, float &r, float &g, float &b)
	{
		if(h > 360)
		{
			h = h - 360;
		}

		float c = v*s;   // chroma
		float hp = h / 60;

		float hpmod2 = hp - (float)((int)(hp/2))*2;

		float x = c*(1 - fabs(hpmod2 - 1));
		float m = v - c;

		float r1, g1, b1;

		if(0 <= hp && hp < 1){
			r1 = c;
			g1 = x;
			b1 = 0;
		}
		else if(1 <= hp && hp < 2){
			r1 = x;
			g1 = c;
			b1 = 0;
		}
		else if(2 <= hp && hp < 3){
			r1 = 0;
			g1 = c;
			b1 = x;
		}
		else if(3 <= hp && hp < 4){
			r1 = 0;
			g1 = x;
			b1 = c;
		}
		else if(4 <= hp && hp < 5){
			r1 = x;
			g1 = 0;
			b1 = c;
		}
		else
		{
			r1 = c;
			g1 = 0;
			b1 = x;
		}
		r = (float)(255*(r1 + m));
		g = (float)(255*(g1 + m));
		b = (float)(255*(b1 + m));
	}

	// Draws the circular legend for the color field, indicating direction and magnitude
	void drawLegendHSV(IplImage* imgColor, int radius, int cx, int cy)
	{
		int width = radius*2 + 1;
		int height = width;

		IplImage* imgLegend = cvCreateImage( cvSize(width, height), 8, 3 );
		IplImage* imgMask = cvCreateImage( cvSize(width, height), 8, 1 );
		IplImage* sub_img = cvCreateImageHeader(cvSize(width, height),8,3);

		float* legend_ptr;
		float angle, h, s, v, legend_max_s;
		float r,g,b;
		int deltaX, deltaY;

		legend_max_s = radius*sqrt(2);

		for(int y=0; y < imgLegend->height; y++)
		{
			legend_ptr = (float*)(imgLegend->imageData + y*imgLegend->widthStep);
			for(int x=0; x < imgLegend->width; x++)
			{
				deltaX = x-radius;
				deltaY = -(y-radius);
				angle = atan2(deltaY,deltaX);

				if(angle < 0)
					angle += 2*M_PI;

				h = angle * 180 / M_PI;
				s = sqrt(deltaX*deltaX + deltaY*deltaY) / legend_max_s;
				v = 0.9;

				hsv2rgb(h, s, v, r, g, b);

				legend_ptr[3*x] = b;
				legend_ptr[3*x+1] = g;
				legend_ptr[3*x+2] = r;
			}
		}
		cvZero(imgMask);
		cvCircle( imgMask, cvPoint(radius,radius) , radius, CV_RGB(255,255,255), -1,8,0 );

		sub_img->origin = imgColor->origin;
		sub_img->widthStep = imgColor->widthStep;
		sub_img->imageData = imgColor->imageData + (cy-radius) * imgColor->widthStep + (cx-radius) * imgColor->nChannels;

		cvCopy(imgLegend, sub_img, imgMask);
		cvCircle( imgColor, cvPoint(cx,cy) , radius, CV_RGB(0,0,0), 1,CV_AA,0 );

		cvReleaseImage(&imgLegend);
		cvReleaseImage(&imgMask);
		cvReleaseImageHeader(&sub_img);
	}

};


#endif /* OF_VISUALIZATION_H_ */
