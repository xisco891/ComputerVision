/*
 * sevenpoint.h
 *
 *  Created on: Jun 17, 2014
 *      Author: hossein
 */

#ifndef SEVENPOINT_H_
#define SEVENPOINT_H_

#include <stdlib.h>
#include <stdio.h>

#include <cstdio>

#include <cmath>
#include <iostream>
#include <fstream>
using std::ofstream;
#include <cstdlib>
using namespace std;
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv2/video/tracking.hpp>
using namespace cv;
#include <Eigen/Dense>
using namespace Eigen;

inline static int OpticalC(const Mat& frame1, const Mat& frame2, vector<vector<Point2f> >& frame_features, int ws, float trh)
{
	int width=frame1.size().width;
	int height=frame1.size().height;

	Mat frame1_g,frame2_g;
	if (frame1.type()>0)
		cvtColor(frame1, frame1_g, CV_BGR2GRAY);
	else
		frame1_g=frame1;
	if(frame2.type()>0)
		cvtColor(frame2, frame2_g, CV_BGR2GRAY);
	else
		frame2_g=frame2;


	Size frame_size;
	frame_size.width=width;
	frame_size.height=height;

	Mat mask;
	//	int number_of_features=f_n;
	goodFeaturesToTrack(frame1_g,frame_features[0],500,trh,10,mask,3,false);


	cout<<"number_of_features: "<<frame_features[0].size()<<endl;


	Size optical_flow_window = cvSize(ws,ws);

	CvTermCriteria optical_flow_termination_criteria = cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3 );

	vector< unsigned char> status;
	Mat err,flow;
	calcOpticalFlowPyrLK(frame1_g,frame2_g,frame_features[0],frame_features[1],status,err,Size(11,11),4,
			optical_flow_termination_criteria);

	vector<Point2f> tempfeatures;
	Mat status1;
	calcOpticalFlowPyrLK(frame2_g,frame1_g,frame_features[1],tempfeatures,status1,err,Size(11,11),4,
			optical_flow_termination_criteria);

	int wfn=0;

	int feature_count=0;
	for(int i = 0; i < frame_features[0].size(); i++)
	{

		if ( status[i] == 0	)
		{
			status[i] = 0;
			frame_features[1][i].x=-10;
			frame_features[1][i].y=-10;
			//cout<<"f number:  "<<i<<"   "<<map.size()<<endl;
			wfn++;
			//fprintf(stderr, "not found point: %f\n",frame_features[k+1][i].x);
		}
		else
		{
			float dx=abs(tempfeatures[i].x-frame_features[0][i].x);
			float dy=abs(tempfeatures[i].y-frame_features[0][i].y);
			if (dx+dy<3)
				feature_count++;
			else
			{
				frame_features[1][i].x=-10;
				frame_features[1][i].y=-10;
			}

		}
	}

	return feature_count;

}


class Motion{
public:
	Matrix3d Ff;
	Matrix3d Ef;
	Matrix3d R;
	Vector3d T;
	Mat Ac,Bc,Cc;
	vector<double> distances;
	Point2f epi1,epi2;
	Motion(vector<vector<Point2f> >frame_features, Matrix3d camin, int rin,int wd, int hg);
	Motion(Mat I1, Mat I2, Matrix3d camin, int rin);
	void motionRecovery();
	void normalizePoints();
	void findDistances();
	void findLine(Point2f& p1, Point2f& p2, int ix);
	void findFoundamental(vector<Matrix3d>& E);
	void distanceToepiLines(vector<Matrix3d>F1, VectorXd& dist);
	void findFlow(const Mat& frame, Mat& u, Mat& v, double depth);
	void drawFlow(Mat frame, Mat& u, Mat& v);
	void calcLineImages();
	Mat testPyramid(int level);
	void findEpipoles();

private:
	Matrix3d Tr1,Tr2;
	vector<int> rinx;
	int rn;
	Matrix3d cam,cami;
	vector<double> x,y,xp,yp;
	vector<Vector3d> xy,xyp;
	int fsize;
	vector<vector<Point2f> > features;
	int width;
	int height;
};

void Motion::findEpipoles()
{
	Vector3d cof;
	Vector3d X1;X1<<0,0,1;
	Vector3d X2;X2<<100,100,1;
	Vector3d cof1=Ff*X1;
	Vector3d cof2=Ff*X2;
	Matrix2d A;
	A<<cof1(0),cof1(1),
	  cof2(0),cof2(1);
	Vector2d b,c;
	b<<-cof1(2),-cof2(2);
	c=A.inverse()*b;
	epi1=Point2f(c(0),c(1));

	cof1=Ff.transpose()*X1;
	cof2=Ff.transpose()*X2;
	A<<cof1(0),cof1(1),
	  cof2(0),cof2(1);
	b<<-cof1(2),-cof2(2);
	c=A.inverse()*b;
	epi2=Point2f(c(0),c(1));



}

Mat Motion::testPyramid(int level)
{
	Mat dimg=Mat::zeros(Size(width, height),CV_8UC3);
	for (int i=1;i<level;i++)
	{
		for(int x=0;x<dimg.cols;x+=10)
			for(int y=0;y<dimg.rows;y+=10)
			{
				Vector3d cofs;
				Vector3d X1;X1<<x,y,1;
				cofs=Ff*X1;
				Point2f p1,p2;
				if (abs(cofs(0))>abs(cofs(1)))
				{
					double m,b;
					m=-cofs(1)/cofs(0);
					b=-cofs(2)/cofs(0);
					p1.x=b/(double)i;
					p1.y=0;

					p2.y=dimg.rows;
					p2.x=m*p2.y+b/pow(2,i-1);
				}
				else
				{
					double m,b;
					m=-cofs(0)/cofs(1);
					b=-cofs(2)/cofs(1);
					p1.y=b/(double)i;
					p1.x=0;

					p2.x=dimg.cols;
					p2.y=m*p2.x+b/pow(2,i-1);
				}
				line(dimg,p1,p2,Scalar(0,0,255),1);
			}
		imshow("frlows",dimg);
		waitKey();
		resize(dimg,dimg,Size(width/(1<<i),height/(1<<i)));
		dimg*=0;
	}
	return dimg;
}
void Motion::calcLineImages()
{
	Ac=Mat::zeros(Size(width,height),CV_32F);
	Bc=Ac.clone();
	Cc=Ac.clone();
	float * Ad=(float*)Ac.data;
	float * Bd=(float*)Bc.data;
	float * Cd=(float*)Cc.data;
	for(int x=0;x<width;x++)
		for(int y=0;y<height;y++)
		{
			Vector3d cof;
			Vector3d X1;X1<<x,y,1;
			cof=Ff*X1;
			//			cout<<"cof= "<<cof.transpose()<<endl;
			Ad[x+y*width]=cof(0);
			Bd[x+y*width]=cof(1);
			Cd[x+y*width]=cof(0)*x+cof(1)*y+cof(2);
		}
}
Motion::Motion(Mat I1, Mat I2, Matrix3d camin, int rin)
{
	cam=camin;
	cami=camin.inverse();
	rn=rin;
	xy.resize(rn);
	xyp.resize(rn);
	width=I1.cols;
	height=I1.rows;
	features.resize(2);
	int nmatch= OpticalC(I1,I2,features,13,.03);

	if (nmatch<30)
	{
		nmatch= OpticalC(I1,I2,features,13,.01);
	}

	fsize=features[0].size();
	//	cout<<"nmatch= "<<nmatch<<endl;
	double dx=0,dy=0;
	for(int j=0;j<features[0].size();j++)
		if(features[1][j].y>0 && features[1][j].y<height &&
				features[1][j].x>0 && features[1][j].x<width &&
				features[0][j].y>0 && features[0][j].y<height &&
				features[0][j].x>0 && features[0][j].x<width)
		{
			x.push_back(features[0][j].x);
			y.push_back(features[0][j].y);

			xp.push_back(features[1][j].x);
			yp.push_back(features[1][j].y);
			dx+=abs(features[1][j].x-features[0][j].x);
			dy+=abs(features[1][j].y-features[0][j].y);
		}

	dx/=(double)x.size();
	dy/=(double)y.size();
	cout<<"dx="<<dx<<endl;
	cout<<"dy="<<dy<<endl;
	//	if(dx+dy<3)
	//	{
	//		Ff.fill(0);
	//		Ef.fill(0);
	//		R=Matrix3d::Identity();
	//		T.fill(0);
	//		Ac=Mat::zeros(Size(width,height),CV_32F);
	//		Bc=Ac.clone();
	//		Cc=Ac.clone();
	//		return;
	//	}

	motionRecovery();
	Ff*=1000;
	calcLineImages();
	findEpipoles();
	cout<<"Ff= "<<endl<<Ff<<endl;
	cout<<"R= "<<endl<<R<<endl;
	cout<<"T= "<<endl<<T<<endl;
	cout<<"epi1="<<epi1<<endl;
	cout<<"epi2="<<epi2<<endl;
//	imshow("Ac",Ac);
//	waitKey(1);
//	getchar();
}

void Motion::findFlow(const Mat& frame, Mat& u, Mat& v, double depth)
{
	u=Mat::zeros(Size(frame.cols,frame.rows),CV_32F);
	v=Mat::zeros(Size(frame.cols,frame.rows),CV_32F);
	float * udata=(float*) u.data;
	float * vdata=(float*) v.data;
	int w=u.cols;
	cout<<"R= "<<endl<<R<<endl;
	cout<<"T= "<<endl<<T<<endl;
	cout<<"cam= "<<endl<<cam<<endl;
	cout<<"cami= "<<endl<<cami<<endl;
	for(int x=0;x<frame.cols;x++)
		for(int y=0;y<frame.rows;y++)
		{
			static Vector3d X1;X1<<x,y,1;
			X1=cami*X1;
			X1.normalize();
			static Vector3d X2;
			X2=R*(depth*X1-T);
			X2/=X2(2);
			X2=cam*X2;
			udata[x+w*y]=X2(0)-x;
			vdata[x+w*y]=X2(1)-y;
		}
}
void Motion::drawFlow(Mat frame, Mat& u, Mat& v)
{
	Mat tframe=frame.clone();
	for(int x=0;x<u.cols;x+=20)
		for(int y=0;y<u.rows;y+=20)
		{
			Point p1=Point(x,y);
			Point p2=Point(x+(int)u.at<float>(y,x),y+(int)v.at<float>(y,x));
			line(tframe,p1,p2,Scalar(0,0,255),1);
			circle(tframe,p2,3,Scalar(0,255.0),1);
		}
	imshow("frlows",tframe);
	waitKey(10);
}
void Motion::findDistances()
{

	distances.assign(features[0].size(),-100);
	Vector3d X1,X2;
	for(int j=0;j<features[0].size();j++)
	{
		if(features[1][j].y<0 || features[1][j].y>height ||
				features[1][j].x<0 || features[1][j].x>width ||
				features[0][j].y<0 || features[0][j].y>height ||
				features[0][j].x<0 || features[0][j].x>width)
			continue;

		double x1=features[0][j].x;
		double y1=features[0][j].y;

		double x2=features[1][j].x;
		double y2=features[1][j].y;

		X1<<x1,y1,1;
		X2<<x2,y2,1;

		Vector3d  cofs=Ff*X1;
		double dist1=abs((X2.transpose()*cofs)(0));
		dist1/=sqrt((cofs.head(2).transpose()*cofs.head(2))(0));

		cofs=Ff.transpose()*X2;
		double dist2=abs((X2.transpose()*cofs)(0));
		dist2/=sqrt((cofs.head(2).transpose()*cofs.head(2))(0));

		distances[j]=(dist1+dist2)/2;
	}
}

void Motion::distanceToepiLines(vector<Matrix3d> F1, VectorXd& dist)
{

	dist.resize(F1.size());
	Vector3d X1,X2;
	for(int i=0;i<F1.size();i++)
	{
		for(int j=0;j<rinx.size();j++)
		{
			double x1=features[0][rinx[j]].x;
			double y1=features[0][rinx[j]].y;

			double x2=features[1][rinx[j]].x;
			double y2=features[1][rinx[j]].y;

			X1<<x1,y1,1;
			X2<<x2,y2,1;

			Vector3d  cofs=Ff*X1;
			double dist1=abs((X2.transpose()*cofs)(0));
			dist1/=sqrt((cofs.head(2).transpose()*cofs.head(2))(0));

			cofs=Ff.transpose()*X2;
			double dist2=abs((X2.transpose()*cofs)(0));
			dist2/=sqrt((cofs.head(2).transpose()*cofs.head(2))(0));
			dist(i)+=(dist1+dist2)/2;
		}

	}
}
Motion::Motion(vector<vector<Point2f> >frame_features, Matrix3d camin, int rin, int wd, int hg)
{

	cam=camin;
	cami=camin.inverse();
	rn=rin;
	xy.resize(rn);
	xyp.resize(rn);
	fsize=frame_features[0].size();
	features=frame_features;
	width=wd;
	height=hg;

	for(int j=0;j<frame_features[0].size();j++)
		if(frame_features[1][j].y>0 && frame_features[1][j].y<height &&
				frame_features[1][j].x>0 && frame_features[1][j].x<width &&
				frame_features[0][j].y>0 && frame_features[0][j].y<height &&
				frame_features[0][j].x>0 && frame_features[0][j].x<width)
		{
			x.push_back(frame_features[0][j].x);
			y.push_back(frame_features[0][j].y);

			xp.push_back(frame_features[1][j].x);
			yp.push_back(frame_features[1][j].y);
		}
}

void Motion::normalizePoints()
{
	for(int j=0;j<rn;j++)
	{
		xy[j]<<x[rinx[j]], y[rinx[j]],1;
		xyp[j]<<xp[rinx[j]], yp[rinx[j]],1;

	}

	Vector3d cent1,cent2;
	cent1<<0,0,0;
	cent2<<0,0,0;
	for(int j=0;j<rn;j++)
	{
		cent1+=xy[j];
		cent2+=xyp[j];
	}

	cent1/=(double)xy.size();
	cent2/=(double)xyp.size();

	for(int j=0;j<rn;j++)
	{
		xy[j]<<xy[j](0)-cent1(0),xy[j](1)-cent1(1),1;
		xyp[j]<<xyp[j](0)-cent2(0),xyp[j](1)-cent2(1),1;
	}

	double distance1,distance2;
	distance1=0;
	distance2=0;
	for(int j=0;j<rn;j++)
	{
		distance1+=sqrt((xy[j].head(2).transpose()*xy[j].head(2))(0));
		distance2+=sqrt((xyp[j].head(2).transpose()*xyp[j].head(2))(0));
	}

	distance1/=(double)rn;
	distance2/=(double)rn;
	double scale1=sqrt(2)/distance1;
	double scale2=sqrt(2)/distance2;
	for(int j=0;j<rn;j++)
	{
		xy[j]<<scale1*xy[j](0),scale1*xy[j](1),1;
		xyp[j]<<scale2*xyp[j](0),scale2*xyp[j](1),1;
		//		cout<<"xy normalize= "<<xy[j].transpose()<<",  "<<xyp[j].transpose()<<endl;

	}



	Tr1<<scale1,0    ,-scale1*cent1(0),
			0,   scale1,-scale1*cent1(1),
			0,0,1;

	Tr2<<scale2,0    ,-scale2*cent2(0),
			0,   scale2,-scale2*cent2(1),
			0,0,1;
}

void Motion::findFoundamental(vector<Matrix3d>& E)
{
	MatrixXd a;
	a.resize(rn,9);
	E.clear();

	for(int j=0;j<rn;j++)
	{

		normalizePoints();
		double x1=xy[j](0);
		double y1=xy[j](1);

		double x2=xyp[j](0);
		double y2=xyp[j](1);
		a.row(j)<<x2*x1,x2*y1,x2,y2*x1,y2*y1,y2,x1,y1,1;
	}

	JacobiSVD<MatrixXd> svd(a, ComputeThinU | ComputeThinV);

	VectorXd sigv=svd.singularValues();

	MatrixXd u= svd.matrixU();
	MatrixXd v= svd.matrixV();
	MatrixXd sig=sigv.asDiagonal();

	MatrixXd c=v.col(8);

	VectorXd e,f,ef(9);
	e=v.col(7);
	f=v.col(8);

	double c3= e(0)*e(4)*e(8) - e(0)*e(5)*e(7) - e(1)*e(3)*e(8) + e(1)*e(5)*e(6) + e(2)*e(3)*e(7) - e(2)*e(4)*e(6);

	double c2=e(0)*e(4)*f(8) - e(0)*e(5)*f(7) - e(0)*e(7)*f(5) + e(0)*e(8)*f(4) - e(1)*e(3)*f(8) + e(1)*e(5)*f(6) +
			e(1)*e(6)*f(5) - e(1)*e(8)*f(3) + e(2)*e(3)*f(7) - e(2)*e(4)*f(6) - e(2)*e(6)*f(4) + e(2)*e(7)*f(3) +
			e(3)*e(7)*f(2) - e(3)*e(8)*f(1) - e(4)*e(6)*f(2) + e(4)*e(8)*f(0) + e(5)*e(6)*f(1) - e(5)*e(7)*f(0);

	double c1=e(0)*f(4)*f(8) - e(0)*f(5)*f(7) - e(1)*f(3)*f(8) + e(1)*f(5)*f(6) + e(2)*f(3)*f(7) - e(2)*f(4)*f(6) -
			e(3)*f(1)*f(8) + e(3)*f(2)*f(7) + e(4)*f(0)*f(8) - e(4)*f(2)*f(6) - e(5)*f(0)*f(7) + e(5)*f(1)*f(6) +
			e(6)*f(1)*f(5) - e(6)*f(2)*f(4) - e(7)*f(0)*f(5) + e(7)*f(2)*f(3) + e(8)*f(0)*f(4) - e(8)*f(1)*f(3);

	double c0=f(0)*f(4)*f(8) - f(0)*f(5)*f(7) - f(1)*f(3)*f(8) + f(1)*f(5)*f(6) + f(2)*f(3)*f(7) - f(2)*f(4)*f(6);

	c2=c2/c3; c1=c1/c3;c0=c0/c3;
	Matrix3cf poly;
	poly.fill(0);
	poly.real()<<0,    1,   0,
			0,    0,   1,
			-c0, -c1, -c2;

	ComplexEigenSolver<Matrix3cf> ces;
	ces.compute(poly);
	Vector3cf egs=ces.eigenvalues();

	VectorXd eft(9);
	int rroot=0;
	for (unsigned int r=0;r<3;r++)
	{
		if (abs(egs(r).imag())>.001)
			continue;
		eft=egs(r).real()*e+f;
		eft.normalize();
		Matrix3d Et;Et <<eft(0),eft(1),eft(2),eft(3),eft(4),eft(5),eft(6),eft(7),eft(8);
		rroot++;
		Et=Tr2.transpose()*Et*Tr1;
		//		Et.normalize();
		E.push_back(Et);
	}
	if (rroot==0)
	{
		Matrix3d Et;Et <<f(0),f(1),f(2),f(3),f(4),f(5),f(6),f(7),f(8);
		Et=Tr2.transpose()*Et*Tr1;
		//		Et.normalize();
		E.push_back(Et);
	}
}


void Motion::motionRecovery()
{

	if (x.size()<10)
	{
		Ef=Matrix3d::Identity();
		Ff=Matrix3d::Identity();
		return;
	}
	//	cout<<"sizes= "<<x.size()<<". "<<y.size()<<". "<<xp.size()<<". "<<yp.size()<<endl;

	int fn=x.size();
	int itn=70;
	itn=min(1*fn/rn,itn);

	//	MatrixXd dy(f_n,itn);
	//	MatrixXd dx(f_n,itn);
	vector<Matrix3d> E;

	Vector3d X1,X2;
	vector<int> ransv(fn,0);
	for(int it=0;it<itn;it++)
	{

		rinx.assign(rn,0);
		int k=0;
		while(k<rn)
		{
			int rt=rand()%fn;
			if (ransv[rt]>=0)
			{
				rinx[k]=rt;
				ransv[rt]=-1;
				k++;
			}
		}


		MatrixXd a;
		a.resize(rn,9);
		//		double cf=1;

		normalizePoints();
		vector<Matrix3d> F1;
		findFoundamental(F1);
		for (int j=0;j<F1.size();j++)
			E.push_back(F1[j]);

	}


	VectorXd err(E.size());
	err.fill(0);
	for (int i=0;i<E.size();i++)
		for(int j=0;j<fn;j++)
		{
			double y1=y[j];
			double y2=yp[j];
			double x1=x[j];
			double x2=xp[j];
			X1<<x1,y1,1;
			X2<<x2,y2,1;

			Vector3d  cofs=E[i]*X1;
			double dist1=abs((X2.transpose()*cofs)(0));
			dist1/=sqrt((cofs.head(2).transpose()*cofs.head(2))(0));

			cofs=E[i].transpose()*X2;
			double dist2=abs((X2.transpose()*cofs)(0));
			dist2/=sqrt((cofs.head(2).transpose()*cofs.head(2))(0));

			err(i)+=dist1+dist2;
		}

	int rw,cl;
	err.minCoeff(&rw,&cl);
	Ff=E[rw];
	//	cout<<"det(F)= "<<Ff.determinant()<<endl;
	//	findDistances();
	//	rinx.clear();
	//	for (int i=0;i<distances.size();i++)
	//	{
	//		if(abs(distances[i])<3)
	//			rinx.push_back(i);
	//	}
	//	normalizePoints();
	//	vector<Matrix3d> F1;
	//	findFoundamental(F1);
	//	VectorXd dist;
	//	distanceToepiLines(F1,dist);

	//	dist.minCoeff(&rw,&cl);
	//	Ff=F1[rw];
	for(int i=0;i<itn;i++)
	{
		err.minCoeff(&rw,&cl);
		Ef=E[rw];
		//	cout<<"F= "<<Ef<<endl;
		Ef=cam.transpose()*Ef*cam;

		//	cout<<"E= "<<Ef<<endl;
		//	cvWaitKey();

		JacobiSVD<MatrixXd> svd(Ef, ComputeThinU | ComputeThinV);
		Vector3d sigv=svd.singularValues();
		sigv(2)=0;
		MatrixXd u= svd.matrixU();
		MatrixXd v= svd.matrixV();
		MatrixXd sig=sigv.asDiagonal();
		Matrix3d w;
		w << 0, -1, 0,
				1, 0, 0,
				0, 0, 1;

		MatrixXd Ra=u*w*v.transpose();
		Ra=Ra.determinant()*Ra;
		MatrixXd Rb=u*w.transpose()*v.transpose();
		Rb=Rb.determinant()*Rb;
		cout<<"sig: "<<endl;
		cout<<sig<<endl;

		cout<<"Ra: "<<endl;
		cout<<Ra<<endl;

		cout<<"Rb: "<<endl;
		cout<<Rb<<endl;

		Matrix3d z;
		z << 0, -1, 0,
				1, 0, 0,
				0, 0, 0;


		MatrixXd Tx=v*z.transpose()*v.transpose();//v*z*v.transpose();
		//cout<<Tx<<endl;
		cout<<"T: "<<T<<endl;
		T(0)=Tx(1,2);
		T(1)=Tx(2,0);
		T(2)=Tx(0,1);
		cout<<"Tx1: "<<endl;
		if(T(2)<0)
			T=-T;

		if(Ra(0,0)>0.1 && Ra(2,2)>0.1)
		{
			R=Ra;
			return;
		}
		else if (Rb(0,0)>0.1 && Rb(2,2)>0.1)
		{
			R=Rb;
			return;
		}
		else
		{
			cout<<"Ra: "<<endl;
			cout<<Ra<<endl;

			cout<<"Rb: "<<endl;
			cout<<Rb<<endl;
			cout<<"hey wieder----"<<endl;

			cout<<err<<endl<<endl;
			err(rw,0)=10000;
			cout<<err<<endl;
			//	cvWaitKey();

		}
	}
	T<<0,0,1;
	R= Matrix3d::Identity();
	Ef= Matrix3d::Identity();
	Ff= Matrix3d::Identity();
	return;
}

void Motion::findLine(Point2f& p1, Point2f& p2, int inx)
{
	Vector3d cofs;
	Vector3d X1;X1<<features[0][inx].x,features[0][inx].y,1;
	cofs=Ff*X1;
	if (abs(cofs(0))>abs(cofs(1)))
	{
		p1.x=-cofs(2)/cofs(0);
		p1.y=0;

		p2.y=height;
		p2.x=-(int)(p2.y*cofs(1)+cofs(2))/cofs(0);
	}
	else
	{
		p1.x=0;
		p1.y=-cofs(2)/cofs(1);

		p2.x=width;
		p2.y=-(p2.x*cofs(0)+cofs(2))/cofs(1);
	}


}



#endif /* SEVENPOINT_H_ */
