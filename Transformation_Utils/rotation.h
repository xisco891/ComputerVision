/*
 * rotation.h
 *
 *  Created on: Jul 31, 2013
 *      Author: hossein
 */
#include <eigen3/Eigen/Dense>
using namespace Eigen;

static const double pi = 3.14159265358979323846;

void extractAngles(Matrix3f R, Vector3f& omega)
{
	/*  omega(1)=asin(R(2,0));
   omega(0)=atan2(R(2,1)/cos(omega(1)),R(2,2)/cos(omega(1)));
   omega(2)=atan2(R(1,0)/cos(omega(1)),R(0,0)/cos(omega(1)));*/
	omega(1)=asin(R(2,0));
	if (R(2,2)<0)
	{
		if (omega(1)>0)
			omega(1)=pi-omega(1);
		else
			omega(1)=-pi-omega(1);
	}
	omega(0)=atan2(R(2,1)/cos(omega(1)),R(2,2)/cos(omega(1)));
	omega(2)=atan2(R(1,0)/cos(omega(1)),R(0,0)/cos(omega(1)));
	return;
}
//----------------------------------------
void makeRotation(Matrix3f& R, Vector3f omega)
{
	float ux=omega(0);
	float uy=omega(1);
	float uz=omega(2);
	Matrix3f Rx; Rx<< 1, 0,0,
					  0, cos(ux), -sin(ux),
					  0, sin(ux), cos(ux);

	Matrix3f Ry; Ry<< cos(uy), 0, -sin(uy),
					  0, 1, 0,
					  sin(uy), 0, cos(uy);

	Matrix3f Rz; Rz<< cos(uz), -sin(uz), 0,
					  sin(uz), cos(uz), 0,
					  0, 0, 1;
	R=Rz*Ry*Rx;
}
//----------------------------------------
void makeRotation(Matrix3f& R, 	float ux, float uy, float uz)
{
	Matrix3f Rx; Rx<< 1, 0,0,
					 0, cos(ux), -sin(ux),
					 0, sin(ux), cos(ux);
	Matrix3f Ry; Ry<< cos(uy), 0, -sin(uy),
	                  0, 1, 0,
	                  sin(uy), 0, cos(uy);
	Matrix3f Rz; Rz<< cos(uz), -sin(uz), 0,
					  sin(uz), cos(uz), 0,
					  0, 0, 1;
	R=Rz*Ry*Rx;
}

//----------------------------------------------------
void rotationdx(Matrix3f& R,float ux,float uy, float uz)
{
	Matrix3f Rx; Rx<<0, 0, 0,0, -sin(ux), -cos(ux),0, cos(ux), -sin(ux);
	Matrix3f Ry;Ry<< cos(uy), 0, -sin(uy),0, 1, 0, sin(uy), 0, cos(uy);
	Matrix3f Rz;Rz<<cos(uz), -sin(uz), 0,sin(uz), cos(uz), 0,0, 0, 1;
	R=Rz*Ry*Rx;
}
//----------------------------------------------------
void rotationdy(Matrix3f& R,float ux,float uy, float uz)
{
	Matrix3f Rx;Rx<<1, 0, 0,0, cos(ux), -sin(ux),0, sin(ux), cos(ux);
	Matrix3f Ry;Ry<<-sin(uy), 0, -cos(uy),0, 0, 0, cos(uy), 0, -sin(uy);
	Matrix3f Rz;Rz<<cos(uz), -sin(uz), 0,sin(uz), cos(uz), 0,0, 0, 1;
	R=Rz*Ry*Rx;
}
//----------------------------------------------------
void  rotationdz(Matrix3f& R,float ux,float uy, float uz)
{
	Matrix3f Rx;Rx<<1, 0, 0,0, cos(ux), -sin(ux),0, sin(ux), cos(ux);
	Matrix3f Ry;Ry<<cos(uy), 0,- sin(uy),0, 1, 0, sin(uy), 0, cos(uy);
	Matrix3f Rz;Rz<<-sin(uz), -cos(uz), 0,cos(uz), -sin(uz), 0,0, 0,0;
	R=Rz*Ry*Rx;
}

void rottoQuat(Matrix3f R,VectorXf& q)
{

	float trc=R(0,0)+R(1,1)+R(2,2);
	    float ctt=(trc-1)/2;
	    if (ctt>=1)
	    {
	        q<<1,0,0,0;
	    	return;
	    }

	    float tt=acos(ctt);
	    cout<<"tt: "<<tt<<endl;
	    float ux=(R(0,0)-ctt)/(1-ctt);

	    if (ux>=0)
	    	ux=sqrt(ux);
	    else
	    	ux=0;


	    float a=ux*(1-ctt);
	    float b=sin(tt);
	    float g=R(1,0);
	    float l=R(2,0);

	    float uz=(b*g+a*l)/(a*a+b*b);
	    float uy=0;
	    if (a!=0)
	        uy=(g-b*uz)/a;

	    float r32=uz*uy*(1-ctt)+ux*sin(tt);
	    if (abs(r32- R(2,1))>.001)
	    {
	        tt=-tt;
	        b=sin(tt);
	        g=R(1,0);
	        l=R(2,0);
	        uz=(b*g+a*l)/(a*a+b*b);
	        if (a!=0)
	            uy=(g-b*uz)/a;
	        else
	            uy=0;
	    }


	    if (ux==0 && uy==0 && uz==0)
	        uy=-R(2,0)/sin(tt);
	    q<<cos(tt/2),ux*sin(tt/2),uy*sin(tt/2),uz*sin(tt/2);
}
//------------------------
void quattoRot(Matrix3f& R,VectorXf q)
{
	R<<1-2*(q(2)*q(2)+q(3)*q(3)),     2*(q(1)*q(2)-q(0)*q(3)), 2*(q(0)*q(2)+q(1)*q(3)),
			2*(q(1)*q(2)+q(0)*q(3)), 1-2*(q(1)*q(1)+q(3)*q(3))    , 2*(q(2)*q(3)-q(0)*q(1)),
			2*(q(1)*q(3)-q(0)*q(2)), 2*(q(0)*q(1)+q(2)*q(3)), 1-2*(q(1)*q(1)+q(2)*q(2));
}
//------------------------
void qmul(VectorXf a,VectorXf b,VectorXf& q )
{
	  q<<a(0)*b(0)-a(1)*b(1)-a(2)*b(2)-a(3)*b(3),
	       a(0)*b(1)+a(1)*b(0)+a(2)*b(3)-a(3)*b(2),
	       a(0)*b(2)+a(2)*b(0)+a(3)*b(1)-a(1)*b(3),
	       a(0)*b(3)+a(3)*b(0)+a(1)*b(2)-a(2)*b(1);
	  q.normalize();
}

void conj(VectorXf q,VectorXf& qc )
{
	qc<<q(1),-q(2),-q(3),-q(4);
}
