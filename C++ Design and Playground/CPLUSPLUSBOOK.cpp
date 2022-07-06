#include <iostream>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
//#include "opencv2/cudalegacy.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudastereo.hpp"
#include "opencv2/cudawarping.hpp"
//#include "opencv2/cudev.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

//FOR A FUNCTION TO BE USABLE IN A CONSTANT EXPRESSION

constexpr double square(double x){return x*x;}

bool accept(){

	cout<<"Do you want to proceed(y or n)?\n";
	char answer=0;
	cin >> answer; // Standard input stream

	if(answer=='y') return true;
	return false;


	switch(answer){
		case 'y':return true;
		case 'n':return false;
		default: cout<<"Sorry, I don't understand that.\n"<<endl;
				 return false;
	}

}

bool accept3(){
	int tries=1;
	while(tries<4){
		cout<<"Do you want to proceed:(y or n)?\n";
		char answer=0;
		cin >> answer;
		switch(answer){
			case 'y':return true;
			case 'n':return false;
			default:cout<<"Sorry, I don't understand that.\n";
			++tries;
		}
		cout<<"I'll take that for a no.\n";
		return false;
	}
}

int count_x(char* p, char x){

	//Count the number of occurrences of x in p[]
	//p is assumed to point to a zero-terminated array of char (or to nothing)
	if(p==nullptr) return 0;
	int count=0;

	for(; *p!=0;++p){
		if(*p==x)
			++count;
	return count;
	}

}


int main(){

//INITIALIZE A VECTOR
vector<int> v{1,2,3,4,5};
vector<int> v1= (vector<int> << 1,2,3,4,5);

//CONSTANTS
//const:"I promise to not change this value."
//constexpr:"to be evaluated at compile time"
const int dmv=17;
int var=17;
constexpr double max1=1.4*square(dmv);
constexpr double max2=1.4*square(var);
const double max3=1.4*square(var);

//POINTERS AND REFERENCE

double* pd= nullptr;
Link<Record>* Ist=nullptr;
int x = nullptr;

for(int i=0;i<v.size();i++){
		cout<<"v["<<i<<"]:"<<v[i]<<endl;
	}
	cout<<""<<endl;
	for(int j=0;j<v1.size();j++){
		cout<<"v1["<<j<<"]:"<<v[j]<<endl;
	}

}
