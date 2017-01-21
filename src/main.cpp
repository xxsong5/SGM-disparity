//

#include"sgm.h"
#include<iostream>

using namespace std;

int main(int argc, char** argv)
{
//	const char* lname = "../data/imLLL.jpg";
//	const char* rname = "../data/imRRR.jpg";

	const char* lname = "../data/imLLL.png";
	const char* rname = "../data/imRRR.png";

	IplImage* l_image = cvLoadImage(lname);
	IplImage* r_image = cvLoadImage(rname);
	IplImage* disparity = cvCreateImage(cvSize(l_image->width, l_image->height), IPL_DEPTH_32F, 1);
	IplImage* disparityr = cvCreateImage(cvSize(r_image->width, r_image->height), IPL_DEPTH_32F, 1);


	//this is the final func
	SGM(l_image, r_image, disparity, disparityr);



	//HMI_SGM(l_image, r_image, disparity, disparityr);
//	int s(0), m(0);
//	ss = clock();
//	RangeDisparity(l_image, r_image, &s, &m);
//	
//	IplImage* l = cvCreateImage(cvSize(l_image->width, l_image->height), IPL_DEPTH_8U, 1);
//	IplImage* r = cvCreateImage(cvSize(r_image->width, r_image->height), IPL_DEPTH_8U, 1);
	
//	cvConvertImage(l_image, l);
//	cvConvertImage(r_image, r);
//
//	int *p = new int[l->width*l->height * (m-s+1)/2];
//	Co_Cost(l, r, (unsigned int*)p, cvSize(l->width, l->height), s, m, 0);
//	Disparity((unsigned int*)p, s, m, disparity, l);
////	unsigned short* sp = (unsigned short*)(p);
////	for (int i = 100*l_image->width*l_image->height / 4; i < 100*l_image->width*l_image->height /2; ++i)
////		cout << sp[i] << "\t";


	cv::Mat disparity_l(disparity,1);
	cv::Mat disparity_r(disparityr,1);
	cv::Mat image_left(l_image);
	disparity_l.convertTo(disparity_l,CV_32FC1,1/60.0);
	disparity_r.convertTo(disparity_r,CV_32FC1,1/60.0);
	cv::imshow("disparity_left",disparity_l);
	cv::imshow("left_image",image_left);
	cv::imshow("disparity_right",disparity_r);
	cv::waitKey(0);

	return 0;
}

