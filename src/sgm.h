#pragma once
#include<cv.h>
#include<highgui.h>
#include<math.h>
#include<x86intrin.h>
#include<omp.h>
#include<iostream>
using namespace std;
#define AD_STRUCTURE 0
#define RANK_STRUCTURE 1
struct Point_Int
{
	int a_;
	int b_;
};

void RangeDisparity(const IplImage* L_image, const IplImage* R_image, int* min_dis, int* max_dis);

void Init_Cost(unsigned int* cost, CvSize size, int min_dis, int max_dis, int data);

void Co_Cost(const IplImage* L_image,const IplImage* R_image,unsigned int* cost,CvSize size,int min_dis,int max_dis,int Tag_L0_R1,int kernal);//tag_L0_r1 L:0,R:1 AD-Sructure

void RS_Cost(const IplImage* L_image, const IplImage* R_image, unsigned int* cost, CvSize size, int min_dis, int max_dis, int Tag_L0_R1,unsigned int kernal);

void Disparity(unsigned int* cost, int min_dis, int max_dis, IplImage* disparity, IplImage* L_Rimage, unsigned short P1=9, unsigned short MAX_=20);

void Check(IplImage* L_disparity, IplImage* R_disparity,int do_r_dis=0);//ze dang wu pi pei bu tong invalid//要不要检测右图

void Interpolation(IplImage* disparity);

void PostProcessing();

void SGM(const IplImage* l_image, const IplImage* r_image, IplImage* disparity, IplImage* R_disparity = NULL, int type = AD_STRUCTURE,int Post_Process = 0);

void HMI_SGM(const IplImage* l_image, const IplImage* r_image, IplImage* disparity, IplImage* R_disparity = NULL, int Post_Process = 0);




//common
void Acc_4(const unsigned int* src, unsigned int* dest, int counts);				//快速累加;
int min_index_common_4_loop(const unsigned short* data, int counts, unsigned short* result = NULL);
void Value_P2(const IplImage* image, IplImage* P2, int P1, int MAX_);//P1+MAX_ largest;
void Peak(IplImage* disprity, int con_point_one_dir, int data);
float Max_float_avx_4loop(const float* data, int counts);
Point_Int Statistics(const IplImage* disprity, float proprotion, int min_disprity, int max_disprity);
void Warp_(const IplImage* disprity, const IplImage* l_image, const IplImage* r_image, IplImage* warp);
