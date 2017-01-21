#include"sgm.h"


int table[256];
void Init_Cost(unsigned int* cost, CvSize size, int min_dis, int max_dis,int data)
{
	int count = size.height*size.width*(max_dis-min_dis+1)/2;
	int loop = count / 4;
	int left = count % 4;
	unsigned int* cp0(cost), *cp1(cost + 1), *cp2(cost + 2), *cp3(cost + 3);
	for (int i = 0; i < loop; ++i)
	{
		*cp0 = *cp1 = *cp2 = *cp3 = data;
		cp0 += 4;
		cp1 += 4;
		cp2 += 4;
		cp3 += 4;
	}
	for (int i = 0; i < left; ++i)
	{
		cp0[i] = data;
	}
}
void Co_Cost(const IplImage* L_image, const IplImage* R_image, unsigned int* cost, CvSize size, int min_dis, int max_dis, int Tag_L0_R1,int kernel)//mix_dis-min_dis+1 Îª8»ò4µÄÕûÊý±¶
{
	//set table
	for (int i = 0; i != 256; ++i)
	{
		table[i] = i>>1 ;
	}
	//init
	Init_Cost(cost, size, min_dis, max_dis, 8323199);
	const int counts = max_dis - min_dis + 1;
	const int  next = sizeof(short)*counts;
	const int costwidthstep = next*size.width;
	unsigned short* spcost = (unsigned short*)cost;
	int Height, Width;
	unsigned char* cp(NULL), *cp1(NULL), *rcp, *rcp1, *rcp_(NULL), *rcp1_(NULL), *rcp__(NULL), *rcp1__(NULL), *_rcp(NULL), *_rcp1(NULL), *__rcp(NULL), *__rcp1(NULL);
	int thisd(1);
	int imageStep,imageStep1;
	int maxx;
	if (Tag_L0_R1)
	{
		Height = R_image->height;
		Width = R_image->width;
		cp= (unsigned char*)R_image->imageData;
		cp1=(unsigned char*)L_image->imageData;
		thisd = 1;
		imageStep = R_image->widthStep;
		imageStep1 = L_image->widthStep;
		maxx = L_image->width;
	}
	else
	{
		Height = L_image->height;
		Width = L_image->width;
		cp = (unsigned char*)(L_image->imageData);
		cp1 = (unsigned char*)(R_image->imageData);
		thisd = -1;
		imageStep = L_image->widthStep;
		imageStep1 = R_image->widthStep;
		maxx = R_image->width;
	}
	//cost calc
	for (int i = 0; i < Height; ++i)
	{
		rcp = cp + imageStep*i;
		rcp1 = cp1 + imageStep1*i;

		if (i>kernel-1 && i < Height - kernel)
		{
			rcp_ = cp + imageStep*(i - kernel);
			rcp1_ = cp1 + imageStep1*(i - kernel);
			rcp__ = cp + imageStep*(i + kernel);
			rcp1__ = cp1 + imageStep1*(i + kernel);
		}
		for (int j = 0; j < Width; ++j)
		{
			if (j>kernel-1 && j < Width - kernel)
			{
				_rcp = rcp - kernel;
				_rcp1 = rcp1 - kernel;
				__rcp = rcp + kernel;
				__rcp1 = rcp1 + kernel;
			}
			spcost = (unsigned short*)((char*)cost + costwidthstep*i + j*next);
			for (int k = 0; k < counts; ++k)
			{
				if (j + thisd*k >= 0 && (j + thisd*k) < maxx)
				{
					spcost[k] = table[abs(rcp[j] - rcp1[j + thisd*k])];
					if (i>kernel - 1 && i < Height - kernel)
					{
						spcost[k] += table[abs(abs(rcp[j]-rcp_[j])-abs(rcp1[j+thisd*k]-rcp1_[j+thisd*k]))];
						spcost[k] += table[abs(abs(rcp[j] - rcp__[j]) - abs(rcp1[j + thisd*k] - rcp1__[j + thisd*k]))];
					}
					if (j>kernel - 1 && j < Width - kernel)
					{
						spcost[k] += table[abs(abs(rcp[j] - _rcp[j]) - abs(rcp1[j + thisd*k] - _rcp1[j + thisd*k]))];
						spcost[k] += table[abs(abs(rcp[j] - __rcp[j]) - abs(rcp1[j + thisd*k] - __rcp1[j + thisd*k]))];
					}					
				}
					
			}
		}
	}
}

void MI(const IplImage* base, const IplImage* warp, const IplImage* l_image, const IplImage* r_image, int* cost, int min_disprity, int max_disprity, int* cost_r)
{
	//if (base->height != cost->rows&&base->width != cost->cols)throw"MI_SGM,COST MAT is not crorespondent";
	CvMat* Pxy = cvCreateMat(256, 256, CV_32FC1);
	CvMat* Px = cvCreateMat(1, 256, CV_32FC1);
	CvMat* Py = cvCreateMat(1, 256, CV_32FC1);
	cvSet(Pxy, cvScalar(0.0000001));
	cvSet(Px, cvScalar(0));
	cvSet(Py, cvScalar(0));
	unsigned char* pbase(NULL);
	unsigned char* pwarp(NULL);
	float* pf(Pxy->data.fl);
	int sum(0);
	//	clock_t start, end;
	//	start = clock();
	int temp0, temp1;
	for (int i = 0; i <base->height; ++i)
	{
		pbase = (unsigned char*)base->imageData + i*base->widthStep;
		pwarp = (unsigned char*)warp->imageData + i*warp->widthStep;
		for (int j = 0; j != base->width; ++j)
		{
			temp0 = *(pwarp + j);
			temp1 = *(pbase + j);
			if (temp0 >= 0 && temp0 < 256 && temp1 >=0 && temp1 < 256)
			{
				*(pf + temp0 * 256 + temp1) += 1;
				++sum;
			}
		}
	}
//	int pro =(int) sum*0.0005;
	//	 sum =(float) base->width*base->height;
#pragma omp parallel for private(pf)
	for (int i = 0; i <Pxy->height; ++i)
	{
		pf = (float*)(Pxy->data.ptr + Pxy->step*i);
		for (int j = 0; j != Pxy->width; ++j)
		{
			//if (*(pf+j)>pro)
			*(pf + j) /= sum;
		//	else *(pf + j) = 0.000001;
		}
	}
	float* px(Px->data.fl), *py(Py->data.fl), *pxy1(NULL), *pxy2(NULL);
#pragma omp parallel sections
	{
#pragma omp  section
		{
			for (int i = 0; i != Pxy->height; ++i)
			{
				pxy1 = (float*)(Pxy->data.ptr + i*Pxy->step);
				for (int j = 0; j != Pxy->width; ++j)
					px[i] += pxy1[j];
				//	px[i] = sumfloat_avx_4loop_U(pxy1, 256);
			}
		}
#pragma omp  section
		{
		for (int i = 0; i != Pxy->width; ++i)
		{
			pxy2 = (float*)(Pxy->data.fl + i);
			for (int j = 0; j != Pxy->height; ++j)
			{
				pf = pxy2 + j * 256;
				py[i] += (*pf);
			}
		}
	}
	}
	cvSmooth(Pxy, Pxy, CV_GAUSSIAN, 7);
	cvSmooth(Px, Px, CV_GAUSSIAN, 7);
	cvSmooth(Py, Py, CV_GAUSSIAN, 7);
	pf = Px->data.fl;
	for (int i = 0; i != 256; ++i)
		pf[i] = -log(pf[i]);
	pf = Py->data.fl;
	for (int i = 0; i != 256; ++i)
		pf[i] = -log(pf[i]);
//#pragma omp parallel for private(pf)
	for (int i = 0; i <Pxy->height; ++i)
	{
		pf = (float*)(Pxy->data.ptr + Pxy->step*i);
		for (int j = 0; j != Pxy->width; ++j)
		{
			*(pf + j) = -log(*(pf + j));
		}
	}
	cvSmooth(Pxy, Pxy, CV_GAUSSIAN, 7);
	cvSmooth(Px, Px, CV_GAUSSIAN, 7);
	cvSmooth(Py, Py, CV_GAUSSIAN, 7);

	//	pf = Pxy->data.fl;
	//	for (int i = 0; i != 8255; ++i)
	//		cout <<pf[i] << "\t";
		cvNamedWindow("1");
		IplImage* image = cvCreateImage(cvSize(Pxy->cols, Pxy->rows), IPL_DEPTH_32F, 1);
		float kkk0 = Max_float_avx_4loop(Pxy->data.fl, Pxy->cols*Pxy->rows);
		cvCopy(Pxy, image);
		for (int i = 0; i <Pxy->height; ++i)
		{
			pf = (float*)(image->imageData + image->widthStep*i);
			for (int j = 0; j != Pxy->width; ++j)
			{
				*(pf + j) =*(pf + j)/kkk0;
			}
		}
		
	//	image->imageData =(char*) Pxy->data.ptr;
		cvShowImage("1", image);
		cvWaitKey();


	const int step_d = (max_disprity - min_disprity + 1)/2;

	unsigned char* bip = (unsigned char*)l_image->imageData;
	unsigned char* wip(NULL);
	unsigned char b_gray, w_gray;
	unsigned short* cost_xy_d(NULL);
	int co_x;
	int co_y;

#pragma omp parallel for private(bip,wip,b_gray,w_gray,cost_xy_d,co_x,co_y)
	for (int i = 0; i < l_image->height; ++i)
	{
		bip = (unsigned char*)(l_image->imageData + l_image->widthStep*i);
		for (int j = 0; j != l_image->width; ++j)
		{
			cost_xy_d =(unsigned short*)(cost + l_image->width*step_d*i + step_d*j);
	//		temp_line = EpipolarLine_C(j, i, l_image, rectified);
			b_gray = bip[j];
			for (int d = min_disprity, k = 0; d <= max_disprity; ++d, ++k)
			{
				co_x = j - d;
				co_y = i;
				if (co_x >= 0 && co_x < r_image->width&&co_y >= 0 && co_y < r_image->height)
				{
					w_gray = *(r_image->imageData + r_image->widthStep*co_y + co_x);
					cost_xy_d[k] = *((float*)(Pxy->data.ptr + w_gray*Pxy->step + b_gray*sizeof(float))) - *(Px->data.fl + w_gray) - *(Py->data.fl + b_gray);
					
				}

			}
		}
	}

	//	unsigned char* wip;
	Point_Int point;
	//	end = clock();
	//	cout << (float)(end - start) / 1000 << " s" << endl;
	if (cost_r)
	{
#pragma omp parallel for private(wip,point,cost_xy_d,w_gray,co_x,co_y,b_gray)
		for (int i = 0; i < r_image->height; ++i)
		{
			wip = (unsigned char*)(r_image->imageData + i*r_image->widthStep);
			point.b_ = i;
			for (int j = 0; j != r_image->width; ++j)
			{
				cost_xy_d =(unsigned short*) (cost_r + r_image->width*step_d*i + step_d*j);
//				point.a_ = j;
				w_gray = wip[j];
			//	temp_line = EpipolarLine_I(&point, r_image, rectified);
				for (int d = min_disprity, k = 0; d <= max_disprity; ++d, ++k)
				{
					co_x = j + d;
					co_y = i;
					if (co_x >= 0 && co_x < l_image->width&&co_y >= 0 && co_y < l_image->height)
					{
						b_gray = *(l_image->imageData + l_image->widthStep*co_y + co_x);
						cost_xy_d[k] = *((float*)(Pxy->data.ptr + w_gray*Pxy->step + b_gray*sizeof(float))) - *(Px->data.fl + w_gray) - *(Py->data.fl + b_gray);
					}

				}
			}
		}
	}
	cvReleaseMat(&Pxy);
	cvReleaseMat(&Px);
	cvReleaseMat(&Py);
}

unsigned char RT_rank(int i, int j, const unsigned char* pimage,const unsigned char* pcenter, int step, int height, int width,int kernal)
{
	
	const unsigned char* ptemp;
	const unsigned char center = *pcenter;
	unsigned int sum(0);
	int initH = i - kernal;
	int initW = j - kernal;
	int H = i + kernal;
	int W = j + kernal;
	initH=(initH < 0)? 0:initH;
	initW= (initW < 0)?0:initW;
	H= (H>height - 1)?height - 1:H;
	W = (W>width - 1) ? width - 1 : W;
	for (int i = initH; i <= H; ++i)
	{
		ptemp = pimage + i*step;
		for (int j = initW; j <= W; ++j)
		{
			sum += ptemp[j] < center ? 1 : 0;
			//sum += table[abs(center - ptemp[j])];
		}
	}
	//sum /= ((H - initH)*(W - initW));
	return sum;
}
void RS_Cost(const IplImage* L_image, const IplImage* R_image, unsigned int* cost, CvSize size, int min_dis, int max_dis, int Tag_L0_R1,unsigned int kernal)
{
	IplImage* rankL = cvCreateImage(cvSize(L_image->width, L_image->height), L_image->depth, L_image->nChannels);
	IplImage* rankR = cvCreateImage(cvSize(R_image->width, R_image->height), R_image->depth, R_image->nChannels);
	for (int i = 0; i != 256; ++i)
	{
		table[i] = i;
	}
	unsigned char* pimage(NULL),*prankimage(NULL);
	int Width, Height,Step;
	unsigned char* pc, *pc1;
	//unsigned char temp1, temp2, tmep3;
	for (int ii = 0; ii < 2; ++ii)
	{
		//initi
		if (ii)
		{
			Width = R_image->width;
			Height = R_image->height;
			pimage = (unsigned char*)R_image->imageData;
			prankimage = (unsigned char*)rankR->imageData;
			Step = R_image->widthStep;
		}
		else
		{
			Width = L_image->width;
			Height = L_image->height;
			pimage = (unsigned char*)L_image->imageData;
			prankimage = (unsigned char*)rankL->imageData;
			Step = L_image->widthStep;
		}
		//rank transfrom
		for (int i = 0; i < Height; ++i)
		{
			pc = (unsigned char*)(pimage + i*Step);
			pc1 = (unsigned char*)(prankimage + i*Step);
			for (int j = 0; j < Width; ++j)
			{
				pc1[j] = RT_rank(i, j, pimage,pc, Step, Height, Width,kernal);
			}
		}
	}

	//
	
	//init
	int kernel = 5;
	Init_Cost(cost, size, min_dis, max_dis, 8323199);
	const int counts = max_dis - min_dis + 1;
	const int  next = sizeof(short)*counts;
	const int costwidthstep = next*size.width;
	unsigned short* spcost = (unsigned short*)cost;
//	int Height, Width;
	unsigned char* c0p(NULL), *c0p1(NULL), *rc0p, *rc0p1, *rc0p_(NULL), *rc0p1_(NULL), *rc0p__(NULL), *rc0p1__(NULL), *_rc0p(NULL), *_rc0p1(NULL), *__rc0p(NULL), *__rc0p1(NULL);
	unsigned char* rankcp, *rankcp1;
	int thisd(1);
	int imageStep, imageStep1;
	int maxx;
	if (Tag_L0_R1)
	{
		Height = R_image->height;
		Width = R_image->width;
		c0p = (unsigned char*)R_image->imageData;
		c0p1 = (unsigned char*)L_image->imageData;
		rankcp = (unsigned char*)rankR->imageData;
		rankcp1 = (unsigned char*)rankL->imageData;
		thisd = 1;
		imageStep = R_image->widthStep;
		imageStep1 = L_image->widthStep;
		maxx = L_image->width;
	}
	else
	{
		Height = L_image->height;
		Width = L_image->width;
		c0p = (unsigned char*)(L_image->imageData);
		c0p1 = (unsigned char*)(R_image->imageData);
		rankcp1 = (unsigned char*)rankR->imageData;
		rankcp = (unsigned char*)rankL->imageData;
		thisd = -1;
		imageStep = L_image->widthStep;
		imageStep1 = R_image->widthStep;
		maxx = R_image->width;
	}
	//cost calc
	for (int i = 0; i < Height; ++i)
	{
		rc0p = rankcp + imageStep*i;
		rc0p1 = rankcp1 + imageStep1*i;

		if (i>kernel - 1 && i < Height - kernel)
		{
			rc0p_ = c0p + imageStep*(i - kernel);
			rc0p1_ = c0p1 + imageStep1*(i - kernel);
			rc0p__ = c0p + imageStep*(i + kernel);
			rc0p1__ = c0p1 + imageStep1*(i + kernel);
		}
		for (int j = 0; j < Width; ++j)
		{
			if (j>kernel - 1 && j < Width - kernel)
			{
				_rc0p = rc0p - kernel;
				_rc0p1 = rc0p1 - kernel;
				__rc0p = rc0p + kernel;
				__rc0p1 = rc0p1 + kernel;
			}
			spcost = (unsigned short*)((char*)cost + costwidthstep*i + j*next);
			for (int k = 0; k < counts; ++k)
			{
				if (j + thisd*k >= 0 && (j + thisd*k) < maxx)
				{
					spcost[k] = table[abs(rc0p[j] - rc0p1[j + thisd*k])];
					if (i>kernel - 1 && i < Height - kernel)
					{
						spcost[k] += table[abs(abs(rc0p[j] - rc0p_[j]) - abs(rc0p1[j + thisd*k] - rc0p1_[j + thisd*k]))];
						spcost[k] += table[abs(abs(rc0p[j] - rc0p__[j]) - abs(rc0p1[j + thisd*k] - rc0p1__[j + thisd*k]))];
					}
					if (j>kernel - 1 && j < Width - kernel)
					{
						spcost[k] += table[abs(abs(rc0p[j] - _rc0p[j]) - abs(rc0p1[j + thisd*k] - _rc0p1[j + thisd*k]))];
						spcost[k] += table[abs(abs(rc0p[j] - __rc0p[j]) - abs(rc0p1[j + thisd*k] - __rc0p1[j + thisd*k]))];
					}

				}

			}
		}
	}
	cvReleaseImage(&rankL);
	cvReleaseImage(&rankR);
}



void Disparity(unsigned int* cost, int min_dis, int max_dis, IplImage* disparity, IplImage* L_Rimage,unsigned short P1,unsigned short MAX_)
{
	int counts = max_dis - min_dis + 1;
	unsigned int* S = new unsigned int[disparity->width*disparity->height*counts/2];
	Init_Cost((unsigned int*)S, cvSize(disparity->width, disparity->height), min_dis, max_dis, 0);

	unsigned short* pscalc = (unsigned short*)S;
	unsigned short* pcost = (unsigned short*)cost;
	const int calccount = counts;
//	unsigned int * psmove = S;
	const int movecount = counts / 2;

//	unsigned short P1 = 9;
//	unsigned short MAX_ = 20;
	IplImage* P2 = cvCreateImage(cvSize(L_Rimage->width, L_Rimage->height), IPL_DEPTH_8U, 1);
	Value_P2(L_Rimage, P2, P1, MAX_);

//	int small_index(0);
	unsigned int* sdp1(S), *sdp2(S);
	const unsigned int* cdp1(cost), *cdp2(cost);
	const int Width(P2->width), Height(P2->height);
	Point_Int Point_diriction[16] = { 1, 0, 0, -1, 0, 1, -1, 0, 1, 1, 1, -1, -1, 1, -1, -1, 1, 2, 1, -2, -1, 2, -1, -2, 2, 1, 2, -1, -2, 1, -2, -1 };
	Point_Int point_0, point_1, point_2, point_3;
	int step_int = Width*movecount;
	int step_short = Width*calccount;


	const unsigned char* p2(NULL);
	unsigned short* ps0;
	unsigned short* ps;
	const unsigned short* pc;
	unsigned short Smin;
	unsigned short min;
	unsigned short Save;
	for (int i = 0; i != 16; ++i)
	{
		for (int ii = 0; ii < Width; ++ii)
		{
			cdp1 = cost + ii*movecount;
			cdp2 = cost + (Height - 1)*step_int + ii*movecount;
			sdp1 = S + ii*movecount;
			sdp2 = S + (Height - 1)*step_int + ii*movecount;

			// memcpy(sdp1,cdp1,counts_d)
			//
			for (int j = 0; j < movecount; ++j)
			{
				sdp1[j] = cdp1[j];
				sdp2[j] = cdp2[j];
			}
		}

		for (int ii = 0; ii < Height; ++ii)
		{
			cdp1 = cost + ii*step_int;
			cdp2 = cost + ii*step_int + (Width - 1)*movecount;
			sdp1 = S + ii*step_int;
			sdp2 = S + ii*step_int + (Width - 1)*movecount;

			// memcpy(sdp1,cdp1,counts_d)
			//
			for (int j = 0; j < movecount; ++j)
			{
				sdp1[j] = cdp1[j];
				sdp2[j] = cdp2[j];
			}

		}
#pragma omp parallel for private(point_0,ps0,ps,p2,pc,Smin,min,Save)
		for (int j = 0; j < Width; ++j)
		{
			point_0 = { j, 0 };
			ps0 = (pscalc + j*calccount);
			point_0.a_ += Point_diriction[i].a_;
			point_0.b_ += Point_diriction[i].b_;
			while (point_0.a_ >= 0 && point_0.a_ < Width&&point_0.b_ >= 0 && point_0.b_ < Height)
			{
				min_index_common_4_loop(ps0, calccount, &Smin);
				ps = (pscalc + point_0.a_*calccount + point_0.b_*step_short);
				pc = (pcost + point_0.a_*calccount + point_0.b_*step_short);
				p2 = (unsigned char*)(P2->imageData + P2->widthStep*point_0.b_ + point_0.a_);
				min = (Smin + *p2);
				min = min < ps0[0] ? min : ps0[0];
				min = min < (ps0[1] + P1) ? min : (ps0[1] + P1);
				min = min - Smin;
				for (int h = 1; h < calccount; ++h)
				{
					Save = ps0[h - 1];
					ps0[h - 1] = min + pc[h - 1];
					min = (Smin + *p2);
					min = min < (Save + P1) ? min : (Save + P1);
					min = min < ps0[h] ? min : ps0[h];
					if (h < calccount - 1)min = min < (ps0[h + 1] + P1) ? min : (ps0[h + 1] + P1);
					min = min - Smin;
				}
				ps0[calccount - 1] = min + pc[calccount - 1];
				Acc_4((unsigned int*)ps0, (unsigned int*)ps, movecount);
				point_0.a_ += Point_diriction[i].a_;
				point_0.b_ += Point_diriction[i].b_;
			}
		}
#pragma omp parallel for private(point_1,ps0,ps,p2,pc,Smin,min,Save) 
		for (int k = 0; k < Height; ++k)
		{
			point_1 = { 0, k };
			ps0 = (pscalc + k*step_short);
			point_1.a_ += Point_diriction[i].a_;
			point_1.b_ += Point_diriction[i].b_;
			while (point_1.a_ >= 0 && point_1.a_ < Width&&point_1.b_ >= 0 && point_1.b_ < Height)
			{
				min_index_common_4_loop(ps0, calccount, &Smin);
				ps = (pscalc + point_1.a_*calccount + point_1.b_*step_short);
				pc = (pcost + point_1.a_*calccount + point_1.b_*step_short);
				p2 = (unsigned char*)(P2->imageData + P2->widthStep*point_1.b_ + point_1.a_);
				min = (Smin + *p2);
				min = min < ps0[0] ? min : ps0[0];
				min = min < (ps0[1] + P1) ? min : (ps0[1] + P1);
				min = min - Smin;
				for (int h = 1; h < calccount; ++h)
				{
					Save = ps0[h - 1];
					ps0[h - 1] = min + pc[h - 1];
					min = (Smin + *p2);
					min = min < (Save + P1) ? min : (Save + P1);
					min = min < ps0[h] ? min : ps0[h];
					if (h < calccount- 1)min = min < (ps0[h + 1] + P1) ? min : (ps0[h + 1] + P1);
					min = min - Smin;
				}
				ps0[calccount - 1] = min + pc[calccount- 1];
				Acc_4((unsigned int*)ps0, (unsigned int*)ps, movecount);
				point_1.a_ += Point_diriction[i].a_;
				point_1.b_ += Point_diriction[i].b_;

			}
		}
#pragma omp parallel for private(point_2,ps0,ps,p2,pc,Smin,min,Save)
		for (int m = 0; m < Width; ++m)
		{
			point_2 = { m, Height - 1 };
			ps0 = (pscalc + m*calccount + (Height - 1)*step_short);
			point_2.a_ += Point_diriction[i].a_;
			point_2.b_ += Point_diriction[i].b_;
			while (point_2.a_ >= 0 && point_2.a_ < Width&&point_2.b_ >= 0 && point_2.b_ < Height)
			{
				min_index_common_4_loop(ps0, calccount, &Smin);
				ps = (pscalc + point_2.a_*calccount + point_2.b_*step_short);
				pc = (pcost + point_2.a_*calccount + point_2.b_*step_short);
				p2 = (unsigned char*)(P2->imageData + P2->widthStep*point_2.b_ + point_2.a_);
				min = (Smin + *p2);
				min = min < ps0[0] ? min : ps0[0];
				min = min < (ps0[1] + P1) ? min : (ps0[1] + P1);
				min = min - Smin;
				for (int h = 1; h < calccount; ++h)
				{
					Save = ps0[h - 1];
					ps0[h - 1] = min + pc[h - 1];
					min = (Smin + *p2);
					min = min < (Save + P1) ? min : (Save + P1);
					min = min < ps0[h] ? min : ps0[h];
					if (h<calccount - 1)min = min < (ps0[h + 1] + P1) ? min : (ps0[h + 1] + P1);
					min = min - Smin;
				}
				ps0[calccount - 1] = min + pc[calccount - 1];
				Acc_4((unsigned int*)ps0, (unsigned int*)ps, movecount);
				point_2.a_ += Point_diriction[i].a_;
				point_2.b_ += Point_diriction[i].b_;
			}

		}
#pragma omp parallel for private(point_3,ps0,ps,p2,pc,Smin,min,Save)
		for (int n = 0; n < Height; ++n)
		{
			point_3 = { Width - 1, n };
			ps0 = (pscalc + (Width - 1)*calccount + n*step_short);
			point_3.a_ += Point_diriction[i].a_;
			point_3.b_ += Point_diriction[i].b_;
			while (point_3.a_ >= 0 && point_3.a_ < Width&&point_3.b_ >= 0 && point_3.b_ < Height)
			{
				min_index_common_4_loop(ps0, calccount, &Smin);
				ps = (pscalc + point_3.a_*calccount + point_3.b_*step_short);
				pc = (pcost + point_3.a_*calccount + point_3.b_*step_short);
				p2 = (unsigned char*)(P2->imageData + P2->widthStep*point_3.b_ + point_3.a_);
				min = (Smin + *p2);
				min = min < ps0[0] ? min : ps0[0];
				min = min < (ps0[1] + P1) ? min : (ps0[1] + P1);
				min = min - Smin;
				for (int h = 1; h < calccount; ++h)
				{
					Save = ps0[h - 1];
					ps0[h - 1] = min + pc[h - 1];
					min = (Smin + *p2);
					min = min < (Save + P1) ? min : (Save + P1);
					min = min < ps0[h] ? min : ps0[h];
					if (h<calccount - 1)min = min < (ps0[h + 1] + P1) ? min : (ps0[h + 1] + P1);
					min = min - Smin;
				}
				ps0[calccount - 1] = min + pc[calccount - 1];
				Acc_4((unsigned int*)ps0, (unsigned int*)ps, movecount);
				point_3.a_ += Point_diriction[i].a_;
				point_3.b_ += Point_diriction[i].b_;
			}

		}
	}

	float* dp(NULL);
#pragma omp parallel for private(ps,dp)
	for (int i = 0; i < disparity->height; ++i)
	{
		dp = (float*)(disparity->imageData + i*disparity->widthStep);
		ps = pscalc + step_short*i;
		for (int j = 0; j<disparity->width; ++j)
		{
			dp[j] = (float)(min_dis + min_index_common_4_loop(ps, calccount));
			ps += calccount;
		}
	}
	cvReleaseImage(&P2);
	delete[]S;
}

void RangeDisparity(const IplImage* L_image, const IplImage* R_image, int* min_dis, int* max_dis)
{
	IplImage* L_ = cvCreateImage(cvSize(L_image->width / 2, L_image->height / 2), L_image->depth, L_image->nChannels);
	IplImage* R_ = cvCreateImage(cvSize(R_image->width / 2, R_image->height/2), R_image->depth, R_image->nChannels);
	cvResize(L_image, L_);
	cvResize(R_image, R_);
	int initd = ((L_->width /4) / 4 + 1) * 4;
	IplImage* l = cvCreateImage(cvSize(L_->width, L_->height), IPL_DEPTH_8U, 1);
	IplImage* r = cvCreateImage(cvSize(R_->width, R_->height), IPL_DEPTH_8U, 1);
	IplImage* disparity = cvCreateImage(cvSize(L_->width, L_->height), IPL_DEPTH_32F, 1);
	cvConvertImage(L_, l);
	cvConvertImage(R_, r);

	int *p = new int[l->width*l->height * initd];
	Co_Cost(l, r, (unsigned int*)p, cvSize(l->width, l->height), 0, 2*initd-1, 0,5);
	Disparity((unsigned int*)p, 0, 2*initd-1, disparity, l);
	Point_Int minmax;
	minmax=Statistics(disparity, 5, 0, 2 * initd - 1);
	*min_dis = minmax.a_;
	*max_dis =minmax.a_+(2*(minmax.b_ - minmax.a_) / 8 + 2) * 8 - 1;

	cvReleaseImage(&disparity);
	cvReleaseImage(&L_);
	cvReleaseImage(&R_);
	cvReleaseImage(&l);
	cvReleaseImage(&r);
}

void Check(IplImage* Disprity_l,  IplImage* Disprity_r,int do_r_dis)
{
	Point_Int point;
	float* pf(NULL), *pfr(NULL);
	IplImage* disparityl_copy = cvCreateImage(cvSize(Disprity_l->width, Disprity_l->height), IPL_DEPTH_32F, 1);
	cvSmooth(Disprity_r, Disprity_r, CV_MEDIAN);
	cvSmooth(Disprity_l, Disprity_l, CV_MEDIAN);
	if (do_r_dis)
	{
		cvCopy(Disprity_l, disparityl_copy);
	}
#pragma omp parallel for private(point,pf,pfr)
	for (int i = 0; i < Disprity_l->height; ++i)
	{
		pf = (float*)(Disprity_l->imageData + Disprity_l->widthStep*i);
		for (int j = 0; j != Disprity_l->width; ++j)
		{
			point.a_ = j - pf[j];
			point.b_ = i;
			pfr = (float*)(Disprity_r->imageData + Disprity_r->widthStep*point.b_ + point.a_*sizeof(float));
			if (point.a_<0 || point.a_>=Disprity_r->width || point.b_<0 || point.b_>=Disprity_r->height || (abs(pf[j] - *pfr)>2))
			{
				if (point.a_>=0&&point.a_<Disprity_r->width&&pf[j] < *pfr)pf[j] = 0;  //ÕÚµ²
				else pf[j] = 0;				//ÎóÆ¥Åä	

			}
		}
	}
	if (do_r_dis)
  {
#pragma omp parallel for private(point,pf,pfr)
	for (int i = 0; i < Disprity_r->height; ++i)
	{
		pf = (float*)(Disprity_r->imageData + Disprity_r->widthStep*i);
		for (int j = 0; j != Disprity_r->width; ++j)
		{
			point.a_ = j + pf[j];
			point.b_ = i;
			pfr = (float*)(disparityl_copy->imageData + disparityl_copy->widthStep*point.b_ + point.a_*sizeof(float));
			if (point.a_<0 || point.a_>=Disprity_r->width || point.b_<0 || point.b_>=Disprity_r->height || (abs(pf[j] - *pfr)>2))
			{
				if (point.a_>=0&&point.a_<Disprity_r->width&&pf[j] < *pfr)pf[j] = 0;  //ÕÚµ²
				else pf[j] = 0;				//ÎóÆ¥Åä	

			}
		}
	}
  }
	cvReleaseImage(&disparityl_copy);
}

void HMI_SGM(const IplImage* l_image, const IplImage* r_image, IplImage* disparity, IplImage* R_disparity , int Post_Process)
{
	CvSize lsize, rsize;
	int count_d{0}, temp{0};
	int* p{NULL}, *rp{NULL};
	IplImage* prelimage(NULL);
	IplImage* warp(NULL);
//	IplImage* P2_r, *P2;
	Point_Int minmax{0,0};
	IplImage* disprity_r, *disprity, *Wdisprity(NULL);
	IplImage* limage, *rimage;
	IplImage* l = cvCreateImage(cvSize(l_image->width, l_image->height), IPL_DEPTH_8U, 1);
	IplImage* r = cvCreateImage(cvSize(r_image->width, r_image->height), IPL_DEPTH_8U, 1);
	cvConvertImage(l_image, l);
	cvConvertImage(r_image, r);

	for (int i = 0; i != 7; ++i)
	{
		if (i < 3)
		{
			lsize.width = l_image->width / 16;
			lsize.height = l_image->height / 16;
			rsize.width = r_image->width / 16;
			rsize.height = r_image->height / 16;
		}
		else
		{
			temp = 1;
			temp = temp << (6 - i);
			if (i == 7)temp = 1;
			lsize.width = l_image->width / temp;
			lsize.height = l_image->height / temp;
			rsize.width = r_image->width / temp;
			rsize.height = r_image->height / temp;
		}
		limage = cvCreateImage(lsize, IPL_DEPTH_8U, 1);
		rimage = cvCreateImage(rsize, IPL_DEPTH_8U, 1);
		cvResize(l, limage);
		cvResize(r, rimage);
		if (i == 0)
		{
			prelimage = l;
			warp = r;//
			count_d = (limage->width / 8+1) * 8;
			minmax.a_ = 0;
			minmax.b_ = count_d - 1;
		}

		disprity = cvCreateImage(cvSize(limage->width, limage->height), IPL_DEPTH_32F, 1);
		disprity_r = cvCreateImage(cvSize(limage->width, limage->height), IPL_DEPTH_32F, 1);
		p = new int[limage->width*limage->height * count_d/2];

		rp = new int[rimage->width*rimage->height * count_d/2];
		Init_Cost((unsigned int*)p, cvSize(limage->width, limage->height), minmax.a_, minmax.b_, 8323199);
		Init_Cost((unsigned int*)rp, cvSize(rimage->width, rimage->height), minmax.a_, minmax.b_, 8323199);
		if (i >= 3){
			prelimage = limage;
			warp = cvCreateImage(cvSize(limage->width, limage->height), limage->depth, limage->nChannels);
			Warp_(Wdisprity, limage, rimage, warp);
			cvReleaseImage(&Wdisprity);
		}
		if (i >= 2)Wdisprity = cvCreateImage(cvSize(limage->width*2, limage->height*2), IPL_DEPTH_32F, 1);
		MI(prelimage, warp,limage, rimage, p, minmax.a_, minmax.b_, rp);

//		cvNamedWindow("warp");
//		cvShowImage("warp", warp);
//		cvWaitKey();

		Disparity((unsigned int*)p, minmax.a_, minmax.b_, disprity, limage,40,100);
		Disparity((unsigned int*)rp, minmax.a_, minmax.b_, disprity_r, rimage,40,100);
		if (prelimage != l)
		{
			cvReleaseImage(&prelimage);
			cvReleaseImage(&warp);
		}	
		delete[]p;
		delete[]rp;
		if (i == 6)
		{

			//cvSmooth(disprity, disprity, CV_MEDIAN,3);
			//Peak(disprity, 2);
			//Peak(disprity, 2);
			Peak(disprity, 1,0);
			//Peak(disprity, 0);
			Check(disprity, disprity_r, 1);
			//Peak(disprity, 2);
			cvCopy(disprity, disparity);			
			if (R_disparity)cvCopy(disprity_r, R_disparity);
		}
		cvReleaseImage(&disprity_r);
		minmax = Statistics(disprity, 2, minmax.a_, minmax.b_);
		count_d = ((minmax.b_ - minmax.a_) /8 + 1) * 8;		//0-max
		if (i > 3 && i<7)count_d *= 2;
		minmax.b_ = minmax.a_ + count_d - 1;
		if(i>=2)cvResize(disprity, Wdisprity);	
		//cvReleaseImage(&limage);
		cvReleaseImage(&rimage);
		cvReleaseImage(&disprity);

	}
	cvReleaseImage(&prelimage);
	cvReleaseImage(&warp);
	cvReleaseImage(&Wdisprity);
}


void SGM(const IplImage* l_image, const IplImage* r_image, IplImage* disparity, IplImage* R_disparity, int type, int Post_Process)
{
	int s(0), m(0);
	RangeDisparity(l_image, r_image, &s, &m);

	IplImage* l = cvCreateImage(cvSize(l_image->width, l_image->height), IPL_DEPTH_8U, 1);
	IplImage* r = cvCreateImage(cvSize(r_image->width, r_image->height), IPL_DEPTH_8U, 1);
	IplImage* disparity_l = cvCreateImage(cvSize(l_image->width, l_image->height), IPL_DEPTH_32F, 1);
	IplImage* disparity_r = cvCreateImage(cvSize(r_image->width, r_image->height), IPL_DEPTH_32F, 1);
	cvConvertImage(l_image, l);
	cvConvertImage(r_image, r);

	int *p = new int[l->width*l->height * (m - s + 1) / 2];
	if (type == AD_STRUCTURE)Co_Cost(l, r, (unsigned int*)p, cvSize(l->width, l->height), s, m, 0, 5);
	else
		RS_Cost(l, r, (unsigned int*)p, cvSize(l->width, l->height), s, m, 0, 3);
	Disparity((unsigned int*)p, s, m, disparity_l, l);
	delete[]p;
	p = new int[r->width*r->height*(m - s + 1) / 2];
	if (type == AD_STRUCTURE)Co_Cost(l, r, (unsigned int*)p, cvSize(l->width, l->height), s, m, 1, 5);
	else
		RS_Cost(l, r, (unsigned int*)p, cvSize(l->width, l->height), s, m, 1, 3);
	Disparity((unsigned int*)p, s, m, disparity_r, r);
	delete[]p;

	if (!R_disparity)
		Check(disparity_l, disparity_r);
	else
		Check(disparity_l, disparity_r, 1);
	if (disparity)
	{
		Peak(disparity_l, 2, 0);
		Interpolation(disparity_l);
		//if(Post_Process) PostProcessing();
		cvCopy(disparity_l, disparity);
	}
	if (R_disparity)
	{
		Peak(disparity_r, 0, 0);
		Interpolation(disparity_r);
		//if(Post_Process)PostProcessing();
		cvCopy(disparity_r, R_disparity);
	}
	cvReleaseImage(&l);
	cvReleaseImage(&r);
	cvReleaseImage(&disparity_l);
	cvReleaseImage(&disparity_r);
}
















float Max_float_avx_4loop(const float* data, int counts)
{
	int loop = counts / 32;
	int left = counts % 32;
	__m256 min1, min2, min3, min4, load1, load2, load3, load4, min;
	const float* p(data), *q;
	float temp(*data);
	min1 = _mm256_loadu_ps(p);
	min2 = _mm256_loadu_ps(p);
	min4 = _mm256_loadu_ps(p);
	min3 = _mm256_loadu_ps(p);

	for (int i = 0; i != loop; ++i)
	{
		load1 = _mm256_loadu_ps(p);
		load2 = _mm256_loadu_ps(p + 8);
		load3 = _mm256_loadu_ps(p + 16);
		load4 = _mm256_loadu_ps(p + 24);
		p = p + 32;
		min1 = _mm256_max_ps(min1, load1);
		min2 = _mm256_max_ps(min2, load2);
		min3 = _mm256_max_ps(min3, load3);
		min4 = _mm256_max_ps(min4, load4);
	}
	min = _mm256_max_ps(min1, min2);
	min1 = _mm256_max_ps(min3, min4);
	min = _mm256_max_ps(min, min1);
	q = (const float*)&min;
	for (int i = 0; i != 8; ++i)
	if (temp<q[i])
		temp = q[i];
	for (int i = 0; i != left; ++i)
	if (temp < p[i])
		temp = p[i];
	return temp;
}
void Value_P2(const IplImage* base, IplImage* P2, int P1, int MAX_)//P1+MAX_ largest;
{
	IplImage* pfi = cvCreateImage(cvSize(base->width, base->height), IPL_DEPTH_32F, 1);
	cvLaplace(base, pfi);
	int counts = base->width*base->height;
	float* f = (float*)pfi->imageData;
	float max = Max_float_avx_4loop(f, counts);
	unsigned char* p2(NULL);
#pragma omp parallel for private(p2,f)
	for (int i = 0; i < base->height; ++i)
	{
		p2 = (unsigned char*)(P2->imageData + P2->widthStep*i);
		f = (float*)(pfi->imageData + pfi->widthStep*i);
		for (int j = 0; j != base->width; ++j)
		{
			p2[j] =(unsigned char)( P1 + MAX_*(max - f[j]) / max);
		}
	}
	cvReleaseImage(&pfi);
}

int min_index_common_4_loop(const unsigned short* data, int counts, unsigned short* result)
{
	int index0(0), index1(0), index2(0), index3(0);
	int loop = counts / 4;
	int left = counts % 4;
	unsigned short min0(*data), min1(*data), min2(*data), min3(*data);
	const unsigned short *p(data);
	for (int i = 0; i != loop; ++i)
	{
		index0 = min0 < *(p) ? index0 : (p - data);
		min0 = min0 < *(p) ? min0 : *(p);
		index1 = min1 < *(p + 1) ? index1 : (p + 1 - data);
		min1 = min1 < *(p + 1) ? min1 : *(p + 1);
		index2 = min2 < *(p + 2) ? index2 : (p + 2 - data);
		min2 = min2 < *(p + 2) ? min2 : *(p + 2);
		index3 = min3 < *(p + 3) ? index3 : (p + 3 - data);
		min3 = min3 < *(p + 3) ? min3 : *(p + 3);
		p = p + 4;
	}
	index0 = min0 < min1 ? index0 : index1;
	min0 = min0 < min1 ? min0 : min1;
	index0 = min0 < min2 ? index0 : index2;
	min0 = min0 < min2 ? min0 : min2;
	index0 = min0 < min3 ? index0 : index3;
	min0 = min0 < min3 ? min0 : min3;
	for (int k = 0; k != left; ++k)
	{
		index0 = min0 < *(p) ? index0 : (p - data);
		min0 = min0 < *(p) ? min0 : *(p);
		++p;
	}
	if (result)*result = min0;
	return index0;
}

void Acc_4(const unsigned int* src, unsigned int* dest, int counts)
{
	const unsigned int* p0(src), *p1(src + 1), *p2(src + 2), *p3(src + 3);
	unsigned int* pp0(dest), *pp1(dest + 1), *pp2(dest + 2), *pp3(dest + 3);
	int loop = counts / 4;
	int left = counts % 4;
	for (int i = 0; i != loop; ++i)
	{
		*(pp0) += *p0;
		*(pp1) += *p1;
		*(pp2) += *p2;
		*(pp3) += *p3;
		p0 += 4;
		p1 += 4;
		p2 += 4;
		p3 += 4;
		pp0 += 4;
		pp1 += 4;
		pp2 += 4;
		pp3 += 4;
	}

	for (int i = 0; i != left; ++i)
	{
		*(pp3) += *(p3);
		++pp3;
		++p3;
	}

}

void Peak(IplImage* disprity_, int con_point_one_dir,int data)
{
	IplImage* disprity = cvCreateImage(cvSize(disprity_->width, disprity_->height), disprity_->depth, disprity_->nChannels);
	cvCopy(disprity_, disprity);
	const int counts = 4 * con_point_one_dir + 1;
	int sum(0);
//	float* pf;
	float temp1, temp2;
	int k(0), tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8;
	for (int i = 0; i < disprity->height; ++i)
	{
//		pf = (float*)(disprity->imageData + disprity->widthStep*i);
		for (int j = 0; j < disprity->width; ++j)
		{
			k = tag1 = tag2 = tag3 = tag4 = tag5 = tag6 = tag7 = tag8 = 1;
			sum = 0;
			temp1 = *((float*)(disprity->imageData + disprity->widthStep*i + sizeof(float)*j));
			while (1)
			{
				if (tag1 && (i - k) >= 0)
				{
					temp2 = *((float*)(disprity->imageData + disprity->widthStep*(i - k) + sizeof(float)*j));
					if (abs(temp1 - temp2) <2)sum += 1;
					else
						tag1 = 0;
				}
				else
					tag1 = 0;
				if (tag2 && (i + k) <disprity->height)
				{
					temp2 = *((float*)(disprity->imageData + disprity->widthStep*(i + k) + sizeof(float)*j));
					if (abs(temp1 - temp2) <2)sum += 1;
					else tag2 = 0;
				}
				else tag2 = 0;
				if (tag3 && (j - k) >= 0)
				{
					temp2 = *((float*)(disprity->imageData + disprity->widthStep*i + sizeof(float)*(j - k)));
					if (abs(temp1 - temp2) <2)sum += 1;
					else tag3 = 0;
				}
				else tag3 = 0;
				if (tag4 && (j + k) < disprity->width)
				{
					temp2 = *((float*)(disprity->imageData + disprity->widthStep*i + sizeof(float)*(j + k)));
					if (abs(temp1 - temp2) <2)sum += 1;
					else tag4 = 0;
				}
				else tag4 = 0;
				if (tag5 && (i - k) >= 0 && (j - k) >= 0)
				{
					temp2 = *((float*)(disprity->imageData + disprity->widthStep*(i - k) + sizeof(float)*(j - k)));
					if (abs(temp1 - temp2) <2)sum += 1;
					else tag5 = 0;
				}
				else tag5 = 0;
				if (tag6 && (i - k) >= 0 && (j + k)<disprity->width)
				{
					temp2 = *((float*)(disprity->imageData + disprity->widthStep*(i - k) + sizeof(float)*(j + k)));
					if (abs(temp1 - temp2) <2)sum += 1;
					else tag6 = 0;
				}
				else tag6 = 0;
				if (tag7 && (i + k) <disprity->height && (j - k) <= 0)
				{
					temp2 = *((float*)(disprity->imageData + disprity->widthStep*(i + k) + sizeof(float)*(j - k)));
					if (abs(temp1 - temp2) <2)sum += 1;
					else tag7 = 0;
				}
				else tag7 = 0;
				if (tag8 && (i + k) < disprity->height && (j + k)<disprity->width)
				{
					temp2 = *((float*)(disprity->imageData + disprity->widthStep*(i + k) + sizeof(float)*(j + k)));
					if (abs(temp1 - temp2) <2)sum += 1;
					else tag8 = 0;
				}
				else tag8 = 0;
				++k;
				if (sum>counts || !(tag1 || tag2 || tag3 || tag4 || tag5 || tag6 || tag7 || tag8))
				{
					if (sum>counts)break;
					else
					{
						*((float*)(disprity_->imageData + disprity_->widthStep*i + sizeof(float)*j)) = data;
						break;
					}
				}
			}
		}
	}
	cvReleaseImage(&disprity);
}

Point_Int Statistics(const IplImage* disprity, float proprotion, int min_disprity, int max_disprity)
{
	Point_Int minmax{ min_disprity, max_disprity };
	const int Counts = disprity->width*disprity->height;
	int drop = Counts*proprotion*0.01;
	int HC = max_disprity - min_disprity + 1;
	int* hestt = new int[HC];
	int* hest = hestt;
	for (int i = 0; i != HC; ++i)
		hest[i] = 0;
	const float* pf;
	int temp;
	for (int i = 0; i < disprity->height; ++i)
	{
		pf = (float*)(disprity->imageData + disprity->widthStep*i);
		for (int j = 0; j < disprity->width; ++j)
		{
			if (pf[j] + 1000>0.1 || pf[j] + 1000 < -0.1)
			{
				temp = (int)(pf[j] - min_disprity);
				hest[temp] += 1;
			}
		}
	}

	int* head(hest), *tail(hest + HC - 1);
	int sumH(0), sumT(0);

	int tag1(1), tag2(1);
	for (int i = 0; i != HC; ++i)
	{
		sumH += head[i];
		sumT += *(tail - i);
		if (tag1 && (sumH > drop))
		{
			minmax.a_ = min_disprity + i;
			tag1 = 0;
		}
		if (tag2 && (sumT > drop))
		{
			minmax.b_ = max_disprity - i;
			tag2 = 0;
		}
		if (tag1 == 0 && tag2 == 0)
			break;

	}
	delete[] hestt;
	return minmax;
}
int Smallth(int * data, int counts, int proportion_smallth)
{
	int index = (int)(counts*proportion_smallth*0.01);
	int temp(0);
	int i;
	int j;
	for (i=0; i < counts-1; ++i)
	{
		for (j = i+1; j < counts; ++j)
		{
			if (data[i]>data[j])
			{
				temp = data[i];
				data[i] =data[j];
				data[j] = temp;
			}
		}
	}
	return data[index];
}
void Interpolation(IplImage* disparity)
{
	Point_Int Point_diriction[16] = { 1, 0, 0, -1, 0, 1, -1, 0, 1, 1, 1, -1, -1, 1, -1, -1, 1, 2, 1, -2, -1, 2, -1, -2, 2, 1, 2, -1, -2, 1, -2, -1 };
	float* pf,temp;
	Point_Int point;
	int data[16], index;
//#pragma omp parallel for private(pf,pfc,temp,point,index,data[16])
	//IplImage* copy = cvCreateImage(cvSize(disparity->width, disparity->height), disparity->depth, disparity->nChannels);
	//cvCopy(disparity, copy);
	for (int i = 0; i < disparity->height; ++i)
	{
		pf = (float*)(disparity->imageData + disparity->widthStep*i);
		//pfc = (float*)(copy->imageData + copy->widthStep*i);
		for (int j = 0; j < disparity->width; ++j)
		{
			if (pf[j] < 0.01)
			{
				index = 0;
				for (int k = 0; k < 16; ++k)
				{
					point.a_ = j;
					point.b_ = i;
					for (int n = 0; n < 50; ++n)
					{
						point.a_ += Point_diriction[k].a_;
						point.b_ += Point_diriction[k].b_;
						if (point.a_ >= 0 && point.a_ < disparity->width&&point.b_ >= 0 && point.b_ < disparity->height)
						{
							temp = *((float*)(disparity->imageData + disparity->widthStep*point.b_ + point.a_*sizeof(float)));
							if (temp>0.1)
							{
								data[index++] = (int)temp;
								break;
							}
							
						}
						
					}
				}
				pf[j] = Smallth(data, index, 28);
			}
		}
	}
	//cvReleaseImage(&copy);
}



void Warp_(const IplImage* disprity, const IplImage* l_image, const IplImage* r_image, IplImage* warp)
{
//	Point line;
	Point_Int point;
	char* wp;
	const float* dfp;
	//if(l_image)cvCopy(l_image, warp);
	cvSet(warp, cvScalar(0));
#pragma omp parallel for private(dfp,wp,point)
	for (int i = 0; i < disprity->height; ++i)
	{
		dfp = (float*)(disprity->imageData + disprity->widthStep*i);
		wp = warp->imageData + warp->widthStep*i;
		for (int j = 0; j < disprity->width; ++j)
		{
			if (dfp[j] >0.01 || dfp[j] <-0.01)
			{
				//line = EpipolarLine_C(j, i, l_image, rectified);
				point.a_ = (int)(j - dfp[j]);
				point.b_ = i;
				if (point.a_ >= 0 && point.a_ < r_image->width&&point.b_ >= 0 && point.b_ < r_image->height)
					wp[j] = *(r_image->imageData + r_image->widthStep*point.b_ + point.a_);
			}
		}
	}
}
