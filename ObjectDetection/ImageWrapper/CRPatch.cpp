#include "CRPatch.h"
#include <highgui.h>

// Open Mpi disabled
//#ifdef DWIN
//#include <omp.h>
//#endif

#include <cstdio>
#include <cmath>
#include <deque>

using namespace std;

void CRPatch::extractPatches(IplImage *img, unsigned int n, int label, CvRect* box, std::vector<CvPoint>* vCenter) {
  // extract features
  vector<IplImage*> vImg;
  extractFeatureChannels(img, vImg);

  CvMat tmp;
  int offx = width/2; 
  int offy = height/2;

#if 0
  if(box!=0) {
    cvGetSubRect( vImg[0], &tmp, *box );
    cvNamedWindow("Rect",1);
    cvShowImage("Rect", &tmp);
    cvWaitKey(1000);
    cvDestroyWindow("Rect");
  }
#endif

  // generate x,y locations
  CvMat* locations = cvCreateMat( n, 1, CV_32SC2 );
  if(box==0)
    cvRandArr( cvRNG, locations, CV_RAND_UNI, cvScalar(0,0,0,0), cvScalar(img->width-width,img->height-height,0,0) );
  else
    cvRandArr( cvRNG, locations, CV_RAND_UNI, cvScalar(box->x,box->y,0,0), cvScalar(box->x+box->width-width,box->y+box->height-height,0,0) );

  // reserve memory
  unsigned int offset = vLPatches[label].size();
  vLPatches[label].reserve(offset+n);
  for(unsigned int i=0; i<n; ++i) {
    CvPoint pt = *(CvPoint*)cvPtr1D( locations, i, 0 );
    
    PatchFeature pf;
    vLPatches[label].push_back(pf);

    vLPatches[label].back().roi.x = pt.x;  vLPatches[label].back().roi.y = pt.y;  
    vLPatches[label].back().roi.width = width;  vLPatches[label].back().roi.height = height; 

    if(vCenter!=0) {
      vLPatches[label].back().center.resize(vCenter->size());
      for(unsigned int c = 0; c<vCenter->size(); ++c) {
        vLPatches[label].back().center[c].x = pt.x + offx - (*vCenter)[c].x;
        vLPatches[label].back().center[c].y = pt.y + offy - (*vCenter)[c].y;
        //cout << vLPatches[label].back().center[c].x << "/" << vLPatches[label].back().center[c].y << " ";
      }
      //cout << endl;
    }

    vLPatches[label].back().vPatch.resize(vImg.size());
    for(unsigned int c=0; c<vImg.size(); ++c) {
      cvGetSubRect( vImg[c], &tmp,  vLPatches[label].back().roi );
      vLPatches[label].back().vPatch[c] = cvCloneMat(&tmp);
    }

    //vLPatches[label].back().show(2000);
  }

  cvReleaseMat(&locations);
  for(unsigned int c=0; c<vImg.size(); ++c)
    cvReleaseImage(&vImg[c]);
}


void CRPatch::extractFeatureChannelsHOG6(IplImage *img, std::vector<IplImage*>& vImg) {
  // 32 feature channels
  // 7+9 channels: L, a, b, |I_x|, |I_y|, |I_xx|, |I_yy|, HOGlike features with 9 bins (weighted orientations 5x5 neighborhood)
  // 16+16 channels: minfilter + maxfilter on 5x5 neighborhood 

  vImg.resize(32);
  for(unsigned int c=0; c<vImg.size(); ++c)
    vImg[c] = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_8U , 1); 

  // Get intensity
  cvCvtColor( img, vImg[0], CV_RGB2GRAY );

  // Temporary images for computing I_x, I_y (Avoid overflow for cvSobel)
  IplImage* I_x = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_16S, 1); 
  IplImage* I_y = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_16S, 1); 
  
  // |I_x|, |I_y|
  //#pragma omp parallel sections
  {
    //#pragma omp section
    cvSobel(vImg[0],I_x,1,0,3);      

    //#pragma omp section
    cvSobel(vImg[0],I_y,0,1,3);      
  }

  //#pragma omp parallel sections
  {
    //#pragma omp section
    cvConvertScaleAbs( I_x, vImg[3], 0.25);

    //#pragma omp section
    cvConvertScaleAbs( I_y, vImg[4], 0.25);
  
    //#pragma omp section
    {
    short* dataX;
    short* dataY;
    uchar* dataZ;
    int stepX, stepY, stepZ;
    CvSize size;
    int x, y;

    cvGetRawData( I_x, (uchar**)&dataX, &stepX, &size);
    cvGetRawData( I_y, (uchar**)&dataY, &stepY);
    cvGetRawData( vImg[1], (uchar**)&dataZ, &stepZ);
    stepX /= sizeof(dataX[0]);
    stepY /= sizeof(dataY[0]);
    stepZ /= sizeof(dataZ[0]);

    // Orientation of gradients
    for( y = 0; y < size.height; y++, dataX += stepX, dataY += stepY, dataZ += stepZ  )
      for( x = 0; x < size.width; x++ ) {
        // Avoid division by zero
        float tx = (float)dataX[x] + (float) copysign(0.000001f, (float)dataX[x]);
        // Scaling [-pi/2 pi/2] -> [0 80*pi]
        dataZ[x]=uchar( ( atan((float)dataY[x]/tx)+3.14159265f/2.0f ) * 80 ); 
      }
    }

    //#pragma omp section
    {
    short* dataX;
    short* dataY;
    uchar* dataZ;
    int stepX, stepY, stepZ;
    CvSize size;
    int x, y;

    cvGetRawData( I_x, (uchar**)&dataX, &stepX, &size);
    cvGetRawData( I_y, (uchar**)&dataY, &stepY);
    cvGetRawData( vImg[2], (uchar**)&dataZ, &stepZ);
    stepX /= sizeof(dataX[0]);
    stepY /= sizeof(dataY[0]);
    stepZ /= sizeof(dataZ[0]);

    // Magnitude of gradients
    for( y = 0; y < size.height; y++, dataX += stepX, dataY += stepY, dataZ += stepZ  )
      for( x = 0; x < size.width; x++ ) {
        dataZ[x] = (uchar)( sqrt((float)dataX[x]*(float)dataX[x] + (float)dataY[x]*(float)dataY[x]) );
      }
    }
  }

  //#pragma omp parallel sections
  {
    //#pragma omp section
    // 9-bin HOG feature stored at vImg[7] - vImg[15] 
    hog.extractOBin(vImg[1], vImg[2], vImg, 7);

    // |I_xx|, |I_yy|
    //#pragma omp section
    {
    cvSobel(vImg[0],I_x,2,0,3);
    cvConvertScaleAbs( I_x, vImg[5], 0.25);  
    }

    //#pragma omp section
    {
    cvSobel(vImg[0],I_y,0,2,3);
    cvConvertScaleAbs( I_y, vImg[6], 0.25);
    }

    //#pragma omp section
    // L, a, b
    cvCvtColor( img, img, CV_RGB2Lab  );
  }

  cvReleaseImage(&I_x);
  cvReleaseImage(&I_y);  
  
  cvSplit( img, vImg[0], vImg[1], vImg[2], 0);

  
  // min filter
  //#pragma omp parallel for
  for(int c=0; c<16; ++c)
    minfilt(vImg[c], vImg[c+16], 5);

  //max filter
  //#pragma omp parallel for
  for(int c=0; c<16; ++c)
    maxfilt(vImg[c], 5);

  // for debugging only
#if 0
  char buffer[40];
  for(unsigned int i = 0; i<vImg.size();++i) {
    sprintf_s(buffer,"out-%d.png",i);
    cvNamedWindow(buffer,1);
    cvShowImage(buffer, vImg[i]);
    //cvSaveImage( buffer, vImg[i] );
  }

  cvWaitKey();

  for(unsigned int i = 0; i<vImg.size();++i) {
    sprintf_s(buffer,"%d",i);
    cvDestroyWindow(buffer);
  }
#endif
}


void CRPatch::extractFeatureChannelsGrayHOG6(IplImage *img, std::vector<IplImage*>& vImg) {
    //cout << "HOG6" << endl;
    vImg.resize(28);
    for(unsigned int c=0; c<vImg.size(); ++c)
        vImg[c] = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_8U , 1);

    cvCopy(img,vImg[0]);

    IplImage* I_x = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_16S, 1);
    IplImage* I_y = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_16S, 1);

    //double max_val, min_val;

    cvSobel(vImg[0],I_x,1,0,3);
    cvSobel(vImg[0],I_y,0,1,3);

    cvConvertScaleAbs( I_x, vImg[1], 0.25);
    cvConvertScaleAbs( I_y, vImg[2], 0.25);

    short* dataX;
    short* dataY;
    uchar* dataZ;
    int stepX, stepY, stepZ;
    CvSize size;
    int x, y;

    cvGetRawData( I_x, (uchar**)&dataX, &stepX, &size);
    cvGetRawData( I_y, (uchar**)&dataY, &stepY);
    cvGetRawData( vImg[3], (uchar**)&dataZ, &stepZ);
    stepX /= sizeof(dataX[0]);
    stepY /= sizeof(dataY[0]);
    stepZ /= sizeof(dataZ[0]);


    for( y = 0; y < size.height; y++, dataX += stepX, dataY += stepY, dataZ += stepZ  )
        for( x = 0; x < size.width; x++ ) {
            float tx = (float)dataX[x] + (float) copysign(0.000001f, (float)dataX[x]);
            dataZ[x]=uchar( ( atan((float)dataY[x]/tx)+3.14159265f/2.0f) * 80 );
            //cout << (int)uchar( (atan((float)dataY[x]/tx)+3.14159265f/2.0f ) * 80 ) << " ";
        }

    cvGetRawData( I_x, (uchar**)&dataX);
    cvGetRawData( I_y, (uchar**)&dataY);
    cvGetRawData( vImg[4], (uchar**)&dataZ);

    for( y = 0; y < size.height; y++, dataX += stepX, dataY += stepY, dataZ += stepZ  )
        for( x = 0; x < size.width; x++ ) {
            //cout << (int)(uchar)(atan2((float)dataY[x]+0.1f,(float)dataX[x]+0.1f)*160) << " ";
            dataZ[x] = (uchar)( sqrt((float)dataX[x]*(float)dataX[x] + (float)dataY[x]*(float)dataY[x]) );
        }

    hog.extractOBin(vImg[3], vImg[4], vImg, 5);

    cvSobel(vImg[0],I_x,2,0,3);
    cvSobel(vImg[0],I_y,0,2,3);

    cvConvertScaleAbs( I_x, vImg[3], 0.25);
    cvConvertScaleAbs( I_y, vImg[4], 0.25);

    cvReleaseImage(&I_x);
    cvReleaseImage(&I_y);

#if 0
    double min_val, max_val;
    cvMinMaxLoc( vImg[5], &min_val, &max_val);
    cout << "min/max " << min_val << "/" << max_val << " ";
    cvMinMaxLoc( vImg[6], &min_val, &max_val);
    cout << min_val << "/" << max_val << " ";
    cvMinMaxLoc( vImg[7], &min_val, &max_val);
    cout << min_val << "/" << max_val << " ";
    cvMinMaxLoc( vImg[8], &min_val, &max_val);
    cout << min_val << "/" << max_val << " ";
    cvMinMaxLoc( vImg[9], &min_val, &max_val);
    cout << min_val << "/" << max_val << " ";
    cvMinMaxLoc( vImg[10], &min_val, &max_val);
    cout << min_val << "/" << max_val << " ";
    cvMinMaxLoc( vImg[11], &min_val, &max_val);
    cout << min_val << "/" << max_val << " ";
    cvMinMaxLoc( vImg[12], &min_val, &max_val);
    cout << min_val << "/" << max_val << " ";
    cvMinMaxLoc( vImg[13], &min_val, &max_val);
    cout << min_val << "/" << max_val << endl; 
#endif


    for(unsigned int c=0; c<14; ++c)
        minfilt(vImg[c], vImg[c+14], 5);

    for(unsigned int c=0; c<14; ++c)
        maxfilt(vImg[c], 5);

#if 0
    char buffer[40];
    for(unsigned int i = 0; i<vImg.size();++i) {
        sprintf_s(buffer,"out-%d.png",i);
        cvNamedWindow(buffer,1);
        cvShowImage(buffer, vImg[i]);
        //cvSaveImage( buffer, vImg[i] );
    }

    cvWaitKey();

    for(unsigned int i = 0; i<vImg.size();++i) {
        sprintf_s(buffer,"%d",i);
        cvDestroyWindow(buffer);
    }
#endif
}

int mainR() {
  IplImage *im = cvLoadImage("8.png", 1);
  CRPatch p(NULL, im->width, im->height, 0);
  std::vector<IplImage*> vImg;
  p.extractFeatureChannelsHOG6(im, vImg);

  char path[256];
  for(int i = 0; i < 32; i++)
  {
    sprintf(path, "8_%d.png", i);
    cvSaveImage( path, vImg[i]);
    cvReleaseImage(&vImg[i]);
  }
  
  cvReleaseImage(&im);
  return 0;
}



int mainR2() {
  for(int i = 0; i < 170; i++)
  {
    char path[256];
    sprintf(path, "cardata\\raw\\test\\test-%d.png", i);

    IplImage *im = cvLoadImage(path,-1);
    CRPatch p(NULL, im->width, im->height, 0);
    std::vector<IplImage*> vImg;
    p.extractFeatureChannelsGrayHOG6(im, vImg);
    //p.extractFeatureChannelsHOG6(im, vImg);

    for(int j = 0; j < 28; j++)
    {
      sprintf(path, "cardata\\testJ\\%03d_%d.png", i+1, j);
      cvSaveImage( path, vImg[j]);
      cvReleaseImage(&vImg[j]);
    }
    
    cvReleaseImage(&im);
  }
  return 0;
}


int mainReadTUD() {
  for(int i = 0; i < 201; i++)
  {
    char path[256];
    sprintf(path, "tuddata\\crossing\\DaSide0811-seq6-%03d.png", 1+i);

    double scales[] = { 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

    IplImage *im = cvLoadImage(path,-1);


    for(int s = 0; s < 6; s++)
    {

      IplImage *imS = cvCreateImage(cvSize(int(im->width*scales[s]), int(im->height*scales[s])), IPL_DEPTH_8U, 3);
      cvResize(im, imS);
      CRPatch p(NULL, imS->width, imS->height, 0);
      std::vector<IplImage*> vImg;
      p.extractFeatureChannelsHOG6(imS, vImg);
      //p.extractFeatureChannelsHOG6(im, vImg);

      for(int j = 0; j < 32; j++)
      {
        sprintf(path, "tuddata\\Channels\\crossing%03d_0%d_%d.png", 1+i, int(scales[s]*10), j);
        cvSaveImage( path, vImg[j]);
        cvReleaseImage(&vImg[j]);
      }
      cvReleaseImage(&imS);
    }
    
    cvReleaseImage(&im);
  }
  return 0;
}

int mainReadTUDSingle() {
  for(int i = 0; i < 201; i++)
  {
    char path[256];
    sprintf(path, "tuddata\\crossing\\DaSide0811-seq7-%03d.png", 1+i);

    //double scales[] = { 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

    IplImage *im = cvLoadImage(path,-1);


    //for(int s = 0; s < 1; s++)
    {

      IplImage *imS = cvCreateImage(cvSize(int(0.55*im->width), int(0.55*im->height)), IPL_DEPTH_8U, 3);
      cvResize(im, imS);
      sprintf(path, "tuddata\\Single\\%03d.png", 1+i);
      cvSaveImage( path, imS);

      CRPatch p(NULL, imS->width, imS->height, 0);
      std::vector<IplImage*> vImg;
      p.extractFeatureChannelsHOG6(imS, vImg);
      //p.extractFeatureChannelsHOG6(im, vImg);

      for(int j = 0; j < 32; j++)
      {
        sprintf(path, "tuddata\\Single\\%03d_%d.png", 1+i, j);
        cvSaveImage( path, vImg[j]);
        cvReleaseImage(&vImg[j]);
      }
      cvReleaseImage(&imS);
    }
    
    cvReleaseImage(&im);
  }
  return 0;
}



void CRPatch::maxfilt(IplImage *src, unsigned int width) {

  uchar* s_data;
  int step;
  CvSize size;

  cvGetRawData( src, (uchar**)&s_data, &step, &size );
  step /= sizeof(s_data[0]);

  //#pragma omp parallel for
  for(int  y = 0; y < size.height; y++) {
    maxfilt(s_data+y*step, 1, size.width, width);
  }

  cvGetRawData( src, (uchar**)&s_data);

  //#pragma omp parallel for
  for(int  x = 0; x < size.width; x++)
    maxfilt(s_data+x, step, size.height, width);

}

void CRPatch::maxfilt(IplImage *src, IplImage *dst, unsigned int width) {

  uchar* s_data;
  uchar* d_data;
  int step;
  CvSize size;

  cvGetRawData( src, (uchar**)&s_data, &step, &size );
  cvGetRawData( dst, (uchar**)&d_data, &step, &size );
  step /= sizeof(s_data[0]);

  //#pragma omp parallel for
  for(int  y = 0; y < size.height; y++)
    maxfilt(s_data+y*step, d_data+y*step, 1, size.width, width);

  cvGetRawData( src, (uchar**)&d_data);

  //#pragma omp parallel for
  for(int  x = 0; x < size.width; x++)
    maxfilt(d_data+x, step, size.height, width);

}

void CRPatch::minfilt(IplImage *src, unsigned int width) {

  uchar* s_data;
  int step;
  CvSize size;

  cvGetRawData( src, (uchar**)&s_data, &step, &size );
  step /= sizeof(s_data[0]);

  //#pragma omp parallel for
  for(int  y = 0; y < size.height; y++)
    minfilt(s_data+y*step, 1, size.width, width);

  cvGetRawData( src, (uchar**)&s_data);

  //#pragma omp parallel for
  for(int  x = 0; x < size.width; x++)
    minfilt(s_data+x, step, size.height, width);

}

void CRPatch::minfilt(IplImage *src, IplImage *dst, unsigned int width) 
{
  uchar* s_data;
  uchar* d_data;
  int step;
  CvSize size;

  cvGetRawData( src, (uchar**)&s_data, &step, &size );
  cvGetRawData( dst, (uchar**)&d_data, &step, &size );
  step /= sizeof(s_data[0]);

  //#pragma omp parallel for
  for(int  y = 0; y < size.height; y++)
    minfilt(s_data+y*step, d_data+y*step, 1, size.width, width);

  cvGetRawData( src, (uchar**)&d_data);

  //#pragma omp parallel for
  for(int  x = 0; x < size.width; x++)
    minfilt(d_data+x, step, size.height, width);

}


void CRPatch::maxfilt(uchar* data, uchar* maxvalues, unsigned int step, unsigned int size, unsigned int width) {

  unsigned int d = int((width+1)/2)*step; 
  size *= step;
  width *= step;

  maxvalues[0] = data[0];
  for(unsigned int i=0; i < d-step; i+=step) {
    for(unsigned int k=i; k<d+i; k+=step) {
      if(data[k]>maxvalues[i]) maxvalues[i] = data[k];
    }
    maxvalues[i+step] = maxvalues[i];
  }

  maxvalues[size-step] = data[size-step];
  for(unsigned int i=size-step; i > size-d; i-=step) {
    for(unsigned int k=i; k>i-d; k-=step) {
      if(data[k]>maxvalues[i]) maxvalues[i] = data[k];
    }
    maxvalues[i-step] = maxvalues[i];
  }

    deque<int> maxfifo;
    for(unsigned int i = step; i < size; i+=step) {
    if(i >= width) {
      maxvalues[i-d] = data[maxfifo.size()>0 ? maxfifo.front(): i-step];
    }
    
    if(data[i] < data[i-step]) { 

      maxfifo.push_back(i-step);
      if(i==  width+maxfifo.front()) 
        maxfifo.pop_front();

    } else {

      while(maxfifo.size() > 0) {
        if(data[i] <= data[maxfifo.back()]) {
          if(i==  width+maxfifo.front()) 
            maxfifo.pop_front();
        break;
        }
        maxfifo.pop_back();
      }

    }

    }  

    maxvalues[size-d] = data[maxfifo.size()>0 ? maxfifo.front():size-step];
 
}

void CRPatch::maxfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width) {

  unsigned int d = int((width+1)/2)*step; 
  size *= step;
  width *= step;

  deque<uchar> tmp;

  tmp.push_back(data[0]);
  for(unsigned int k=step; k<d; k+=step) {
    if(data[k]>tmp.back()) tmp.back() = data[k];
  }

  for(unsigned int i=step; i < d-step; i+=step) {
    tmp.push_back(tmp.back());
    if(data[i+d-step]>tmp.back()) tmp.back() = data[i+d-step];
  }


    deque<int> minfifo;
    for(unsigned int i = step; i < size; i+=step) {
    if(i >= width) {
      tmp.push_back(data[minfifo.size()>0 ? minfifo.front(): i-step]);
      data[i-width] = tmp.front();
      tmp.pop_front();
    }
    
    if(data[i] < data[i-step]) { 

      minfifo.push_back(i-step);
      if(i==  width+minfifo.front()) 
        minfifo.pop_front();

    } else {

      while(minfifo.size() > 0) {
        if(data[i] <= data[minfifo.back()]) {
          if(i==  width+minfifo.front()) 
            minfifo.pop_front();
        break;
        }
        minfifo.pop_back();
      }

    }

    }  

  tmp.push_back(data[minfifo.size()>0 ? minfifo.front():size-step]);
  
  for(unsigned int k=size-step-step; k>=size-d; k-=step) {
    if(data[k]>data[size-step]) data[size-step] = data[k];
  }

  for(unsigned int i=size-step-step; i >= size-d; i-=step) {
    data[i] = data[i+step];
    if(data[i-d+step]>data[i]) data[i] = data[i-d+step];
  }

  for(unsigned int i=size-width; i<=size-d; i+=step) {
    data[i] = tmp.front();
    tmp.pop_front();
  }
 
}

void CRPatch::minfilt(uchar* data, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width) {

  unsigned int d = int((width+1)/2)*step; 
  size *= step;
  width *= step;

  minvalues[0] = data[0];
  for(unsigned int i=0; i < d-step; i+=step) {
    for(unsigned int k=i; k<d+i; k+=step) {
      if(data[k]<minvalues[i]) minvalues[i] = data[k];
    }
    minvalues[i+step] = minvalues[i];
  }

  minvalues[size-step] = data[size-step];
  for(unsigned int i=size-step; i > size-d; i-=step) {
    for(unsigned int k=i; k>i-d; k-=step) {
      if(data[k]<minvalues[i]) minvalues[i] = data[k];
    }
    minvalues[i-step] = minvalues[i];
  }

    deque<int> minfifo;
    for(unsigned int i = step; i < size; i+=step) {
    if(i >= width) {
      minvalues[i-d] = data[minfifo.size()>0 ? minfifo.front(): i-step];
    }
    
    if(data[i] > data[i-step]) { 

      minfifo.push_back(i-step);
      if(i==  width+minfifo.front()) 
        minfifo.pop_front();

    } else {

      while(minfifo.size() > 0) {
        if(data[i] >= data[minfifo.back()]) {
          if(i==  width+minfifo.front()) 
            minfifo.pop_front();
        break;
        }
        minfifo.pop_back();
      }

    }

    }  

    minvalues[size-d] = data[minfifo.size()>0 ? minfifo.front():size-step];
 
}

void CRPatch::minfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width) {

  unsigned int d = int((width+1)/2)*step; 
  size *= step;
  width *= step;

  deque<uchar> tmp;

  tmp.push_back(data[0]);
  for(unsigned int k=step; k<d; k+=step) {
    if(data[k]<tmp.back()) tmp.back() = data[k];
  }

  for(unsigned int i=step; i < d-step; i+=step) {
    tmp.push_back(tmp.back());
    if(data[i+d-step]<tmp.back()) tmp.back() = data[i+d-step];
  }


    deque<int> minfifo;
    for(unsigned int i = step; i < size; i+=step) {
    if(i >= width) {
      tmp.push_back(data[minfifo.size()>0 ? minfifo.front(): i-step]);
      data[i-width] = tmp.front();
      tmp.pop_front();
    }
    
    if(data[i] > data[i-step]) { 

      minfifo.push_back(i-step);
      if(i==  width+minfifo.front()) 
        minfifo.pop_front();

    } else {

      while(minfifo.size() > 0) {
        if(data[i] >= data[minfifo.back()]) {
          if(i==  width+minfifo.front()) 
            minfifo.pop_front();
        break;
        }
        minfifo.pop_back();
      }

    }

    }  

  tmp.push_back(data[minfifo.size()>0 ? minfifo.front():size-step]);
  
  for(unsigned int k=size-step-step; k>=size-d; k-=step) {
    if(data[k]<data[size-step]) data[size-step] = data[k];
  }

  for(unsigned int i=size-step-step; i >= size-d; i-=step) {
    data[i] = data[i+step];
    if(data[i-d+step]<data[i]) data[i] = data[i-d+step];
  }
 
  for(unsigned int i=size-width; i<=size-d; i+=step) {
    data[i] = tmp.front();
    tmp.pop_front();
  }
}

void CRPatch::maxminfilt(uchar* data, uchar* maxvalues, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width) {

  unsigned int d = int((width+1)/2)*step; 
  size *= step;
  width *= step;

  maxvalues[0] = data[0];
  minvalues[0] = data[0];
  for(unsigned int i=0; i < d-step; i+=step) {
    for(unsigned int k=i; k<d+i; k+=step) {
      if(data[k]>maxvalues[i]) maxvalues[i] = data[k];
      if(data[k]<minvalues[i]) minvalues[i] = data[k];
    }
    maxvalues[i+step] = maxvalues[i];
    minvalues[i+step] = minvalues[i];
  }

  maxvalues[size-step] = data[size-step];
  minvalues[size-step] = data[size-step];
  for(unsigned int i=size-step; i > size-d; i-=step) {
    for(unsigned int k=i; k>i-d; k-=step) {
      if(data[k]>maxvalues[i]) maxvalues[i] = data[k];
      if(data[k]<minvalues[i]) minvalues[i] = data[k];
    }
    maxvalues[i-step] = maxvalues[i];
    minvalues[i-step] = minvalues[i];
  }

    deque<int> maxfifo, minfifo;
  //int comp = 0;
    for(unsigned int i = step; i < size; i+=step) {
    if(i >= width) {
      maxvalues[i-d] = data[maxfifo.size()>0 ? maxfifo.front(): i-step];
      minvalues[i-d] = data[minfifo.size()>0 ? minfifo.front(): i-step];
    }
    
    if(data[i] > data[i-step]) { 

      minfifo.push_back(i-step);
      if(i==  width+minfifo.front()) 
        minfifo.pop_front();
      while(maxfifo.size() > 0) {
        if(data[i] <= data[maxfifo.back()]) {
          if (i==  width+maxfifo.front()) 
            maxfifo.pop_front();
          break;
        }
        maxfifo.pop_back();
      }

    } else {

      maxfifo.push_back(i-step);
      if (i==  width+maxfifo.front()) 
        maxfifo.pop_front();
      while(minfifo.size() > 0) {
        if(data[i] >= data[minfifo.back()]) {
          if(i==  width+minfifo.front()) 
            minfifo.pop_front();
        break;
        }
        minfifo.pop_back();
      }

    }

    }  

    maxvalues[size-d] = data[maxfifo.size()>0 ? maxfifo.front():size-step];
  minvalues[size-d] = data[minfifo.size()>0 ? minfifo.front():size-step];
 
}
