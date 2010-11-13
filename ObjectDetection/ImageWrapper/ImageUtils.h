#pragma once

#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <fstream>

// =======================================================================================================
// resizes one channel MultiImage from (in_iOrigWidth,in_iOrigHeight) to (in_iResizedWidth, in_iResizedHeight)
template<class T> void ResizeImageOneChannel(const T* in_OrigImageData, 
                       int in_iOrigWidth, int in_iOrigHeight, 
                       int in_iResizedWidth, int in_iResizedHeight,
                       T** out_ResizedImageData)
{
  IplImage *origIplImage = NULL, *resizedIplImage = NULL;

  ConvertImageToIplOneChannel(in_OrigImageData, in_iOrigWidth, in_iOrigHeight, &origIplImage);
  resizedIplImage = cvCreateImage( cvSize(in_iResizedWidth, in_iResizedHeight), IPL_DEPTH_32F, 1 );  

  cvResize((CvArr*)origIplImage, (CvArr*)resizedIplImage, CV_INTER_CUBIC);
  
  ConvertIplToImageOneChannel(resizedIplImage, out_ResizedImageData, in_iResizedWidth, in_iResizedHeight);
  
  cvReleaseImage(&origIplImage);
  cvReleaseImage(&resizedIplImage);

}

// =======================================================================================================
// converts one-channel MultiImage to OpenCV IplImage
template<class T> void ConvertIplToImageOneChannel(const IplImage *in_IplImage, T** out_ImageData, int &out_Width, int &out_Height)
{
  out_Width = in_IplImage->width;
  out_Height = in_IplImage->height;
  *out_ImageData = new T[out_Width * out_Height];

  for(int y = 0, i = 0; y < out_Height; y++)
  {
    for(int x = 0; x < out_Width; x++, i++)
    {
      if (in_IplImage->depth == 32)
      {
        (*out_ImageData)[i] = (T)(((float*)(in_IplImage->imageData + in_IplImage->widthStep*y))[x] * 255.0);
      }
      else
      {
        (*out_ImageData)[i] = (T)(((uchar*)(in_IplImage->imageData + in_IplImage->widthStep*y))[x]);
      }
    }
  }
}

// =======================================================================================================
// converts OpenCV IplImage to one-channel MultiImage
template<class T> void ConvertImageToIplOneChannel(const T* in_ImageData, int width, int height, IplImage **out_IplImage)
{
  if (*out_IplImage != NULL)
  {
    cvReleaseImage(out_IplImage);
  }

  *out_IplImage = cvCreateImage( cvSize(width, height), IPL_DEPTH_32F, 1 );

  for(int y = 0, i = 0; y < height; y++)
  {
    for(int x = 0; x < width; x++, i++)
    {
      ((float*)(*out_IplImage)->imageData)[(*out_IplImage)->width*y + x ] = in_ImageData[i] / 255.0;
    }
  }
}

// =======================================================================================================
// loads MultiImage with 8 bits per pixel
template<class T> T* LoadImage8bpp(const char *filename, int& width, int& height)
{
  IplImage *im = cvLoadImage(filename, 0);
  if(!im) return NULL;
  
  width = im->width;
  height = im->height;
  T *image = new T[width*height];

  for(int y = 0, i = 0; y < height; y++)
    for(int x = 0; x < width; x++, i++)
      image[i] = (T)(((uchar*)(im->imageData + im->widthStep*y))[x]);

  cvReleaseImage(&im);
  return image;
}

// =======================================================================================================
// loads MultiImage with 24 bits per pixel
template<class T> T* LoadImage24bpp(const char *filename, int& width, int& height)
{
  IplImage *im = cvLoadImage(filename, 1);
  if(!im) return NULL;
  
  width = im->width;
  height = im->height;
  T *image = new T[width*height*3];

  for(int y = 0, i = 0; y < height; y++)
    for(int x = 0; x < width; x++,i++)
      for(int c = 0; c < 3; c++)
        image[3*i+c] = (T)(((uchar*)(im->imageData + im->widthStep*y))[3*x+c]);

  cvReleaseImage(&im);
  return image;
}


//////////////////////////////////
//RGB 2 HSV color space conversion
#define RETURN_RGB(x,y,z) { r = 255.0*x; g = 255.0*y; b = 255.0*z; return; }
#define PI (4*atan(1.0))

// =======================================================================================================
inline void HSV2RGB( double h, double s, double v, double& r, double& g, double& b ) 
{ 
   double m, n, f;  
   int i;  
   
   h /= PI/3;
   i = (int)floor(h);  
   f = h - i;  
   if(!(i & 1)) f = 1 - f; // if i is even  
   m = v * (1 - s);  
   n = v * (1 - s * f);  
   switch (i) {  
    case 0: RETURN_RGB(v, n, m);  
    case 1: RETURN_RGB(n, v, m);  
    case 2: RETURN_RGB(m, v, n)  
    case 3: RETURN_RGB(m, n, v);  
    case 4: RETURN_RGB(n, m, v);  
    case 5: RETURN_RGB(v, m, n);  
   }  
} 

// =======================================================================================================
template<class T> void ShowImage24bpp(T *image, int width, int height, int pause, const char *caption, const char *outFile = NULL, double loupe = -1)
{

  IplImage *out = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);

  for(int y = 0, i =0; y < height; y++)
    for(int x = 0; x < width; x++, i++)
    {
      CvScalar s;
      s.val[0] = image[3*i];
      s.val[1] = image[3*i+1];
      s.val[2] = image[3*i+2];
      //s.val[0] = s.val[1] = s.val[2] = double(image[i]-minVal)/double(maxVal-minVal+0.0001)*255;
      cvSet2D(out, y, x, s);
    }

  if(outFile)
    cvSaveImage(outFile, out);
  else
  {
    cvNamedWindow(caption, CV_WINDOW_AUTOSIZE);
    
    if(loupe < 0)
    {
      cvShowImage(caption, out);
      cvWaitKey(pause);
    }
    else {
      IplImage *out2 = cvCreateImage(cvSize(int(width*loupe), int(height*loupe)), IPL_DEPTH_8U, 3);
      cvResize(out, out2);
      cvShowImage(caption, out2);
      cvWaitKey(pause);
      cvReleaseImage(&out2);
    }
  }

  
  cvReleaseImage(&out);
}

// =======================================================================================================
template<class T> IplImage *ConvertToIplImage8bpp(T *image, int width, int height)
{
  IplImage *out = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);

  for(int y = 0, i =0; y < height; y++)
    for(int x = 0; x < width; x++, i++)
      ((uchar*)(out->imageData + out->widthStep*y))[x] = image[i];
  
  return out;
}

// =======================================================================================================
template<class T> void ShowImage8bpp(const T *image, int width, int height, int pause, const char *caption, const char *outFile = NULL, T minval = 1, T maxval = 0)
{
  T minVal, maxVal;
  if(minval > maxval)
  {
    minVal = image[0];
    maxVal = image[0];

    for(int i = 0; i < width*height; i++)
    {
      if(image[i] < minVal) minVal = image[i];
      if(image[i] > maxVal) maxVal = image[i];
    }
  }
  else
  {
    minVal = minval;
    maxVal = maxval;
  }

  IplImage *out = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);


  printf("Image %s: minVal = %lf, maxVal = %lf\n", caption, double(minVal), double(maxVal));
  for(int y = 0, i =0; y < height; y++)
    for(int x = 0; x < width; x++, i++)
    {
      CvScalar s;
      HSV2RGB(2*3.14*(0.7*double(image[i]-minVal)/double(maxVal-minVal+0.0001)),1,1, s.val[0],s.val[1],s.val[2]);
      s.val[0] = s.val[1] = s.val[2] = double(image[i]-minVal)/double(maxVal-minVal+0.0001)*255;
      cvSet2D(out, y, x, s);
    }


  if(outFile)
  {
    cvSaveImage(outFile, out);
  }
  else
  {
    cvNamedWindow(caption, CV_WINDOW_AUTOSIZE);

    cvShowImage(caption, out);
    cvWaitKey(pause);
  }

  cvReleaseImage(&out);

}

// =======================================================================================================
template<class T, class U> void ShowImageMask(T *image, U *mask, int width, int height, int pause, const char *caption, const char *outFile = NULL)
{
  T minVal = image[0];
  T maxVal = image[0];

  for(int i = 0; i < width*height; i++)
  {
    if(image[i] < minVal) minVal = image[i];
    if(image[i] > maxVal) maxVal = image[i];
  }

  IplImage *out = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);

  cvNamedWindow(caption, CV_WINDOW_AUTOSIZE);
  

  printf("Image %s: minVal = %lf, maxVal = %lf\n", caption, double(minVal), double(maxVal));
  for(int y = 0, i =0; y < height; y++)
    for(int x = 0; x < width; x++, i++)
    {
      CvScalar s;
      HSV2RGB(4.5, double(mask[i]), double(image[i])/255+mask[i]*0.25, s.val[0],s.val[1],s.val[2]);
      //s.val[0] = s.val[1] = s.val[2] = double(image[i]-minVal)/double(maxVal-minVal+0.0001)*255;
      cvSet2D(out, y, x, s);
    }


  if(outFile)
    cvSaveImage(outFile, out);

  cvShowImage(caption, out);
  cvWaitKey(pause);

  cvReleaseImage(&out);

}

// =======================================================================================================
template<class T, class U> void ShowImageMask24bpp_(T *image, U *mask, int width, int height, int pause, const char *caption, const char *outFile = NULL)
{
  T minVal = 0;
  T maxVal = 255;


  IplImage *out = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);

  cvNamedWindow(caption, CV_WINDOW_AUTOSIZE);
  
  
  printf("Image %s: minVal = %lf, maxVal = %lf\n", caption, double(minVal), double(maxVal));
  for(int y = 0, i =0; y < height; y++)
    for(int x = 0; x < width; x++, i++)
    {
      CvScalar s;
      //double mult = mask[i]? 1.5 : 0.5;
      double mult = mask[i] > 0? 2.0 : 0.25;
      s.val[0] = image[i*3]*mult;
      s.val[1] = image[i*3+1]*mult;
      s.val[2] = image[i*3+2]*mult;;
      cvSet2D(out, y, x, s);
    }


  if(outFile)
    cvSaveImage(outFile, out);

  cvShowImage(caption, out);
  cvWaitKey(pause);

  cvReleaseImage(&out);
}

// =======================================================================================================
template<class T, class U> void Mask2Perim(T *mask, U *perim, int width, int height)
{
  for(int y = 0, i =0; y < height; y++)
    for(int x = 0; x < width; x++, i++)
      perim[i] = (y && mask[i] != mask[i-width] || x && mask[i] != mask[i-1] ||
        y < height-1 && mask[i] != mask[i+width] || x < width-1 && mask[i] != mask[i+1])? 1:0;
}

// =======================================================================================================
template<class T, class U> void DrawSegmentation24bpp(T *im, U *mask, int w, int h)
{
  for(int y = 0, i = 0; y < h; y++)
    for(int x = 0; x < w; x++, i++)
    {
      if(y >= 0  && x >= 0 && x < w && y < h &&
        (y>0 && mask[i] != mask[i-w] || y < h && mask[i] != mask[i+w] ||
        x>0 && mask[i] != mask[i-1] || x < w && mask[i] != mask[i+1] ))
      {
        im[3*i+0] = 0;
        im[3*i+1] = 0;
        im[3*i+2] = 255;
      }
    }
}


