#pragma once

#include "ImageUtils.h"
#include "CRPatch.h"
#include "stdio.h"
#include "vector"
#include "cxcore.h"
#include "cv.h"

// image format used for object detection
struct MultiImage 
{
  int width, width_; // original and extended width of the image
  int height, height_; // original and extended height of the image
  int nChannels; // number of channels for image features

  int *firstChannel; // pointer to the first channel
  int *secondChannel; // pointer to the second channel
  std::vector <int*> allChannels; // vector of pointers to all image channels

  int *integral; // integral image

  IplImage *cvImage; // original OpenCV IplImage
    
  MultiImage(): cvImage(NULL), firstChannel(NULL), secondChannel(NULL), integral(NULL), nChannels(0) {}


  // =======================================================================================================
  // loads original image, resizes it by given koef, and calculates all feature channels
  void LoadMultiImageHog32(const char *fname, double koef = 1.0)
  {
    char path[256];
    sprintf(path, "%s", fname);

    IplImage *orig_im = cvLoadImage(path,-1);
      
    if (!orig_im)
    {
      std::cerr << "Cannot load MultiImage!" << std::endl;
      exit(0);
    }

    cvImage = cvCreateImage(cvSize(int(koef*orig_im->width), int(koef*orig_im->height)), IPL_DEPTH_8U, 3);
    cvResize(orig_im, cvImage);    
    cvReleaseImage(&orig_im);

    nChannels = 32;
    allChannels.resize(nChannels);

    std::vector<IplImage*> vImg;

    CRPatch p(NULL, cvImage->width, cvImage->height, 0);    

    IplImage *cvImageCopy = cvCloneImage(cvImage);
    p.extractFeatureChannelsHOG6(cvImageCopy, vImg);
    cvReleaseImage(&cvImageCopy);

    ConvertIplToImageOneChannel<int>(vImg[0], &firstChannel, width, height);
    ConvertIplToImageOneChannel<int>(vImg[1], &secondChannel, width, height);
        
    width_ = width+1;
    height_ = height+1;
    integral = new int[width_*height_*nChannels];

    for(int j = 0; j < nChannels; j++)
    {
      int *cur_im;

      ConvertIplToImageOneChannel<int>(vImg[j], &cur_im, width, height);

      if(!cur_im) continue;
      int offset = j*width_*height_;
      for(int x = 0; x < width_; x++)
        integral[offset+x] = 0;
      for(int y = 0; y < height_; y++)
        integral[offset+y*width_] = 0;
      for(int y = 1; y < height_; y++)
        for(int x = 1; x < width_; x++)
        {
          integral[offset+y*width_+x] = cur_im[x-1+(y-1)*width]-integral[offset+(y-1)*width_+x-1]+
            integral[offset+(y-1)*width_+x]+integral[offset+y*width_+x-1];
        }
      allChannels[j] = cur_im;
    }
    
    for(int j = 0; j < nChannels; j++)
    {      
      cvReleaseImage(&vImg[j]);
    }    
  }


  // =======================================================================================================
  // loads MultiImage format from file
  void LoadMultiImage(const char *fname, int n_Channels)
  {
    char path[256];

    sprintf(path, "%s.jpg", fname);
    cvImage = cvLoadImage(path);

    nChannels = n_Channels;
    sprintf(path, "%s_%d.png", fname, 0);
    firstChannel = LoadImage8bpp<int>(path, width, height);
    sprintf(path, "%s_%d.png", fname, 1);
    secondChannel = LoadImage8bpp<int>(path, width, height);

    if(!secondChannel)
      throw "The second channel is absent. Cannot load MultiImage!";

    width_ = width+1;
    height_ = height+1;
    integral = new int[width_*height_*nChannels];

    allChannels.resize(nChannels);

    for(int i = 0; i < nChannels; i++)
    {
      int *im;
      sprintf(path, "%s_%d.png", fname, i);
      im = LoadImage8bpp<int>(path, width, height);
      if(!im) continue;
      int offset = i*width_*height_;
      for(int x = 0; x < width_; x++)
        integral[offset+x] = 0;
      for(int y = 0; y < height_; y++)
        integral[offset+y*width_] = 0;
      for(int y = 1; y < height_; y++)
        for(int x = 1; x < width_; x++)
        {
          integral[offset+y*width_+x] = im[x-1+(y-1)*width]-integral[offset+(y-1)*width_+x-1]+
            integral[offset+(y-1)*width_+x]+integral[offset+y*width_+x-1];
        }

      allChannels[i] = im;
    }
  }
  
  ~MultiImage() 
  {
    if(firstChannel) delete []firstChannel;
    if(secondChannel) delete []secondChannel;
    if(integral) delete []integral;
    for (int i = 0; i <nChannels; i ++)
    {
      delete []allChannels[i];
    }
    cvReleaseImage(&cvImage);
  }

  // =======================================================================================================
  // calculates sum of a rectangle at channel nChannel
  int GetBoxSum(int left, int top, int right, int bottom, int nChannel) 
  {
    int offset = nChannel*width_*height_;
    return integral[offset+(bottom+1)*width_+right+1]+integral[offset+top*width_+left]-
      integral[offset+top*width_+right+1]-integral[offset+(bottom+1)*width_+left];
  }
};

