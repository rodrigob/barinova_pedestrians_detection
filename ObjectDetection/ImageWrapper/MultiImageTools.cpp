#include "MultiImage.h"


// =======================================================================================================
void GetResizedMultiImage(const MultiImage& in_OrigMultiImage, 
              unsigned int in_uiResizedWidth, unsigned int in_uiResizedHeight, 
              MultiImage &out_ResizedMultiImage)
{
  out_ResizedMultiImage.nChannels = in_OrigMultiImage.nChannels;
  out_ResizedMultiImage.firstChannel = NULL;
  out_ResizedMultiImage.secondChannel = NULL;
  out_ResizedMultiImage.allChannels.resize(out_ResizedMultiImage.nChannels, NULL);

  out_ResizedMultiImage.width = in_uiResizedWidth;
  out_ResizedMultiImage.height = in_uiResizedHeight;

  out_ResizedMultiImage.width_ = out_ResizedMultiImage.width+1;
  out_ResizedMultiImage.height_ = out_ResizedMultiImage.height+1;
  
  ResizeImageOneChannel(in_OrigMultiImage.firstChannel, in_OrigMultiImage.width, in_OrigMultiImage.height, 
    in_uiResizedWidth, in_uiResizedHeight, &out_ResizedMultiImage.firstChannel);

  ResizeImageOneChannel(in_OrigMultiImage.secondChannel, in_OrigMultiImage.width, in_OrigMultiImage.height, 
    in_uiResizedWidth, in_uiResizedHeight, &out_ResizedMultiImage.secondChannel);

  cvReleaseImage(&out_ResizedMultiImage.cvImage);
  out_ResizedMultiImage.cvImage = cvCreateImage(cvSize(in_uiResizedWidth, in_uiResizedHeight), 
    in_OrigMultiImage.cvImage->depth, in_OrigMultiImage.cvImage->nChannels); 
  cvResize(in_OrigMultiImage.cvImage, out_ResizedMultiImage.cvImage);

  out_ResizedMultiImage.integral = new int[out_ResizedMultiImage.width_ * out_ResizedMultiImage.height_ * out_ResizedMultiImage.nChannels];
  int *im_integral = out_ResizedMultiImage.integral;

  for(int i = 0; i < out_ResizedMultiImage.nChannels; i++)
  {
    ResizeImageOneChannel(in_OrigMultiImage.allChannels[i], in_OrigMultiImage.width, in_OrigMultiImage.height, 
      in_uiResizedWidth, in_uiResizedHeight, &( out_ResizedMultiImage.allChannels[i]));

    int *im = out_ResizedMultiImage.allChannels[i];
        
    int im_width = out_ResizedMultiImage.width;
    int im_height = out_ResizedMultiImage.height;

    int im_width_ = out_ResizedMultiImage.width_;
    int im_height_ = out_ResizedMultiImage.height_;
    
    int offset = i * out_ResizedMultiImage.width_ * out_ResizedMultiImage.height_;

    for(int x = 0; x < out_ResizedMultiImage.width_; x++)
      out_ResizedMultiImage.integral[offset+x] = 0;
    for(int y = 0; y < out_ResizedMultiImage.height_; y++)
      out_ResizedMultiImage.integral[offset+y*out_ResizedMultiImage.width_] = 0;

    for(int y = 1; y < out_ResizedMultiImage.height_; y++)
      for(int x = 1; x < out_ResizedMultiImage.width_; x++)
      {
        out_ResizedMultiImage.integral[offset + y*im_width_ + x] = 
          im[x-1 + (y-1)*im_width] -
          im_integral[offset + (y-1) * im_width_ + x-1] +
          im_integral[offset + (y-1) * im_width_ + x] +
          im_integral[offset + y * im_width_ + x-1];
      }
  }
}


// =======================================================================================================
void SaveImageAsText(const char *in_FileName, IplImage *in_Image)
{
  std::ofstream out_file(in_FileName);

  for (int i = 0; i < in_Image->height; i ++)
  {
    for (int j = 0; j < in_Image->width; j ++)
    {
      float cur_element;
      if (in_Image->depth == 32)
      {
        cur_element = ((float*)(in_Image->imageData))[(in_Image->width*i + j)*in_Image->nChannels];
      }
      else
      {
        cur_element = ((unsigned char *)(in_Image->imageData))[(in_Image->width*i + j)*in_Image->nChannels];
      }
      out_file << cur_element << '\t';
    }
    out_file << std::endl;
  }
  out_file.close();
}