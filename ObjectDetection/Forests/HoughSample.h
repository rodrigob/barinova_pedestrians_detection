#pragma once

#include "../ImageWrapper/MultiImage.h"
#include "cxcore.h"

class HoughSample
{
public:
  MultiImage *im;
  int top, left;
  CvPoint2D64f off;  
  bool isObject;
};