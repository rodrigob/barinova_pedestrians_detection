#include "Vote.h"

// ******************************************************************************
CVote::CVote()
{
  nSamples = 0;
  nFirst = 0;
  blurRadius = 0;
}

// ******************************************************************************
CVote::~CVote()
{    
}

// ******************************************************************************
void CVote::ResetVote() 
{
  nSamples = 0;
  offsets.clear();
  nFirst = 0;
}

// ******************************************************************************
void CVote::UpdateVote(HoughSample *hs)  
{
  if(hs->isObject)
  {
    offsets.push_back(hs->off);
    nFirst ++;
  }
  nSamples++;
}

// ******************************************************************************
void CVote::WriteToFile(FILE *out)  
{
  fwrite( (void *)&nSamples, sizeof(int), 1, out);
  int nOffsets = offsets.size();
  fwrite( (void *)&nOffsets, sizeof(int), 1, out);


  for(int i = 0; i < nOffsets; i++)
  {
    int offsetx = offsets[i].x;
    int offsety = offsets[i].y;
    fwrite( (void *)&offsetx, sizeof(int), 1, out);
    fwrite( (void *)&offsety, sizeof(int), 1, out);
  }
}

// ******************************************************************************
void CVote::ReadFromFile(FILE *in)  
{
  fread( (void *)&nSamples, sizeof(int), 1, in);
  int nOffsets;
  fread( (void *)&nOffsets, sizeof(int), 1, in);
  offsets.clear();
  nFirst = nOffsets;
  offsets.reserve(nOffsets);
  for(int i = 0; i < nOffsets; i++)
  {
    CvPoint2D64f off;
    int dx, dy;

    fread( (void *)&dx, sizeof(int), 1, in);    
    fread( (void *)&dy, sizeof(int), 1, in);

    off.x = dx;
    off.y = dy;
    offsets.push_back(off);
  }
}



// ******************************************************************************
void CVote::SetBlurRadius(double in_dBlurRadius)
{
  blurRadius = in_dBlurRadius;
}