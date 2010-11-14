#pragma once
#include "IVote.h"

class CVote : public IVote
{
public:
  CVote();
  virtual ~CVote();

  void ResetVote();

  void UpdateVote(HoughSample *hs);

  void WriteToFile(FILE *out);
  
  void ReadFromFile(FILE *in);

  void SetBlurRadius(double in_dBlurRadius);
  
  int nSamples;
  int nFirst;
  std::vector<CvPoint2D64f> offsets;
  double blurRadius;

}; // end of class CVote
