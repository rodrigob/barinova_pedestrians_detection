#pragma once

#include "IHoughTree.h"
#include "Vote.h"


class CHoughTree: public IHoughTree <CVote>
{
public:
  CHoughTree(){};
  virtual ~CHoughTree(){};

  virtual double ScoreSplit(int nLeftSamples, int *leftSamples, int nRightSamples, int *rightSamples, BoxTest *test);
  
  int patchWidth, patchHeight;
  bool scoreByOffsets;
};
