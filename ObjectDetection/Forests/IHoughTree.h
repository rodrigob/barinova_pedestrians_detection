#pragma once

#include "../ImageWrapper/MultiImage.h"
#include "Vote.h"
#include <vector>
#include "../Utils/math_functions.h"
#include "stdio.h"
#include "RandomTree.h"
#include "BoxTest.h"
#include "IHoughTree.h"

const int CELL_HOUGH_NTESTS = 3000;


template <typename Vote> class IHoughTree: public RandomTree<HoughSample, BoxTest, typename Vote>
{
public:
  IHoughTree(){};
  virtual ~IHoughTree(){};

  virtual double ScoreSplit(int nLeftSamples, int *leftSamples, int nRightSamples, int *rightSamples, BoxTest *test) = 0;
  
  int patchWidth, patchHeight;
  bool scoreByOffsets;

  

  ///*********************************************************************************************************************
  virtual void DrawSamples()
  {
  }

  ///*********************************************************************************************************************
  void AggregateVote(int nNodeSamples, int *nodeSamples, Vote *vote) 
  {
    vote->ResetVote();
    for(int i = 0; i < nNodeSamples; i++)
      vote->UpdateVote(samples+nodeSamples[i]);
  }

  ///*********************************************************************************************************************
  void DrawTests(int& nTests, BoxTest **tests, int nNodeSamples, int *nodeSamples) 
  {
    int nObject = 0;


    for(int i = 0; i < nNodeSamples; i++)
      if(samples[nodeSamples[i]].isObject)
        nObject++;

    if(nObject == 0  || nObject == nNodeSamples )// || nNodeSamples < 20)
    {
      nTests = 0;
      *tests = NULL;
      return;
    }

    nTests = CELL_HOUGH_NTESTS;
    *tests = new BoxTest[nTests];

    scoreByOffsets = (rand() % 2 > 0) ? true : false;

    for(int i = 0; i < nTests; i++)
    {  
      BoxTest &curTest = (*tests)[i];

      int width = 1;//(rand()%(patchWidth/2))+1;
      int height = 1;//(rand()%(patchHeight/2))+1;

      //int width = int(patchWidth/16.0 + 1.0);
      //int height = int(patchHeight/16.0 + 1.0);

      //int width = (rand()%(patchWidth/2));
      //int height = (rand()%(patchHeight/2));


      curTest.l1 = rand()%(patchWidth-width);
      curTest.l2 = rand()%(patchWidth-width);
      curTest.t1 = rand()%(patchHeight-height);
      curTest.t2 = rand()%(patchHeight-height);
      curTest.r1 = curTest.l1+width;
      curTest.r2 = curTest.l2+width;
      curTest.b1 = curTest.t1+height;
      curTest.b2 = curTest.t2+height;
      curTest.channel = (rand()%(samples[0].im->nChannels-1));

      HoughSample *pos_sample, *neg_sample;

      int pos_sample_idx, neg_sample_idx;
      pos_sample_idx = rand() %nNodeSamples;
      while(!samples[nodeSamples[pos_sample_idx]].isObject)
      {
        pos_sample_idx++;
        if(pos_sample_idx == nNodeSamples)
          pos_sample_idx = 0;
      }
      neg_sample_idx = rand() %nNodeSamples;
      while(samples[nodeSamples[neg_sample_idx]].isObject || pos_sample_idx == neg_sample_idx)
      {
        neg_sample_idx++;
        if(neg_sample_idx == nNodeSamples)
          neg_sample_idx = 0;
      }
      pos_sample = samples + nodeSamples[pos_sample_idx];
      neg_sample = samples + nodeSamples[neg_sample_idx];

      ///////// calculate first boxsum for picked positive sample ///////
      int pos_sample_left_testbox1 = curTest.l1 + pos_sample->left;
      int pos_sample_top_testbox1 = curTest.t1 + pos_sample->top;
      int pos_sample_right_testbox1 = curTest.r1 + pos_sample->left;
      int pos_sample_bottom_testbox1 = curTest.b1 + pos_sample->top; 

      double box1sum_pos = pos_sample->im->GetBoxSum(pos_sample_left_testbox1, pos_sample_top_testbox1,
        pos_sample_right_testbox1, pos_sample_bottom_testbox1, curTest.channel);
      
      ///////// calculate first boxsum for picked negative sample ///////
      int neg_sample_left_testbox1 = curTest.l1 + neg_sample->left;
      int neg_sample_top_testbox1 = curTest.t1 + neg_sample->top;
      int neg_sample_right_testbox1 = curTest.r1 + neg_sample->left;
      int neg_sample_bottom_testbox1 = curTest.b1 + neg_sample->top;

      double box1sum_neg = neg_sample->im->GetBoxSum(neg_sample_left_testbox1, neg_sample_top_testbox1,
        neg_sample_right_testbox1, neg_sample_bottom_testbox1, curTest.channel);

      ///////// calculate second boxsum for picked positive sample //////
      int pos_sample_left_testbox2 = curTest.l2 + pos_sample->left;
      int pos_sample_top_testbox2 = curTest.t2 + pos_sample->top;
      int pos_sample_right_testbox2 = curTest.r2 + pos_sample->left;
      int pos_sample_bottom_testbox2 = curTest.b2 + pos_sample->top; 

      double box2sum_pos = pos_sample->im->GetBoxSum(pos_sample_left_testbox2, pos_sample_top_testbox2,
        pos_sample_right_testbox2, pos_sample_bottom_testbox2, curTest.channel);
      
      ///////// calculate second boxsum for picked negative sample //////
      int neg_sample_left_testbox2 = curTest.l2 + neg_sample->left;
      int neg_sample_top_testbox2 = curTest.t2 + neg_sample->top;
      int neg_sample_right_testbox2 = curTest.r2 + neg_sample->left;
      int neg_sample_bottom_testbox2 = curTest.b2 + neg_sample->top;

      double box2sum_neg = neg_sample->im->GetBoxSum(neg_sample_left_testbox2, neg_sample_top_testbox2,
        neg_sample_right_testbox2, neg_sample_bottom_testbox2, curTest.channel);


      double test_value_pos = box1sum_pos - box2sum_pos;
      double test_value_neg = box1sum_neg - box2sum_neg;

      curTest.tau = (test_value_pos  + test_value_neg)/2.0;
    }
  }
};