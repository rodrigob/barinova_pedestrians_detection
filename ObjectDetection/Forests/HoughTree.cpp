#include "HoughTree.h"


///*********************************************************************************************************************
double CHoughTree::ScoreSplit(int nLeftSamples, int *leftSamples, int nRightSamples, int *rightSamples, BoxTest *test)
{
  if(nLeftSamples <= 1 || nRightSamples <= 1)
    return BAD_SPLIT;

  int weights[] = {nLeftSamples, nRightSamples};
  CVote votes[2]; // stores distribution of samples in positive and negative regions of the space
  AggregateVote(nLeftSamples, leftSamples, votes);
  AggregateVote(nRightSamples, rightSamples, votes+1);

  double score = 0;

  //scoreByOffsets = true;

  if(scoreByOffsets)
  {
    if(votes[0].offsets.size() <= 1 || votes[1].offsets.size() <= 1)
      return BAD_SPLIT;
      
    for(int j = 0; j < 2; j++)
    {
      /* find the point which is closest to its center among samples in a region */
      double minDist = 1e8;

      for(unsigned int i = 0; i < votes[j].offsets.size(); i++)
      {
        double cur_dx = votes[j].offsets[i].x;
        double cur_dy = votes[j].offsets[i].y;
        double cur_dist = sqrt(cur_dx*cur_dx + cur_dy*cur_dy);

        if (cur_dist < minDist)
        {
          minDist = cur_dist;
        }
      }

      /* apply tranformation to all offsets */
      
      std::vector <double> newDX(votes[j].offsets.size());
      std::vector <double> newDY(votes[j].offsets.size());

      for(unsigned int i = 0; i < votes[j].offsets.size(); i++)
      {
        newDX[i] = votes[j].offsets[i].x / (exp(log(1.0 + minDist)*1.2));
        newDY[i] = votes[j].offsets[i].y / (exp(log(1.0 + minDist)*1.2));
      }

      double mean_dx = 0;
      double mean_dy = 0;

      /* find center of the distribution of transformed points */
      for(unsigned int i = 0; i < votes[j].offsets.size(); i++)
      {
        mean_dx += newDX[i];
        mean_dy += newDY[i];
      }
      mean_dx /= votes[j].offsets.size();
      mean_dy /= votes[j].offsets.size();

      /* calculate standard deviation */
      for(unsigned int i = 0; i < votes[j].offsets.size(); i++)
      {
        double d;
        d = newDX[i] - mean_dx;
        score -= d*d;
        d = newDY[i] - mean_dy;
        score -= d*d;
      }        
    }
  }
  else {
    for(int j = 0; j < 2; j++)
    {
      double p = double(votes[j].offsets.size())/votes[j].nSamples;
      score += (p*log(p+1e-10)+(1-p)*log(1-p+1e-10))*weights[j];
    }
  }
  return score;
}

