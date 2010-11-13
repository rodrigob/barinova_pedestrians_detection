#include "GreedyDetection.h"
#include "../ImageWrapper/ImageUtils.h"
#include "iostream"


// =======================================================================================================
float CGreedyDetection::ConvertToScale0(float in_I, int in_iCurScale)
{
 float term = 1.0;
 for (int iScale = 0; iScale < in_iCurScale; iScale++)
 {
  term /= m_fResizeCoef;
 }

 return in_I*term;
}


// =======================================================================================================
float CGreedyDetection::ConvertFromScale0(float in_I, int in_iDestScale)
{
 float term = 1.0;
 for (int iScale = 0; iScale < in_iDestScale; iScale++)
 {
  term *= m_fResizeCoef;
 }

 return in_I*term;
}


// =======================================================================================================
CGreedyDetection::CGreedyDetection()
{
 CurrentGivenCosts = NULL;
 DeltaCosts = NULL;
 BgCostMap = NULL;

 m_vecResizedGivenCosts.clear();
 m_vecResizedGivenCosts.clear();
 m_vecResizedDeltaCosts.clear();
}



// =======================================================================================================
CGreedyDetection::~CGreedyDetection()
{
 for (unsigned int i = 0; i < m_vecAccumulators.size(); i ++)
 {
  if (m_vecAccumulators[i] != NULL) cvReleaseImage(&m_vecAccumulators[i]);
 }

 for (unsigned int i = 0; i < m_vecResizedGivenCosts.size(); i ++)
 {
  if (m_vecResizedGivenCosts[i] != NULL) cvReleaseImage(&m_vecResizedGivenCosts[i]);
 }

 for (unsigned int i = 0; i < m_vecResizedDeltaCosts.size(); i ++)
 {
  if (m_vecResizedDeltaCosts[i] != NULL) cvReleaseImage(&m_vecResizedDeltaCosts[i]);
 }

 cvReleaseImage(&CurrentGivenCosts);
 cvReleaseImage(&BgCostMap);
 cvReleaseImage(&DeltaCosts);
}


// =======================================================================================================
void CGreedyDetection::SetForest( const char *in_forest_path, 
                 float in_fPatchSize, float in_blur_radius,
                 float in_fBgThresh, float in_fProbVoteThreshold,
                 float in_BBoxWidth, float in_BBoxHeight,  
                 float in_HalfMaxBBoxWidth, float in_HalfMaxBBoxHeight, 
                 int in_iNumberOfScales, float in_fResizeCoef )
{
 uForest.getForestFromFile(in_forest_path, in_fPatchSize);

 uForest.setBgThresh(in_fBgThresh);
 uForest.setScalingParameters(in_iNumberOfScales, in_fResizeCoef);

 m_fProbVoteThreshold = in_fProbVoteThreshold;

 BBoxWidth = in_BBoxWidth;
 BBoxHeight = in_BBoxHeight;

 HalfMaxBBoxWidth = in_HalfMaxBBoxWidth;
 HalfMaxBBoxHeight = in_HalfMaxBBoxHeight;

 m_iNumberOfScales = in_iNumberOfScales;
 m_fResizeCoef = in_fResizeCoef;

 m_blur_radius = cvRound(in_blur_radius);
}



// =======================================================================================================
void CGreedyDetection::updateAccumulator(bool init)
{
 for (int iScale = 0; iScale < m_iNumberOfScales; iScale++)
 {
  updateAccumulatorScale(iScale, init);
 }
}

// =======================================================================================================
float CGreedyDetection::getPatchGivenCost(int iScale, int i, int j)
{
 float patchGivenCost = CV_IMAGE_ELEM(m_vecResizedGivenCosts[iScale], float, i, j);

 return patchGivenCost;
}


// =======================================================================================================
float CGreedyDetection::getPatchDeltaCost(int iScale, int i, int j)
{
 float patchDeltaCost = CV_IMAGE_ELEM(m_vecResizedDeltaCosts[iScale], float, i, j);

 return patchDeltaCost;
}

// =======================================================================================================
void CGreedyDetection::updateCostsOfOnePatch(IplImage *in_ProbMap, 
                        int in_iScaledPatchI, 
                        int in_iScaledPatchJ, 
                        float in_fPatchGivenCost, 
                        float in_fPatchDeltaCost, 
                        float in_fPatchBgCost,
                        IplImage *io_AccumMap, 
                        bool init, int iScale,
                        int firstScaledRow, int lastScaledRow,
                        int firstScaledCol, int lastScaledCol)
{
 int curHeight = in_ProbMap->height;
 int curWidth = in_ProbMap->width;
 
 int forestPatchSize = uForest.getPatchSize();


 if (init)
 {
  for (int ii = firstScaledRow; ii < lastScaledRow; ii ++)
  {
   float *probMapPoint = (float*)(&(CV_IMAGE_ELEM( in_ProbMap, float, ii, firstScaledCol )));  
   float *accumPoint = (float*)(&(CV_IMAGE_ELEM( io_AccumMap, float, ii, firstScaledCol )));

   for (int jj = firstScaledCol; jj < lastScaledCol; jj ++)
   {
    // update accumulator value at scale iScale

    float prob = *probMapPoint;

    if (prob > m_fProbVoteThreshold)
    {
     float voteForPoint = safeLog (prob);
     float costForPoint = (-voteForPoint) - in_fPatchBgCost;
     float cur_cost = __min(0.0f , costForPoint - in_fPatchGivenCost);
     *accumPoint += cur_cost;
    }

    probMapPoint ++;
    accumPoint ++;
   }
  }
 }
 else
 {
   
  float prevGivenCost = in_fPatchGivenCost - in_fPatchDeltaCost; 

  for (int ii = firstScaledRow; ii < lastScaledRow; ii ++)
  {
   float *probMapPoint = (float*)(&(CV_IMAGE_ELEM( in_ProbMap, float, ii, firstScaledCol )));  
   float *accumPoint = (float*)(&(CV_IMAGE_ELEM( io_AccumMap, float, ii, firstScaledCol )));

   for (int jj = firstScaledCol; jj < lastScaledCol; jj ++)
   {
    // update accumulator value at scale iScale
    float prob = *probMapPoint;

    if (prob > m_fProbVoteThreshold)
    {
     float voteForPoint = safeLog (prob);
     float costForPoint = (-voteForPoint) - in_fPatchBgCost;


     float cur_cost = __min(0, costForPoint-in_fPatchGivenCost);
     float prev_cost = __min(0, costForPoint-prevGivenCost);

     *accumPoint += ( cur_cost - prev_cost );
    }

    probMapPoint ++;
    accumPoint ++;
   }
  }
 }
}

// =======================================================================================================
void CGreedyDetection::updateAccumulatorScale(int iScale, bool init)
{
 IplImage *accumMap = m_vecAccumulators[iScale];

 int firstRow;
 int firstCol;
 int lastRow;
 int lastCol;

 if (init)
 {
    firstRow = 0;
    firstCol = 0;
    lastRow = height;
    lastCol = width;
 }
 else
 {
    int lastCenterI = m_vecDetections[m_vecDetections.size()-1].y;
    int lastCenterJ = m_vecDetections[m_vecDetections.size()-1].x;
    int lastScale = m_vecDetScales[m_vecDetections.size()-1];

    int firstScaledRow = lastCenterI - HalfMaxBBoxHeight;
    int firstScaledCol = lastCenterJ - HalfMaxBBoxWidth;
    int lastScaledRow = lastCenterI + HalfMaxBBoxHeight;
    int lastScaledCol = lastCenterJ + HalfMaxBBoxWidth;

    firstRow = __max(0, cvRound(ConvertToScale0(firstScaledRow, lastScale)));
    firstCol = __max(0, cvRound(ConvertToScale0(firstScaledCol, lastScale)));
    lastRow = __min(height, cvRound(ConvertToScale0(lastScaledRow, lastScale)));
    lastCol = __min(width, cvRound(ConvertToScale0(lastScaledCol, lastScale)));
 }

  int prevScaledI = -1;
  int prevScaledJ = -1;
  int FirstRow;
  int LastRow;
  int FirstCol;
  int LastCol;

 for ( int i = firstRow; i < lastRow; i++ )
 {
  if ( i%10 == 0 )
  {
    if (init)
      std::cout << " initializing accumulator at scale " << iScale << " for row " << i << std::endl;
    else
      std::cout << " updating accumulator at scale " << iScale << " for row " << i << std::endl;
  }

  for ( int j = firstCol; j < lastCol; j++ )
  {
   float patchGivenCost = getPatchGivenCost(0, i, j);
   float patchDeltaCost = getPatchDeltaCost(0, i, j);
   float patchBgCost = CV_IMAGE_ELEM(BgCostMap, float, i, j);

   if ( patchDeltaCost < 0  // vote has changed
    || init ) // init
   {
    int scaledI = cvRound(ConvertFromScale0(i, iScale));
    int scaledJ = cvRound(ConvertFromScale0(j, iScale));

    if (patchBgCost - m_fBgBias < -safeLog(1.0-uForest.getBgThresh()))
    {
      float bg_prob;

      if (scaledI != prevScaledI || scaledJ != prevScaledJ)
      {
        uForest.fillConditionalsSingleScale(scaledJ, scaledI, iScale, bg_prob, 
          FirstRow, LastRow, FirstCol, LastCol); // calculate probabilities
        uForest.blurConditionalsSingleScale(scaledJ, scaledI, iScale, m_blur_radius,
          FirstRow, LastRow, FirstCol, LastCol);

        prevScaledI = scaledI;
        prevScaledJ = scaledJ;
      }

      IplImage *probMap = uForest.m_vecResizedProbImages[iScale]; // probability map at iScale

      updateCostsOfOnePatch(probMap, scaledI, scaledJ, 
        patchGivenCost, patchDeltaCost, 
        patchBgCost, accumMap, init, iScale,
        FirstRow, LastRow, FirstCol, LastCol);
    }
   } 
  }
 }
}

// =======================================================================================================
bool CGreedyDetection::updateDetections()
{
 float minDeltaEnergy = 1e6;
 int bestI = -1;
 int bestJ = -1;
 int bestScale = -1;
 float peakHeight;

 for (int iScale = 0; iScale < m_iNumberOfScales; iScale++)
 {  
  float curWidth = ConvertFromScale0((float)width, iScale);
  float curHeight = ConvertFromScale0((float)height, iScale);

   IplImage *tmpAccum = cvCloneImage(m_vecAccumulators[iScale]); 
  for( int i = 0; i < tmpAccum->height; i++ )
  {
   for( int j = 0; j < tmpAccum->width; j++ )
   {
    float curAccValue = CV_IMAGE_ELEM( tmpAccum, float, i, j );
    float curDeltaEnergy = curAccValue + m_fPenalty*((float)(width*height))/(curWidth*curHeight);
    
    if (curDeltaEnergy <= minDeltaEnergy)
    {
     bestI = i;
     bestJ = j;
     bestScale = iScale;
     minDeltaEnergy = curDeltaEnergy;
     peakHeight = -curAccValue;
    }
   }
  }
   cvReleaseImage(&tmpAccum);
 }

 // SUMMATION OF LOGS
  if (minDeltaEnergy < 0)
 {
  m_vecDetections.push_back(cvPoint(bestJ, bestI));
  m_vecDetScales.push_back(bestScale);
  m_vecHeights.push_back(peakHeight);

  return true;
 }


 return false;
}


// =======================================================================================================
void CGreedyDetection::updateGivenCosts()
{
 cvSetZero(DeltaCosts);

 int forestPatchSize = uForest.getPatchSize();

 int lastCenterI = m_vecDetections[m_vecDetections.size()-1].y;
 int lastCenterJ = m_vecDetections[m_vecDetections.size()-1].x;
 int lastScale = m_vecDetScales[m_vecDetections.size()-1];

 int firstScaledRow = lastCenterI - HalfMaxBBoxHeight;
 int firstScaledCol = lastCenterJ - HalfMaxBBoxWidth;
 int lastScaledRow = lastCenterI + HalfMaxBBoxHeight;
 int lastScaledCol = lastCenterJ + HalfMaxBBoxWidth;

 int firstRow = __max(0, cvRound(ConvertToScale0(firstScaledRow, lastScale)));
 int firstCol = __max(0, cvRound(ConvertToScale0(firstScaledCol, lastScale)));
 int lastRow = __min(height-1, cvRound(ConvertToScale0(lastScaledRow, lastScale)));
 int lastCol = __min(width-1, cvRound(ConvertToScale0(lastScaledCol, lastScale)));


 
  int prevScaledRow = -1;
  int prevScaledCol = -1;

 for(int row = firstRow; row < lastRow; row++ )
 {
  if (row%10 == 0 )
  {
   std::cout << "updating costs of patches. row " << row << std::endl;
  }

  for(int col = firstCol; col < lastCol; col++ )
  {   
   float bg_prob_of_scale;

   int scaledRow = ConvertFromScale0(row, lastScale);
   int scaledCol = ConvertFromScale0(col, lastScale);

   float patchBgCost = CV_IMAGE_ELEM(BgCostMap, float, row, col);

   if (patchBgCost - m_fBgBias < -safeLog(1.0-uForest.getBgThresh()))
   {
    if (scaledRow != prevScaledRow || scaledCol != prevScaledCol)
    {
      int FirstRow;
      int LastRow;
      int FirstCol;
      int LastCol;

      uForest.fillConditionalsSingleScale( scaledCol, scaledRow, lastScale, bg_prob_of_scale, 
        FirstRow, LastRow, FirstCol, LastCol); // calculate probabilities
      uForest.blurConditionalsSingleScale( scaledCol, scaledRow, lastScale, m_blur_radius, 
        FirstRow, LastRow, FirstCol, LastCol);

      prevScaledRow = scaledRow;
      prevScaledCol = scaledCol;
    }

    IplImage *probMap = uForest.m_vecResizedProbImages[lastScale];

    float patchProbForLastCenter = CV_IMAGE_ELEM(probMap, float, lastCenterI, lastCenterJ);

    if (patchProbForLastCenter > m_fProbVoteThreshold)
    {
     float BgCostOfPatch = CV_IMAGE_ELEM( BgCostMap, float, row, col );
     float patchVoteForLastCenter = safeLog (patchProbForLastCenter);
     float patchCostForLastCenter = (-patchVoteForLastCenter) - BgCostOfPatch;

     float curCost = CV_IMAGE_ELEM( CurrentGivenCosts, float, row, col );

     if ( patchCostForLastCenter < curCost )
     // patch has changed its correspondence to object
     {  
      CV_IMAGE_ELEM( DeltaCosts, float, row, col ) = patchCostForLastCenter - curCost; 
      CV_IMAGE_ELEM( CurrentGivenCosts, float, row, col ) = patchCostForLastCenter;
     }
     
    } 
   }
  }
 }

 fillResizedCosts();
}


// =======================================================================================================
void CGreedyDetection::drawDetections(const char *in_sPath)
{

 IplImage *tmpImage = cvCloneImage(im.cvImage);

 for (unsigned int iDetection = 0; iDetection < m_vecDetections.size(); iDetection++)
 {
  int curDetI = m_vecDetections[iDetection].y;
  int curDetJ = m_vecDetections[iDetection].x;
  int curDetScale = m_vecDetScales[iDetection];


  float curBBoxWidth = BBoxWidth;
  float curBBoxHeight = BBoxHeight;

  CvPoint pt1, pt2;
  pt1.x = cvRound(__max(0, ConvertToScale0(curDetJ - curBBoxWidth/2.0f, curDetScale)) );
  pt1.y = cvRound(__max(0, ConvertToScale0(curDetI - curBBoxHeight/2.0f, curDetScale)) );
  pt2.x = cvRound(__min(width-1, ConvertToScale0(curDetJ + curBBoxWidth/2.0f, curDetScale)) );
  pt2.y = cvRound(__min(height-1, ConvertToScale0(curDetI + curBBoxHeight/2.0f, curDetScale)) );
  cvRectangle( tmpImage, pt1, pt2, CV_RGB(255,255,255), 3, 8 );
 }

 cvSaveImage(in_sPath, tmpImage);
 cvReleaseImage(&tmpImage);
}


// =======================================================================================================
void CGreedyDetection::printDetections(const char *in_sPath)
{
 std::ofstream of(in_sPath);

 for (unsigned int iDetection = 0; iDetection < m_vecDetections.size(); iDetection++)
 {
  int curDetI = m_vecDetections[iDetection].y;
  int curDetJ = m_vecDetections[iDetection].x;
  int curDetScale = m_vecDetScales[iDetection];
  float curHeight = m_vecHeights[iDetection];

  of << curDetJ << '\t' << curDetI << '\t' << curDetScale << '\t' << curHeight << std::endl;
 }
 of.close();
}

// =======================================================================================================
void CGreedyDetection::Detect(const char *input_image_path, 
               float in_fKoef,
               float in_fBgBias, float in_fHypPenalty,
               const char *output_image_path,
               int in_iMaxObjectsCount)
{ 
 m_fPenalty = in_fHypPenalty;
 m_fBgBias = in_fBgBias;

 m_vecDetections.clear();
 m_vecDetScales.clear();
 m_vecHeights.clear();

 printf("Loading image ...\n");
 im.LoadMultiImageHog32(input_image_path, in_fKoef);  
 printf("Done...\n");

 width = im.width;
 height = im.height;

 uForest.setTestImage(&im);

 initAccumulator();
 initGivenCosts();

 char tmpName[256];

 int step = 0; 
 while (step < in_iMaxObjectsCount) 
 {
  if (step == 0)
  {
     updateAccumulator(true); 
   }
   else
   {
     updateAccumulator(false);
  }
  if (! updateDetections() )
  {
    break;
  }
  updateGivenCosts();
  step ++;
 }

 drawDetections(output_image_path);
}


// =======================================================================================================
void CGreedyDetection::initAccumulator()
{
 for (unsigned int i = 0; i < m_vecAccumulators.size(); i ++)
 {
  cvReleaseImage(&m_vecAccumulators[i]);
 }
 m_vecAccumulators.resize(m_iNumberOfScales);

 for (unsigned int iScale = 0; iScale < m_vecAccumulators.size(); iScale ++)
 {
  int curWidth = cvRound(ConvertFromScale0((float)width, iScale));
  int curHeight = cvRound(ConvertFromScale0((float)height, iScale));

  m_vecAccumulators[iScale] = cvCreateImage(cvSize(curWidth, curHeight), IPL_DEPTH_32F, 1);
  
  cvSetZero(m_vecAccumulators[iScale]);
 }

}


// =======================================================================================================
void CGreedyDetection::fillResizedCosts()
{
 for (unsigned int i = 0; i < m_vecResizedGivenCosts.size(); i ++)
 {
  cvReleaseImage(&m_vecResizedGivenCosts[i]);
 }
 m_vecResizedGivenCosts.resize(m_iNumberOfScales);

 for (unsigned int i = 0; i < m_vecResizedDeltaCosts.size(); i ++)
 {
  cvReleaseImage(&m_vecResizedDeltaCosts[i]);
 }
 m_vecResizedDeltaCosts.resize(m_iNumberOfScales);

 for (unsigned int iScale = 0; iScale < m_iNumberOfScales; iScale ++)
 {
  int curWidth = cvRound(ConvertFromScale0((float)width, iScale));
  int curHeight = cvRound(ConvertFromScale0((float)height, iScale));

  m_vecResizedGivenCosts[iScale] = cvCreateImage(cvSize(curWidth, curHeight), IPL_DEPTH_32F, 1);
  m_vecResizedDeltaCosts[iScale] = cvCreateImage(cvSize(curWidth, curHeight), IPL_DEPTH_32F, 1);

  cvResize(CurrentGivenCosts, m_vecResizedGivenCosts[iScale], CV_INTER_CUBIC);
  cvResize(DeltaCosts, m_vecResizedDeltaCosts[iScale], CV_INTER_CUBIC);
 }
}



// =======================================================================================================
void CGreedyDetection::initGivenCosts()
{
 cvReleaseImage(&CurrentGivenCosts);
 CurrentGivenCosts = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);

 cvReleaseImage(&BgCostMap);
 BgCostMap = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);


 //DEBUG 
 cvSet(CurrentGivenCosts, cvScalar(0));

 for ( int i = 0; i < height; i++ )
 {
  if ( i%10 == 0 )
  {
   std::cout << " initializing given costs " << i << std::endl;
  }

  for ( int j = 0; j < width; j++ )
  {    
    std::vector <float> bg_prob;
    uForest.fillBgProbsMultiScale(j, i, bg_prob); // calculated probabilities

    float min_bg_prob = 1e6;
    for (unsigned int iScale = 0; iScale < bg_prob.size(); iScale++)
    {
     if (bg_prob[iScale] < min_bg_prob)
     {
      min_bg_prob = bg_prob[iScale];
     }
    }
    CV_IMAGE_ELEM(BgCostMap, float, i, j) = - safeLog(min_bg_prob) + m_fBgBias;
  }
 }

 cvReleaseImage(&DeltaCosts);
 DeltaCosts = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
 cvSet(DeltaCosts, cvScalar(1e6f));

 fillResizedCosts();
}


// =======================================================================================================
void CGreedyDetection::saveAccumulator(std::string in_sPath)
{
 for (unsigned int iScale = 0; iScale < m_vecAccumulators.size(); iScale ++)
 {
  char tmpName[256];
  sprintf(tmpName, "%s%03d.txt", in_sPath.c_str(), iScale);

  IplImage *tmpAccum = cvCloneImage(m_vecAccumulators[iScale]);

  float curWidth = ConvertFromScale0((float)width, iScale);
  float curHeight = ConvertFromScale0((float)height, iScale);
  
  IplImage *Coeff = cvCreateImage(cvSize(tmpAccum->width, tmpAccum->height), IPL_DEPTH_32F, 1);

   cvSet(Coeff, cvScalar(curWidth*curHeight / (float)(width*height)));

  cvMul(tmpAccum, Coeff, tmpAccum);
  cvReleaseImage(&Coeff);
  
  SaveImageAsText(tmpName, tmpAccum);

  cvReleaseImage(&tmpAccum);  
 }
}