#include "HoughForest.h"
#include "Vote.h"
#include "HoughTree.h"
#include "../ImageWrapper/MultiImageTools.h"

#include <cstddef>

extern float HalfBBoxWidth;
extern float HalfBBoxHeight;
extern float blur_radius;



// =======================================================================================================
CHoughForest::CHoughForest()
{
    m_vecResizedProbImages.clear();
    m_iNumberOfScales = 1.0;
    m_dResizeCoef = 1.0;
}



// =======================================================================================================
void CHoughForest::setScalingParameters(int in_iNumberOfScales, double in_dResizeCoef)
{
    m_iNumberOfScales = in_iNumberOfScales;
    m_dResizeCoef = in_dResizeCoef;
}


// =======================================================================================================
CHoughForest::~CHoughForest()
{
    for (std::size_t i = 0; i < m_vecResizedProbImages.size(); i ++)
    {
        cvReleaseImage(&m_vecResizedProbImages[i]);
    }

    for (std::size_t i = 0; i < m_vecResizedMultiImages.size(); i ++)
    {
        delete m_vecResizedMultiImages[i];
    }

    return;
}


// =======================================================================================================
void CHoughForest::SetNumberOfTrees(std::size_t in_iNumberOfTrees)
{
    nTrees = in_iNumberOfTrees;

    forest.resize(nTrees);
    for (int i = 0; i < nTrees; i ++)
    {
        forest[i] = new CHoughTree;
    }

    for(int i = 0; i < m_vecResizedProbImages.size(); i ++)
    {
        cvReleaseImage(&m_vecResizedProbImages[i]);
    }
}


// =======================================================================================================
int CHoughForest::getPatchSize()
{
    return forest[0]->patchHeight;
}


// =======================================================================================================
void CHoughForest::fillBgProbsMultiScale(unsigned int in_uiPatchCenterX, 
                                         unsigned int in_uiPatchCenterY,
                                         std::vector <float> &out_BgProbsOfScales)
{
    out_BgProbsOfScales.resize(m_iNumberOfScales);
    float term = 1.0;

    float patchSize = getPatchSize();

    for (int idxScale = 0; idxScale < m_iNumberOfScales; idxScale ++)
    {
        int curWidth = m_vecResizedProbImages[idxScale]->width;
        int curHeight = m_vecResizedProbImages[idxScale]->height;

        int scaledPatchLeftX =  term * in_uiPatchCenterX - patchSize/2;
        int scaledPatchTopY =  term * in_uiPatchCenterY - patchSize/2;

        if ( scaledPatchLeftX < 0 || scaledPatchLeftX > curWidth - patchSize-1 ||
                scaledPatchTopY < 0 || scaledPatchTopY > curHeight - patchSize-1 )
        {
            out_BgProbsOfScales[idxScale] = 1.0;
            continue;
        }

        //const float width = m_vecResizedProbImages[0]->width;
        //const float height = m_vecResizedProbImages[0]->height;

        HoughSample hs;
        hs.left = std::min<float>(std::max(0, scaledPatchLeftX), curWidth - patchSize-1);
        hs.top = std::min<float>(std::max(0, scaledPatchTopY), curHeight - patchSize-1);
        hs.im = m_vecResizedMultiImages[idxScale];

        float curBgProb = 0;
        for(std::size_t t = 0; t < nTrees; t+=1)
        {
            CVote *hv = (CVote *) forest[t]->TestSample(&hs);
            int hv_size = hv->offsets.size();

            float leaf_bg_prob = (float)(hv->nSamples - hv_size) / (float)hv->nSamples;

            curBgProb += leaf_bg_prob / (float)nTrees;
        }

        out_BgProbsOfScales[idxScale] = curBgProb;

        term *= m_dResizeCoef;

    }

    return;
} // end of CHoughForest::fillBgProbsMultiScale


// =======================================================================================================
void CHoughForest::blurConditionalsSingleScale(unsigned int in_uiScaledPatchCenterX, 
                                               unsigned int in_uiScaledPatchCenterY,
                                               int in_iScale, int in_iBlurRadius,
                                               int in_iFirstRow, int in_iLastRow,
                                               int in_iFirstCol, int in_iLastCol)
{
    float patchSize = getPatchSize();

    int curWidth = m_vecResizedProbImages[in_iScale]->width;
    int curHeight = m_vecResizedProbImages[in_iScale]->height;

    float scaleCoef = (float)curWidth / (float)m_vecResizedProbImages[0]->width;

    int scaledPatchLeftX =  in_uiScaledPatchCenterX - patchSize/2.0;
    int scaledPatchTopY =  in_uiScaledPatchCenterY - patchSize/2.0;

    if ( in_uiScaledPatchCenterX < patchSize/(2.0*scaleCoef) ||
            in_uiScaledPatchCenterX > curWidth - patchSize/(2.0*scaleCoef) ||
            in_uiScaledPatchCenterY < patchSize/(2.0*scaleCoef) ||
            in_uiScaledPatchCenterY > curHeight - patchSize/(2.0*scaleCoef) )
    {
        return;
    }

    int firstScaledRow = std::max(0, in_iFirstRow - in_iBlurRadius);
    int firstScaledCol = std::max(0, in_iFirstCol - in_iBlurRadius);
    int lastScaledRow = std::min(curHeight-1, in_iLastRow + in_iBlurRadius);
    int lastScaledCol = std::min(curWidth-1, in_iLastCol + in_iBlurRadius);

    cvSetImageROI(m_vecResizedProbImages[in_iScale], cvRect(firstScaledCol, firstScaledRow, lastScaledCol-firstScaledCol+1, lastScaledRow-firstScaledRow+1));
    cvSmooth( m_vecResizedProbImages[in_iScale], m_vecResizedProbImages[in_iScale], CV_BLUR, 2*in_iBlurRadius+1, 2*in_iBlurRadius+1);
    cvResetImageROI(m_vecResizedProbImages[in_iScale]);

}


// =======================================================================================================
void CHoughForest::fillConditionalsSingleScale(unsigned int in_uiScaledPatchCenterX,
                                               unsigned int in_uiScaledPatchCenterY,
                                               int in_iScale, float &out_fBgProb,
                                               int &out_iFirstRow, int &out_iLastRow,
                                               int &out_iFirstCol, int &out_iLastCol)
{  
    float patchSize = getPatchSize();

    int curWidth = m_vecResizedProbImages[in_iScale]->width;
    int curHeight = m_vecResizedProbImages[in_iScale]->height;

    float scaleCoef = (float)curWidth / (float)m_vecResizedProbImages[0]->width;

    int firstScaledRow = std::max(0.0f, in_uiScaledPatchCenterY - HalfBBoxHeight);
    int firstScaledCol = std::max(0.0f, in_uiScaledPatchCenterX - HalfBBoxWidth);
    int lastScaledRow = std::min<float>(curHeight-1, in_uiScaledPatchCenterY + HalfBBoxHeight);
    int lastScaledCol = std::min<float>(curWidth-1, in_uiScaledPatchCenterX + HalfBBoxWidth);


    int scaledPatchLeftX =  in_uiScaledPatchCenterX - patchSize/2.0;
    int scaledPatchTopY =  in_uiScaledPatchCenterY - patchSize/2.0;

    if ( in_uiScaledPatchCenterX < patchSize/(2.0*scaleCoef) ||
            in_uiScaledPatchCenterX > curWidth - patchSize/(2.0*scaleCoef) ||
            in_uiScaledPatchCenterY < patchSize/(2.0*scaleCoef) ||
            in_uiScaledPatchCenterY > curHeight - patchSize/(2.0*scaleCoef) )
    {
        out_iFirstCol = in_uiScaledPatchCenterX;
        out_iLastCol = in_uiScaledPatchCenterX;
        out_iFirstRow = in_uiScaledPatchCenterY;
        out_iLastRow = in_uiScaledPatchCenterY;

        out_fBgProb = 1.0;

        return;
    }

    cvSetImageROI(m_vecResizedProbImages[in_iScale],
                  cvRect(firstScaledCol, firstScaledRow,
                         lastScaledCol-firstScaledCol+1, lastScaledRow-firstScaledRow+1));
    cvSetZero(m_vecResizedProbImages[in_iScale]);
    cvResetImageROI(m_vecResizedProbImages[in_iScale]);

    HoughSample hs;
    hs.left = std::min<float>(std::max(0, scaledPatchLeftX), curWidth - patchSize-1);
    hs.top = std::min<float>(std::max(0, scaledPatchTopY), curHeight - patchSize-1);
    hs.im = m_vecResizedMultiImages[in_iScale];

    float width = m_vecResizedProbImages[0]->width;
    float height = m_vecResizedProbImages[0]->height;

    float CountKoeff = 1.0;
    float curBgProb = 0;
    
    out_iFirstCol = width;
    out_iLastCol = -1;
    out_iFirstRow = height;
    out_iLastRow = -1;

    for(unsigned int t = 0; t < nTrees; t++)
    {
        CVote *hv = (CVote *) forest[t]->TestSample(&hs);
        int hv_size = hv->offsets.size();

        float treeOutput = CountKoeff / (float)(hv->nSamples) / (float)nTrees;

        float leaf_bg_prob = (float)(hv->nSamples - hv_size) / (float)hv->nSamples;

        curBgProb += leaf_bg_prob / (float)nTrees;

        if (leaf_bg_prob < m_fBgThresh)
        {
            std::vector <CvPoint2D64f> :: iterator it = hv->offsets.begin();
            for(unsigned int j = 0; j < hv_size; j++, it++)
            {

                int instanceY = hs.top + it->y;
                int instanceX = hs.left + it->x;

                if ( instanceX >= 0 && instanceY >= 0 &&
                        instanceX < curWidth && instanceY < curHeight )
                {
                    CV_IMAGE_ELEM(m_vecResizedProbImages[in_iScale], float, instanceY, instanceX) += treeOutput;


                    out_iFirstCol = std::min(out_iFirstCol, instanceX);
                    out_iLastCol = std::max(out_iLastCol, instanceX);
                    out_iFirstRow = std::min(out_iFirstRow, instanceY);
                    out_iLastRow = std::max(out_iLastRow, instanceY);
                }
            }
        }
    }

    out_fBgProb = curBgProb;

    if (out_iFirstCol > out_iLastCol || out_iFirstRow > out_iLastRow)
    {
        out_iFirstCol = in_uiScaledPatchCenterX;
        out_iLastCol = in_uiScaledPatchCenterX;
        out_iFirstRow = in_uiScaledPatchCenterY;
        out_iLastRow = in_uiScaledPatchCenterY;
    }

    out_iFirstCol = std::max(firstScaledCol, out_iFirstCol);
    out_iLastCol = std::min(lastScaledCol, out_iLastCol);
    out_iFirstRow = std::max(firstScaledRow, out_iFirstRow);
    out_iLastRow = std::min(lastScaledRow, out_iLastRow);
}



// =======================================================================================================
void CHoughForest::setTestImage(MultiImage *im)
{
    m_testImage = im;
    m_iTestImageWidth = im->width;
    m_iTestImageHeight = im->height;

    for(int i = 0; i < m_vecResizedProbImages.size(); i ++)
    {
        delete m_vecResizedMultiImages[i];
        cvReleaseImage(&m_vecResizedProbImages[i]);
    }

    m_vecResizedProbImages.resize(m_iNumberOfScales);
    m_vecResizedMultiImages.resize(m_iNumberOfScales);

    double term = 1.0;
    for(int i = 0; i < m_iNumberOfScales; i ++)
    {
        MultiImage* im_resized = new MultiImage;
        GetResizedMultiImage(*m_testImage, term*m_iTestImageWidth, term*m_iTestImageHeight, *im_resized);
        m_vecResizedMultiImages[i] = im_resized;
        m_vecResizedProbImages[i] = cvCreateImage(cvSize(term*m_iTestImageWidth, term*m_iTestImageHeight), 32, 1);
        cvSetZero(m_vecResizedProbImages[i]);
        term *= m_dResizeCoef;
    }
}



// =======================================================================================================
void CHoughForest::setBgThresh(float in_fBgThresh)
{
    m_fBgThresh = in_fBgThresh;
}


// =======================================================================================================
float CHoughForest::getBgThresh()
{
    return m_fBgThresh;
}
