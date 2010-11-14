#pragma once
#include "IHoughForest.h"
#include "Vote.h"


class CHoughForest : public IHoughForest <CVote>
{
public:

    CHoughForest();
    virtual ~CHoughForest();

    // sets number of trees in a forest
    void SetNumberOfTrees(std::size_t in_iNumberOfTrees);
    // sets image
    void setTestImage(MultiImage *im);

    // sets threshold on probability of patch to belong to background
    // if probability of background is higher than this threshold, the patch doesn't vote
    void setBgThresh(float in_fBgThresh);
    // get threshold on probability of patch to belong to background
    float getBgThresh();

    // set number of scales for object detection and resize coefficient between subsequent scales
    void setScalingParameters(int in_iNumberOfScales, double in_dResizeCoef);

    // fills conditional probabilities that patch belongs to object centered at image pixels
    // in_iScale - considered scale
    // in_uiScaledPatchCenterX, in_uiScaledPatchCenterY - coordinates of center of a patch in considered scale
    // out_fBgProb - calculated probability of background for the patch
    void fillConditionalsSingleScale(unsigned int in_uiScaledPatchCenterX,
                                     unsigned int in_uiScaledPatchCenterY,
                                     int in_iScale, float &out_fBgProb,
                                     int &out_iFirstRow, int &out_iLastRow,
                                     int &out_iFirstCol, int &out_iLastCol);


    // blurs conditional probability map
    void blurConditionalsSingleScale(unsigned int in_uiScaledPatchCenterX,
                                     unsigned int in_uiScaledPatchCenterY,
                                     int in_iScale, int in_iBlurRadius,
                                     int in_iFirstRow, int in_iLastRow,
                                     int in_iFirstCol, int in_iLastCol);

    // returns probability of background for a patch
    // centered at (in_uiPatchCenterX, in_uiPatchCenterY) for all scales
    void fillBgProbsMultiScale(unsigned int in_uiPatchCenterX,
                               unsigned int in_uiPatchCenterY,
                               std::vector <float> &out_BgProbsOfScales);

    // returns height of the patch
    // patch is supposed to be square here
    int getPatchSize();

    // here conditional probability maps for all scales are stored
    std::vector <IplImage *> m_vecResizedProbImages;

private:

    int m_iNumberOfScales;
    double m_dResizeCoef;

    std::vector <MultiImage*> m_vecResizedMultiImages;

    float m_fBgThresh;

}; // end of class CHoughForest



