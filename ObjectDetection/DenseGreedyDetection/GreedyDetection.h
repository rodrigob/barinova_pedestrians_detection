#pragma once
#include "../ImageWrapper/MultiImageTools.h"
#include "../Forests/HoughForest.h"

#include <cstring>

class CGreedyDetection
{

public:

    // constructor
    CGreedyDetection();

    // destructor
    virtual ~CGreedyDetection();

    // read forest from file, set patch size, blur radius
    void SetForest( const char *in_forest_path,
                   float in_fPatchSize, float in_blur_radius,
                   float in_fBgThresh, float in_fProbVoteThreshold,
                   float in_BBoxWidth, float in_BBoxHeight,
                   float in_HalfMaxBBoxWidth, float in_HalfMaxBBoxHeight,
                   int in_iNumberOfScales, float in_fResizeCoef );

    // read image, detect multiscale objects and save result to image
    void Detect(const char *input_image_path,
                float in_fKoef,
                float in_fBgBias, float in_fHypPenalty,
                const char *output_image_path,
                int in_iMaxObjectsCount);

    // print coordinates of detected objects
    void printDetections(const char *in_sPath);

    // save image with detected objects
    void drawDetections(const char *in_sPath);

protected:

    // updates given costs of patches
    void updateGivenCosts();

    // updates a set of detected objects
    bool updateDetections();

    // updates all accumulators
    void updateAccumulator(bool init);

    // update accumulator at scale iScale
    void updateAccumulatorScale(int iScale, bool init);

    // update cost map by updating costs of one patch
    void updateCostsOfOnePatch(IplImage *in_ProbMap,
                               int in_iPatchI, int in_iPatchJ,
                               float in_fPatchGivenVote,
                               float in_fPatchDeltaVote,
                               float in_fPatchBgCost,
                               IplImage *io_AccumMap,
                               bool init, int iScale,
                               int in_iFirstRow, int in_iLastRow,
                               int in_iFirstCol, int in_iLastCol);

    // get value of cost at position (i, j)
    float getPatchGivenCost(int iScale, int i, int j);

    // get value of delta cost at position (i, j)
    float getPatchDeltaCost(int iScale, int i, int j);

    // init accumulator
    void initAccumulator();

    // init map of given costs
    void initGivenCosts();

    // fill resized costs map
    void fillResizedCosts();

    void saveAccumulator(std::string in_sPath);

    // get row number in scale 0 corresponding to row in_i in scale in_iCurScale
    float ConvertToScale0(float in_I, int in_iCurScale);

    // get row number in in_iCurScale corresponding to row in_i in scale 0
    float ConvertFromScale0(float in_I, int in_iDestScale);

private:

    // image
    MultiImage im;

    // size of object at scale 0
    float BBoxWidth;
    float BBoxHeight;

    // maximum distance from patch to object center along x-axis and y-axis
    float HalfMaxBBoxWidth;
    float HalfMaxBBoxHeight;

    // size of the image
    int width;
    int height;

    // accumulators at different scales
    std::vector <IplImage *> m_vecAccumulators;

    // given costs
    IplImage *CurrentGivenCosts;
    // resized copies of given costs at different scales
    std::vector <IplImage *> m_vecResizedGivenCosts;

    // difference between current costs and previous costs
    IplImage *DeltaCosts;
    // resized copies of delta costs at different scales
    std::vector <IplImage *> m_vecResizedDeltaCosts;

    // costs of background
    IplImage *BgCostMap;

    // parameters of detection:
    // penalty for adding new hypothesis
    float m_fPenalty;
    // additional cost of background
    float m_fBgBias;

    // forest
    CHoughForest uForest;
    // vector of detection positions
    std::vector <CvPoint> m_vecDetections;

    // scale of each detection
    std::vector <int> m_vecDetScales;
    // heights peaks for each detections
    std::vector <float> m_vecHeights;

    // number of scales
    int m_iNumberOfScales;
    // coefficient for resizing images
    float m_fResizeCoef;
    // radius of blur
    int m_blur_radius;
    // patch can vote only for centers which it belongs to with probability higher than threshold
    float m_fProbVoteThreshold;

}; // end of class CGreedyDetection
