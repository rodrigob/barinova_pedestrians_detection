#pragma once

#include <cxcore.h>

#include <vector>

// calculates HoG features
class HoG {
public:
  HoG();
  ~HoG() {cvReleaseMat(&Gauss);cvReleaseMat(&Gauss1);cvReleaseMat(&Gauss2);cvReleaseMat(&Gauss3);cvReleaseMat(&Gauss4);delete ptGauss;}
  void extractHoG(IplImage *Iorient, IplImage *Imagn, IplImage *out1, IplImage *out2, IplImage *out3, IplImage *out4);
  void extractHoG(IplImage *Iorient, IplImage *Imagn, std::vector<IplImage*>& out);
  void extractWeightedOrient(IplImage *Iorient, IplImage *Imagn, IplImage* out);
  void extractOBin(IplImage *Iorient, IplImage *Imagn, std::vector<IplImage*>& out, int off);
private:
  void calcHoGDesc(CvMat* BLorient, CvMat* BLmagn, std::vector<double>& desc);
  double calcWeightedOrient(CvMat* BLorient, CvMat* BLmagn);
  void calcHoGBin(CvMat* BLorient, CvMat* BLmagn, std::vector<double>& desc);
  void calcHoGBin(uchar* ptOrient, uchar* ptMagn, int step, double* desc);
  void binning(float v, float w, std::vector<double>& desc, int offset, int maxb);
  void binning(float v, float w, double* desc, int maxb);

  int bins;
  float binsize; 
  int cell_w;
  int block_w;
  
  CvMat* Gauss1;
  CvMat* Gauss2;
  CvMat* Gauss3;
  CvMat* Gauss4;

  int g_w;
  CvMat* Gauss;

  // Gauss as vector
  float* ptGauss;
};

inline void HoG::calcHoGBin(uchar* ptOrient, uchar* ptMagn, int step, double* desc) {
  for(int i=0; i<bins;i++)
    desc[i]=0;

  uchar* ptO = &ptOrient[0];
  uchar* ptM = &ptMagn[0];
  int i=0;
  for(int y=0;y<g_w; ++y, ptO+=step, ptM+=step) {
    for(int x=0;x<g_w; ++x, ++i) {
      binning((float)ptO[x]/binsize, (float)ptM[x] * ptGauss[i], desc, bins);
    }
  }
}

inline void HoG::binning(float v, float w, double* desc, int maxb) {
  int bin1 = int(v);
  int bin2;
  float delta = v-bin1-0.5f;
  if(delta<0) {
    bin2 = bin1 < 1 ? maxb-1 : bin1-1; 
    delta = -delta;
  } else
    bin2 = bin1 < maxb-1 ? bin1+1 : 0; 
  desc[bin1] += (1-delta)*w;
  desc[bin2] += delta*w;
}

inline void HoG::binning(float v, float w, std::vector<double>& desc, int offb, int maxb) {
  int bin1 = int(v);
  int bin2;
  float delta = v-bin1-0.5f;
  if(delta<0) {
    bin2 = bin1 < 1 ? maxb-1 : bin1-1; 
    delta = -delta;
  } else
    bin2 = bin1 < maxb-1 ? bin1+1 : 0; 
  desc[bin1+offb] += (1-delta)*w;
  desc[bin2+offb] += delta*w;
}
