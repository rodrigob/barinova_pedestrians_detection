#ifdef DWIN
#include "stdafx.h"
#endif

#include <vector>
#include <iostream>
#include "HoG.h"

using namespace std;

HoG::HoG() {
  bins = 9;
  binsize = (3.14159265f*80.0f)/float(bins);
  cell_w = 3;
  block_w = 6;
  Gauss1 = cvCreateMat( cell_w, cell_w, CV_32FC1 );
  Gauss2 = cvCreateMat( cell_w, cell_w, CV_32FC1 );
  Gauss3 = cvCreateMat( cell_w, cell_w, CV_32FC1 );
  Gauss4 = cvCreateMat( cell_w, cell_w, CV_32FC1 );
  double a = -(block_w-1)/2.0;
  double sigma2 = 2*(0.5*block_w)*(0.5*block_w);
  double count = 0;
  for(int x = 0; x<cell_w; ++x) {
    for(int y = 0; y<cell_w; ++y) {
      double tmp = exp(-( (a+x)*(a+x)+(a+y)*(a+y) )/sigma2);
      count += tmp;
      cvSet2D( Gauss1, x, y, cvScalar(tmp) );
    }
  }
  count *=4;
  cvConvertScale( Gauss1, Gauss1, 1.0/count);

  cvFlip( Gauss1, Gauss2, 1);
  cvFlip( Gauss1, Gauss3, 0);
  cvFlip( Gauss1, Gauss4, -1);

  g_w = 5;
  Gauss = cvCreateMat( g_w, g_w, CV_32FC1 );
  a = -(g_w-1)/2.0;
  sigma2 = 2*(0.5*g_w)*(0.5*g_w);
  count = 0;
  for(int x = 0; x<g_w; ++x) {
    for(int y = 0; y<g_w; ++y) {
      double tmp = exp(-( (a+x)*(a+x)+(a+y)*(a+y) )/sigma2);
      count += tmp;
      cvSet2D( Gauss, x, y, cvScalar(tmp) );
    }
  }
  cvConvertScale( Gauss, Gauss, 1.0/count);

  ptGauss = new float[g_w*g_w];
  int i = 0;
  for(int y = 0; y<g_w; ++y) 
    for(int x = 0; x<g_w; ++x)
      ptGauss[i++] = (float)cvmGet( Gauss, x, y );


  //for(int x = 0; x<g_w; ++x) {
  //  for(int y = 0; y<g_w; ++y) {
  //    cout << cvmGet( Gauss, x, y ) << " ";
  //  }
  //  cout << endl;
  //}

  //cvReleaseMat(&Gauss);cvReleaseMat(&Gauss1);cvReleaseMat(&Gauss2);cvReleaseMat(&Gauss3);cvReleaseMat(&Gauss4);
}

void HoG::extractHoG(IplImage *Iorient, IplImage *Imagn, IplImage *out1, IplImage *out2, IplImage *out3, IplImage *out4) {
  vector<double> desc(4*bins);
  cvSetZero( out1 );
  cvSetZero( out2 );
  cvSetZero( out3 );
  cvSetZero( out4 );
  CvMat BLorient; //= cvCreateMat( block_w, block_w, CV_8UC1 );
  CvMat BLmagn; //= cvCreateMat( block_w, block_w, CV_8UC1 );
  CvMat BLout; //= cvCreateMat( cell_w, cell_w, CV_8UC1 );
  CvRect rect;
  rect.height = block_w;
  rect.width = block_w;
  CvRect outr;
  outr.height = cell_w;
  outr.width = cell_w;

  for(int x=0;x<Iorient->width-block_w;x+=cell_w) {
    for(int y=0;y<Iorient->height-block_w;y+=cell_w) {
      rect.x = x;
      rect.y = y;
      cvGetSubRect( Iorient, &BLorient, rect );
      cvGetSubRect( Imagn, &BLmagn, rect );

      calcHoGDesc( &BLorient, &BLmagn, desc );

      outr.x = x;
      outr.y = y;
      cvGetSubRect( out1, &BLout, outr );
      int dind = 0;
      for(int x2=0;x2<cell_w; ++x2) {
        for(int y2=0;y2<cell_w; ++y2) {
          uchar* pt = (uchar*)cvPtr2D( &BLout, x2, y2);
          *pt = (uchar)desc[dind++];
        }
      }

      outr.x = x+cell_w;
      outr.y = y;
      cvGetSubRect( out2, &BLout, outr );
      for(int x2=0;x2<cell_w; ++x2) {
        for(int y2=0;y2<cell_w; ++y2) {
          uchar* pt = (uchar*)cvPtr2D( &BLout, x2, y2);
          *pt = (uchar)desc[dind++];
        }
      }

      outr.x = x;
      outr.y = y+cell_w;
      cvGetSubRect( out3, &BLout, outr );
      for(int x2=0;x2<cell_w; ++x2) {
        for(int y2=0;y2<cell_w; ++y2) {
          uchar* pt = (uchar*)cvPtr2D( &BLout, x2, y2);
          *pt = (uchar)desc[dind++];
        }
      }

      outr.x = x+cell_w;
      outr.y = y+cell_w;
      cvGetSubRect( out4, &BLout, outr );
      for(int x2=0;x2<cell_w; ++x2) {
        for(int y2=0;y2<cell_w; ++y2) {
          uchar* pt = (uchar*)cvPtr2D( &BLout, x2, y2);
          *pt = (uchar)desc[dind++];
        }
      }
      
    }
  }
  //cvReleaseMat(&BLorient);
  //cvReleaseMat(&BLmagn);
  //cvReleaseMat(&BLout);
}

void HoG::extractWeightedOrient(IplImage *Iorient, IplImage *Imagn, IplImage* out) {
  cvSetZero( out );
  CvMat BLorient; 
  CvMat BLmagn; 
  CvRect rect;
  rect.height = g_w;
  rect.width = g_w;
  int off_w = int(g_w/2.0); 

  for(int x=0;x<Iorient->width-g_w;x++) {
    for(int y=0;y<Iorient->height-g_w;y++) {
      rect.x = x;
      rect.y = y;
      cvGetSubRect( Iorient, &BLorient, rect );
      cvGetSubRect( Imagn, &BLmagn, rect );

      double tmp = (uchar)calcWeightedOrient(&BLorient, &BLmagn);
      *(uchar*)cvPtr2D( out, y+off_w, x+off_w) = (uchar)tmp;      
    }
  }

}

#if 0
void HoG::extractOBin(IplImage *Iorient, IplImage *Imagn, std::vector<IplImage*>& out, int off) {
  vector<double> desc(bins);
  for(unsigned int k=off; k<desc.size()+off; ++k)
    cvSetZero( out[k] );
  CvMat BLorient; 
  CvMat BLmagn; 
  CvRect rect;
  rect.height = g_w;
  rect.width = g_w;
  int off_w = int(g_w/2.0); 

  for(int x=0;x<Iorient->width-g_w;x++) {
    for(int y=0;y<Iorient->height-g_w;y++) {
      rect.x = x;
      rect.y = y;
      cvGetSubRect( Iorient, &BLorient, rect );
      cvGetSubRect( Imagn, &BLmagn, rect );

      calcHoGBin( &BLorient, &BLmagn, desc );

      int dy = y+off_w; 
      int dx = x+off_w;
      for(unsigned int k=off, l=0; l<desc.size(); ++k, ++l)
        *(uchar*)cvPtr2D( out[k], dy, dx) = (uchar)desc[l];      
    }
  }
}
#else

void HoG::extractOBin(IplImage *Iorient, IplImage *Imagn, std::vector<IplImage*>& out, int off) {
  double* desc = new double[bins];

  // reset output image (border=0) and get pointers
  uchar** ptOut     = new uchar*[bins];
  uchar** ptOut_row = new uchar*[bins];
  for(int k=off; k<bins+off; ++k) {
    cvSetZero( out[k] );
    cvGetRawData( out[k], (uchar**)&(ptOut[k-off]));
  }

  // get pointers to orientation, magnitude
  int step;
  uchar* ptOrient;
  uchar* ptOrient_row;
  cvGetRawData( Iorient, (uchar**)&(ptOrient), &step);
  step /= sizeof(ptOrient[0]);

  uchar* ptMagn;
  uchar* ptMagn_row;
  cvGetRawData( Imagn, (uchar**)&(ptMagn));

  int off_w = int(g_w/2.0); 
  for(int l=0; l<bins; ++l)
    ptOut[l] += off_w*step;

  for(int y=0;y<Iorient->height-g_w; y++, ptMagn+=step, ptOrient+=step) {

    // Get row pointers
    ptOrient_row = &ptOrient[0];
    ptMagn_row = &ptMagn[0];
    for(int l=0; l<bins; ++l)
      ptOut_row[l] = &ptOut[l][0]+off_w;

    for(int x=0; x<Iorient->width-g_w; ++x, ++ptOrient_row, ++ptMagn_row) {
    
      calcHoGBin( ptOrient_row, ptMagn_row, step, desc );

      //int dy = y+off_w; 
      //int dx = x+off_w;
      for(int l=0; l<bins; ++l) {
        *ptOut_row[l] = (uchar)desc[l];
        ++ptOut_row[l];
      }
    }

    // update pointer
    for(int l=0; l<bins; ++l)
      ptOut[l] += step;
  }

  delete[] desc;
  delete[] ptOut;
  delete[] ptOut_row;
}

#endif

void HoG::extractHoG(IplImage *Iorient, IplImage *Imagn, vector<IplImage*>& out) {
  vector<double> desc(4*bins);
  for(unsigned int k=0; k<desc.size(); ++k)
    cvSetZero( out[k] );

  CvMat BLorient; //= cvCreateMat( block_w, block_w, CV_8UC1 );
  CvMat BLmagn; //= cvCreateMat( block_w, block_w, CV_8UC1 );
  CvMat BLout; //= cvCreateMat( cell_w, cell_w, CV_8UC1 );
  CvRect rect;
  rect.height = block_w;
  rect.width = block_w;
  CvRect outr;
  outr.height = cell_w;
  outr.width = cell_w;

  for(int x=0;x<Iorient->width-block_w;x+=cell_w) {
    for(int y=0;y<Iorient->height-block_w;y+=cell_w) {
      rect.x = x;
      rect.y = y;
      cvGetSubRect( Iorient, &BLorient, rect );
      cvGetSubRect( Imagn, &BLmagn, rect );

      calcHoGDesc( &BLorient, &BLmagn, desc );

      outr.x = x;
      outr.y = y;
      for(int k=0; k<bins; ++k) {
        cvGetSubRect( out[k], &BLout, outr );
        cvSet( &BLout, cvScalar(desc[k]));
      }

      outr.x = x+cell_w;
      outr.y = y;
      for(int k=bins; k<2*bins; ++k) {
        cvGetSubRect( out[k], &BLout, outr );
        cvSet( &BLout, cvScalar(desc[k]));
      }

      outr.x = x;
      outr.y = y+cell_w;
      for(int k=2*bins; k<3*bins; ++k) {
        cvGetSubRect( out[k], &BLout, outr );
        cvSet( &BLout, cvScalar(desc[k]));
      }

      outr.x = x+cell_w;
      outr.y = y+cell_w;
      for(int k=3*bins; k<4*bins; ++k) {
        cvGetSubRect( out[k], &BLout, outr );
        cvSet( &BLout, cvScalar(desc[k]));
      }
      
    }
  }
  //cvReleaseMat(&BLorient);
  //cvReleaseMat(&BLmagn);
  //cvReleaseMat(&BLout);
}

void HoG::calcHoGDesc(CvMat* BLorient, CvMat* BLmagn, vector<double>& desc) {
  CvRect rcell;
  rcell.height = cell_w; rcell.width = cell_w;
  CvMat cellO; //= cvCreateMat( cell_w, cell_w, CV_8UC1 );
  CvMat cellM; //= cvCreateMat( cell_w, cell_w, CV_8UC1 );
  for(unsigned int i=0; i<desc.size();i++)
    desc[i]=0;

  rcell.x = 0;
  rcell.y = 0;
  cvGetSubRect( BLorient, &cellO, rcell );
  cvGetSubRect( BLmagn, &cellM, rcell );

  for(int x2=0;x2<cell_w; ++x2) {
    for(int y2=0;y2<cell_w; ++y2) {
      //int bin = int( (float)*(uchar*)cvPtr2D( &cellO, x2, y2)/binsize );
      //desc[bin] += (float)*(uchar*)cvPtr2D( &cellM, x2, y2) * *(float*)cvPtr2D( Gauss1, x2, y2);
      binning((float)*(uchar*)cvPtr2D( &cellO, x2, y2)/binsize, (float)*(uchar*)cvPtr2D( &cellM, x2, y2) * *(float*)cvPtr2D( Gauss1, x2, y2), desc, 0, bins);
    }
  }

  rcell.x = cell_w;
  rcell.y = 0;
  cvGetSubRect( BLorient, &cellO, rcell );
  cvGetSubRect( BLmagn, &cellM, rcell );
  int i=bins;
  for(int x2=0;x2<cell_w; ++x2) {
    for(int y2=0;y2<cell_w; ++y2) {
      //int bin = int( (float)*(uchar*)cvPtr2D( &cellO, x2, y2)/binsize );
      //desc[bin+i] += (float)*(uchar*)cvPtr2D( &cellM, x2, y2) * *(float*)cvPtr2D( Gauss2, x2, y2);
      binning((float)*(uchar*)cvPtr2D( &cellO, x2, y2)/binsize, (float)*(uchar*)cvPtr2D( &cellM, x2, y2) * *(float*)cvPtr2D( Gauss1, x2, y2), desc, i, bins);
    }
  }

  rcell.x = 0;
  rcell.y = cell_w;
  cvGetSubRect( BLorient, &cellO, rcell );
  cvGetSubRect( BLmagn, &cellM, rcell );
  i += bins;
  for(int x2=0;x2<cell_w; ++x2) {
    for(int y2=0;y2<cell_w; ++y2) {
      //int bin = int( (float)*(uchar*)cvPtr2D( &cellO, x2, y2)/binsize );
      //desc[bin+i] += (float)*(uchar*)cvPtr2D( &cellM, x2, y2) * *(float*)cvPtr2D( Gauss3, x2, y2);
      binning((float)*(uchar*)cvPtr2D( &cellO, x2, y2)/binsize, (float)*(uchar*)cvPtr2D( &cellM, x2, y2) * *(float*)cvPtr2D( Gauss1, x2, y2), desc, i, bins);
    }
  }

  rcell.x = cell_w;
  rcell.y = cell_w;
  cvGetSubRect( BLorient, &cellO, rcell );
  cvGetSubRect( BLmagn, &cellM, rcell );
  i += bins;
  for(int x2=0;x2<cell_w; ++x2) {
    for(int y2=0;y2<cell_w; ++y2) {
      //int bin = int( (float)*(uchar*)cvPtr2D( &cellO, x2, y2)/binsize );
      //desc[bin+i] += (float)*(uchar*)cvPtr2D( &cellM, x2, y2) * *(float*)cvPtr2D( Gauss4, x2, y2);
      binning((float)*(uchar*)cvPtr2D( &cellO, x2, y2)/binsize, (float)*(uchar*)cvPtr2D( &cellM, x2, y2) * *(float*)cvPtr2D( Gauss1, x2, y2), desc, i, bins);
    }
  }

  //normalize
  //cout << "Desc ";
  double count = 0.000001;
  for(unsigned int i=0; i<desc.size();i++) {
    //cout << desc[i] << " ";
    count += desc[i]*desc[i];
  }
  //cout << endl << "Norm ";
  
  count = 1.0/sqrt(count)*255;
  
  for(unsigned int i=0; i<desc.size();i++) {
    desc[i]*=count;
    //cout << desc[i] << " ";
  }
  //cout << endl;
  //cvReleaseMat(&cellO);
  //cvReleaseMat(&cellM);
}

void HoG::calcHoGBin(CvMat* BLorient, CvMat* BLmagn, vector<double>& desc) {
  for(unsigned int i=0; i<desc.size();i++)
    desc[i]=0;

  for(int x=0;x<g_w; ++x) {
    for(int y=0;y<g_w; ++y) {
      binning((float)*(uchar*)cvPtr2D( BLorient, x, y)/binsize, (float)*(uchar*)cvPtr2D( BLmagn, x, y) * *(float*)cvPtr2D( Gauss, x, y), desc, 0, bins);
    }
  }

  //for(unsigned int i=0; i<desc.size();i++)
  //  desc[i]*=1.8;
}

double HoG::calcWeightedOrient(CvMat* BLorient, CvMat* BLmagn) {

  double w = 0;
  double w_o = 0;

  for(int x=0;x<g_w; ++x) {
    for(int y=0;y<g_w; ++y) {
      double tmp = (float)*(uchar*)cvPtr2D( BLmagn, x, y) * *(float*)cvPtr2D( Gauss, x, y);
      w   += tmp;
      w_o += (float)*(uchar*)cvPtr2D( BLorient, x, y) * tmp; 
    }
  }


  //normalize
  return w_o/w;
}


