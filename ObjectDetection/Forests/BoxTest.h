#pragma once

#include "stdio.h"

// test in a a node of random forest, comparison of the values of two rectangles at one channel
class BoxTest
{
public:
  int channel;
  int l1, t1, r1, b1, l2, t2, r2, b2;
  double tau;

  bool TestSample(HoughSample *hs) {
    return hs->im->GetBoxSum(l1+hs->left, t1+hs->top, r1+hs->left, b1+hs->top, channel) > 
      hs->im->GetBoxSum(l2+hs->left, t2+hs->top, r2+hs->left, b2+hs->top, channel)+tau;
  }
  void WriteToFile(FILE *out)  {
    fwrite( (void *)&channel, sizeof(int), 1, out);
    fwrite( (void *)&l1, sizeof(int), 1, out);
    fwrite( (void *)&t1, sizeof(int), 1, out);
    fwrite( (void *)&r1, sizeof(int), 1, out);
    fwrite( (void *)&b1, sizeof(int), 1, out);
    fwrite( (void *)&l2, sizeof(int), 1, out);
    fwrite( (void *)&t2, sizeof(int), 1, out);
    fwrite( (void *)&r2, sizeof(int), 1, out);
    fwrite( (void *)&b2, sizeof(int), 1, out);
    fwrite( (void *)&tau, sizeof(double), 1, out);
  }

  void ReadFromFile(FILE *in)  {
    fread( (void *)&channel, sizeof(int), 1, in);
    fread( (void *)&l1, sizeof(int), 1, in);
    fread( (void *)&t1, sizeof(int), 1, in);
    fread( (void *)&r1, sizeof(int), 1, in);
    fread( (void *)&b1, sizeof(int), 1, in);
    fread( (void *)&l2, sizeof(int), 1, in);
    fread( (void *)&t2, sizeof(int), 1, in);
    fread( (void *)&r2, sizeof(int), 1, in);
    fread( (void *)&b2, sizeof(int), 1, in);
    //fread( (void *)&tau, sizeof(double), 1, in);

    int itau;
    fread( (void *)&itau, sizeof(int), 1, in);
    tau = itau;
  }
};