#include "MultiImage.h"

// resizes MultiImage
void GetResizedMultiImage(const MultiImage& in_OrigMultiImage, 
              unsigned int in_uiResizedWidth, unsigned int in_uiResizedHeight, MultiImage &out_ResizedMultiImage);

// saves OpenCV IplImage to text format
void SaveImageAsText(const char *in_FileName, IplImage *in_Image);
