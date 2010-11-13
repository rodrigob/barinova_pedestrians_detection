#include "DenseGreedyDetection/GreedyDetection.h"
#include "string.h"
#include "Utils/dirent.h"


// folder with input images
char *host_folder; 

// output folder
char *output_folder;

// width of a bounding box at each scale corresponding to detection
int BBoxWidth; 

// height of a bounding box at each scale corresponding to detection
int BBoxHeight; 

// number of scales for multi-scale detection
int iNumberOfScales;

// resize images before detection by this coefficient
double koef;

//forest name
char* forest_path;

//size of patch in a forest
int PatchSize;

// bias of background cost, parameters of detection algorithm
float bg_bias;

// penalty for adding a hypothesis, parameters of detection algorithm
float hyp_penalty;

//only patches with probabilities to belong to object higher than this threshold can vote
double PatchBgThreshold;

//minimal allowed probability of a patch to belong to object, if probability is lesser the vote is ignored
double ProbVoteThreshold;

// radius of blur for hough images
float blur_radius;

// size of images at subsequent scales differ by coefficient 
float ResizeCoef;

// width of a bounding box at each scale corresponding to detection
float HalfBBoxWidth;

// height of a bounding box at each scale corresponding to detection
float HalfBBoxHeight;

// maximum number of objects in an image 
float MaxObjectsCount;




// =======================================================================================================
// load config file for dataset
void loadConfig(const char* filename) 
{
 std::cout << "Loading config file" << std::endl;

 char buffer[400];
 std::ifstream in(filename);

 if(in.is_open()) 
 {
  // folder with input images
  in.getline(buffer,400);
  host_folder = new char[400];
  in.getline(host_folder,400); 

  // output folder
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  output_folder = new char[400];
  in.getline(output_folder,400); 

  // width of a bounding box at each scale corresponding to detection
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in >> BBoxWidth; 

  // height of a bounding box at each scale corresponding to detection
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in >> BBoxHeight; 

  // number of scales for multi-scale detection
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in >> iNumberOfScales; 

  // resize images before detection by this coefficient
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in >> koef; 

  //forest name
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  forest_path = new char[400];
  in.getline(forest_path,400); 

  //size of patch in a forest
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in >> PatchSize; 

  // width of a bounding box at each scale corresponding to detection
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in >> HalfBBoxWidth; 

  // height of a bounding box at each scale corresponding to detection
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in >> HalfBBoxHeight; 
    
  // bias of background cost, parameters of detection algorithm
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in >> bg_bias;

  // penalty for adding a hypothesis, parameters of detection algorithm
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in >> hyp_penalty;

  //only patches with probabilities to belong to object higher than this threshold can vote
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in >> PatchBgThreshold;

  //minimal allowed probability of a patch to belong to object, if probability is lesser the vote is ignored
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in >> ProbVoteThreshold;

  // radius of blur for hough images
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in >> blur_radius; 

  // size of images at subsequent scales differ by coefficient 
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in >> ResizeCoef; 

  // maximum number of objects in an image 
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in.getline(buffer,400); 
  in >> MaxObjectsCount; 


  std::cout << "Done" << std::endl;
 } 
 else 
 {
   std::cerr << "File not found " << filename << std::endl;
   exit(-1);
 }
 in.close();
}



// =======================================================================================================
void main(int argc, char *argv[]) 
{
 if (argc < 2)
 {
   std::cerr << "Provide path to config file as argument" << std::endl;
   return;
 }

 loadConfig(argv[1]);

 // create directory for output
 std::string execstr = "mkdir ";
 execstr += output_folder;
 system( execstr.c_str() );

 DIR *pDIR;
 struct dirent *entry;
 if( pDIR=opendir(host_folder) )
 {
  while(entry = readdir(pDIR))
  {
   if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 )
   {
    char input_image_path[256];
    sprintf(input_image_path, "%s\\%s", host_folder, entry->d_name);

    char output_image_path[256];
    sprintf(output_image_path, "%s\\%s_result.png", output_folder, entry->d_name);

    CGreedyDetection gp;
    gp.SetForest(forest_path, 
         PatchSize, blur_radius,
         PatchBgThreshold, ProbVoteThreshold,
         BBoxWidth, BBoxHeight, 
         HalfBBoxWidth, HalfBBoxHeight, 
         iNumberOfScales, ResizeCoef);

     gp.Detect(input_image_path, koef, bg_bias, hyp_penalty, output_image_path, MaxObjectsCount);
   }
  }
  closedir(pDIR);
 }
}


