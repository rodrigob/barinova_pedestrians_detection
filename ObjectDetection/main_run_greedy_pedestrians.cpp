#include "DenseGreedyDetection/GreedyDetection.h"

#include <cstdlib>
#include <string>
#include <stdexcept>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost;

// folder with input images
filesystem::path host_folder;

// output folder
filesystem::path output_folder;

// width of a bounding box at each scale corresponding to detection
int BBoxWidth; 

// height of a bounding box at each scale corresponding to detection
int BBoxHeight; 

// number of scales for multi-scale detection
int iNumberOfScales;

// resize images before detection by this coefficient
double koef;

//forest name
filesystem::path forest_path;

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
void loadConfig(const filesystem::path filename)
{
    std::cout << "Loading config file" << std::endl;

    const int buffer_size = 400;
    char buffer[buffer_size];
    std::ifstream input_file(filename.string().c_str());

    if(input_file.is_open())
    {
        // folder with input images
        input_file.getline(buffer,buffer_size);
        string host_folder_string;
        std::getline(input_file, host_folder_string);
        //host_folder = host_folder_string.substr(0, host_folder_string.length() - 1);
        host_folder = host_folder_string;

        // output folder
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        string output_folder_string;
        std::getline(input_file, output_folder_string);
        //output_folder = output_folder_string.substr(0, output_folder_string.length() - 1);
        output_folder = output_folder_string;

        // width of a bounding box at each scale corresponding to detection
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file >> BBoxWidth;

        // height of a bounding box at each scale corresponding to detection
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file >> BBoxHeight;

        // number of scales for multi-scale detection
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file >> iNumberOfScales;

        // resize images before detection by this coefficient
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file >> koef;

        //forest name
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        string forest_path_string;
        std::getline(input_file, forest_path_string);
        //forest_path = forest_path_string.substr(0, forest_path_string.length() - 1);
        forest_path = forest_path_string;

        //size of patch in a forest
        input_file.getline(buffer,400);
        input_file.getline(buffer,400);
        input_file >> PatchSize;

        // width of a bounding box at each scale corresponding to detection
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file >> HalfBBoxWidth;

        // height of a bounding box at each scale corresponding to detection
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file >> HalfBBoxHeight;

        // bias of background cost, parameters of detection algorithm
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file >> bg_bias;

        // penalty for adding a hypothesis, parameters of detection algorithm
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file >> hyp_penalty;

        //only patches with probabilities to belong to object higher than this threshold can vote
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file >> PatchBgThreshold;

        //minimal allowed probability of a patch to belong to object, if probability is lesser the vote is ignored
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file >> ProbVoteThreshold;

        // radius of blur for hough images
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file >> blur_radius;

        // size of images at subsequent scales differ by coefficient
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file >> ResizeCoef;

        // maximum number of objects in an image
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file.getline(buffer,buffer_size);
        input_file >> MaxObjectsCount;


        std::cout << "Done" << std::endl;
    }
    else
    {
        std::cerr << "File not found " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    input_file.close();

    return;
}



// =======================================================================================================
int main_run_greedy_pedestrians(int argc, char *argv[])
{
    using namespace boost::filesystem;

    if (argc < 2)
    {
        std::cerr << "Provide path to config file as argument" << std::endl;
        return EXIT_SUCCESS;
    }

    loadConfig(argv[1]);

    printf("Reading images from host_folder == '%s'\n", host_folder.string().c_str());

    if(exists(host_folder) == false)
    {
        printf("IMPORTANT: Is your configuration file in Unix line ending format ? (Windows style line ending not supported)\n");
        throw std::invalid_argument("Indicated host_folder does not exists");
    }

    if(is_directory(host_folder) == false)
    {
        throw std::invalid_argument("Indicated host_folder is not a directory");
    }

    // create directory for output
    {
        create_directory(output_folder);
        printf("Created output folder %s\n", output_folder.string().c_str());
    }

    CGreedyDetection gp;
    gp.SetForest(forest_path.string().c_str(),
                 PatchSize, blur_radius,
                 PatchBgThreshold, ProbVoteThreshold,
                 BBoxWidth, BBoxHeight,
                 HalfBBoxWidth, HalfBBoxHeight,
                 iNumberOfScales, ResizeCoef);

    directory_iterator host_folder_it(host_folder);
    for (; host_folder_it != directory_iterator(); ++host_folder_it)
    {
        if (is_regular_file(host_folder_it->status()))
        {
            const path input_image_path = host_folder_it->path();
            const path output_image_path = output_folder / host_folder_it->path().filename();

            printf("Processing image %s\n", input_image_path.filename().c_str());

            gp.Detect(input_image_path.string().c_str(),
                      koef, bg_bias, hyp_penalty,
                      output_image_path.string().c_str(),
                      MaxObjectsCount);
        } // end of "if current file is a regular file"

    } // end of "for each file inside host_folder"

    return EXIT_SUCCESS;
} // end of main_run_greedy_pedestrians



int main(int argc, char *argv[])
{
    int ret = EXIT_SUCCESS;

    try
    {
        ret = main_run_greedy_pedestrians(argc, argv);
    }
    // on linux re-throw the exception in order to get the information
    catch (std::exception & e)
    {
        cout << "\033[1;31mA std::exception was raised:\033[0m " << e.what () << endl;
        ret = EXIT_FAILURE; // an error appeared
        throw;
    }
    catch (...)
    {
        cout << "\033[1;31mAn unknown exception was raised\033[0m " << endl;
        ret = EXIT_FAILURE; // an error appeared
        throw;
    }

    return ret;
}

