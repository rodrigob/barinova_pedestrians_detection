
Pedestrians detection using Hough forests
==========================================

This a linux port of the original code provided by Olga Barinova from the Vision Group at Moscow State University, 2010.  
Please visit [the project website for more details](http://graphics.cs.msu.ru/en/science/research/machinelearning/hough).

License
-------

This derivative work follows the Microsoft Research Shared Source license, which allows only non-comercial work.  
See original README.txt and MSR-SSLA.txt for more details. 

Citing in publications
----------------------

When using this software, please acknowledge the effort that 
went into development by referencing the paper:

> Barinova O., Lempitsky V., Kohli P.,  
> On detection of multiple object instances using Hough transform,  
> IEEE Conference on Computer Vision and Pattern Recognition (CVPR'10), 2010.

Note that this is not the original software that was used for the paper mentioned above. 
It is a re-implementation. 

Dependencies
------------

This linux version requires:

- [Cmake](http://www.cmake.org)
- [OpenCV](http://opencv.willowgarage.com)
- [Boost](http://www.boost.org/)

Compiling
---------

1. Move inside the `ObjectDetection/ObjectDetection` folder.
2. Run `cmake ./ && make` to build the program `object_detection`

Testing
-------

1. Download the [tud-campus and tud-crossing test sequences](http://www.mis.tu-darmstadt.de/node/382#name:cvpr08_data).
2. Modify `example/campus-config.txt` to fit your local path
3. Make sure the configuration text file is stored in Unix (LF) style line ending
4. Run `object_detection path_to_the_config_txt_file`

You should see progress messages being printed and after some minutes detection images being created in the results folder.

