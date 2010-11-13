// Author: Olga Barinova, Graphics&Media Lab., Vision Group, Moscow State University
// Email: obarinova@graphics.cs.msu.ru
 
// You may use, copy, reproduce, and distribute this Software for any 
// non-commercial purpose, subject to the restrictions of the 
// Microsoft Research Shared Source license agreement ("MSR-SSLA"). 
// Some purposes which can be non-commercial are teaching, academic 
// research, public demonstrations and personal experimentation. You 
// may also distribute this Software with books or other teaching 
// materials, or publish the Software on websites, that are intended 
// to teach the use of the Software for academic or other 
// non-commercial purposes.
// You may not use or distribute this Software or any derivative works 
// in any form for commercial purposes. Examples of commercial 
// purposes would be running business operations, licensing, leasing, 
// or selling the Software, distributing the Software for use with 
// commercial products, using the Software in the creation or use of 
// commercial products or any other activity which purpose is to 
// procure a commercial gain to you or others.
// If the Software includes source code or data, you may create 
// derivative works of such portions of the Software and distribute 
// the modified Software for non-commercial purposes, as provided 
// herein.

// THE SOFTWARE COMES "AS IS", WITH NO WARRANTIES. THIS MEANS NO 
// EXPRESS, IMPLIED OR STATUTORY WARRANTY, INCLUDING WITHOUT 
// LIMITATION, WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A 
// PARTICULAR PURPOSE, ANY WARRANTY AGAINST INTERFERENCE WITH YOUR 
// ENJOYMENT OF THE SOFTWARE OR ANY WARRANTY OF TITLE OR 
// NON-INFRINGEMENT. THERE IS NO WARRANTY THAT THIS SOFTWARE WILL 
// FULFILL ANY OF YOUR PARTICULAR PURPOSES OR NEEDS. ALSO, YOU MUST 
// PASS THIS DISCLAIMER ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR 
// DERIVATIVE WORKS.

// NEITHER MICROSOFT NOR ANY CONTRIBUTOR TO THE SOFTWARE WILL BE 
// LIABLE FOR ANY DAMAGES RELATED TO THE SOFTWARE OR THIS MSR-SSLA, 
// INCLUDING DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL 
// DAMAGES, TO THE MAXIMUM EXTENT THE LAW PERMITS, NO MATTER WHAT 
// LEGAL THEORY IT IS BASED ON. ALSO, YOU MUST PASS THIS LIMITATION OF 
// LIABILITY ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE 
// WORKS.

// When using this software, please acknowledge the effort that 
// went into development by referencing the paper:
//
// Barinova O., Lempitsky V., Kohli P., On detection of multiple 
// object instances using Hough transform, IEEE Conference on Computer Vision and Pattern 
// Recognition (CVPR'10), 2010.

// Note that this is not the original software that was used for 
// the paper mentioned above. It is a re-implementation. 



Microsoft Visual Studio 2005 solution is provided in folder 'ObjectDetection'. 
OpenCV needs to be installed.

Usage: command.exe [config.txt]

Subdirectory 'example' contains:
campus-config.txt - example of config file to use with TUD-campus image dataset
crossing-config.txt - example of config file to use with TUD-crossing image dataset
pedestrian.dat - trained Hough Forests [1] classifier for pedestrian detection 


[1] Gall J. and Lempitsky V., Class-Specific Hough Forests for Object Detection, 
IEEE Conference on Computer Vision and Pattern Recognition (CVPR'09), 2009.

Olga Barinova

Vision Group, Graphics & Media Lab., Moscow State University, 7 June 2010