/**
 * @file sweeping_plane.cuh
 * @brief This file contains the declaration of the sweeping plane algorithm functions for the three approaches described in the report
*/
#ifndef SWEEPING_PLANE_CUH
#define SWEEPING_PLANE_CUH

#include "../src/cam_params.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "../src/constants.hpp"


/**
 * @brief This function apply the sweeping plane algorithm to the cameras to generate a cost cube for each planene
 * 	then build a depth map from the cost cube
 * 
 * @param ref 
 * @param cam_vector 
 * @param window 
 * @return std::vector<cv::Mat> 
 */
std::vector<cv::Mat> sweeping_plane_CPU(cam const ref, std::vector<cam> const &cam_vector, int window);

/**
 * @brief This function apply the sweeping plane algorithm with the first approach on GPU to the cameras to generate a cost cube for each planene
 * 	then build a depth map from the cost cube
 * 
 * @param ref 
 * @param cam_vector 
 * @param window 
 * @return std::vector<cv::Mat> 
 */
std::vector<cv::Mat> sweeping_plane_GPU1(cam const ref, std::vector<cam> const &cam_vector, int window);

/**
 * @brief This function apply the sweeping plane algorithm with the second approach on GPU to the cameras to generate a cost cube for each planene
 * 
 * @param ref
 * @param cam_vector
 * @param window
 * @return std::vector<cv::Mat>
*/
std::vector<cv::Mat> sweeping_plane_GPU2(cam const ref, std::vector<cam> const &cam_vector, int window);

/**
 * @brief WARNING (THIS FUNCTION DOESN'T WORK) This function apply the sweeping plane algorithm with the last approach on GPU to the cameras to generate a cost cube for each planene
 * 
 * @param ref
 * @param cam_vector
 * @param window
 * @return std::vector<cv::Mat>
*/
std::vector<cv::Mat> sweeping_plane_GPU3(cam const ref, std::vector<cam> const &cam_vector, int window);


// Define the type of the function pointer
using SweepFunctionType = std::vector<cv::Mat> (*)(cam const, std::vector<cam> const&, int);
// Define the map of the functions
extern std::map<std::string, SweepFunctionType, std::less<std::string>> sweep_functions;

#endif // SWEEPING_PLANE_CUH