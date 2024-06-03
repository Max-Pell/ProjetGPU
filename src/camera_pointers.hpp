/**
 * @file camera_pointers.hpp
 * @brief This file contains the declaration of the camera_pointers class
*/
#include "constants.hpp"

#ifndef CAMERA_POINTERS_HPP
#define CAMERA_POINTERS_HPP

/**
 * @brief This class contains the pointers to the camera parameters
*/
class camera_pointers
{
public:
    // CPU pointers
    double * h_K;
	double * h_R;
	double * h_t;
	u_int8_t * h_Y;

    // GPU pointers
	u_int8_t * d_Y;
	double * d_K;
	double * d_R;
	double * d_t;
	
    camera_pointers(const cam& cam);
    ~camera_pointers();
};



#endif // CAMERA_POINTERS_HPP
