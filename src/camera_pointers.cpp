/**
 * @file camera_pointers.cpp
 * @brief This file contains the implementation of the camera_pointers class
*/
#include "camera_pointers.hpp"


camera_pointers::camera_pointers(const cam& cam)
{
    // CPU pointers
    h_K = new double[cam.p.K_inv.size()];
	h_R = new double[cam.p.R_inv.size()];
	h_t = new double[cam.p.t_inv.size()];
	h_Y = new u_int8_t[cam.YUV[0].rows * cam.YUV[0].cols];

    // GPU pointers
    d_Y = nullptr;
    d_K = nullptr;
    d_R = nullptr;
    d_t = nullptr;
}

camera_pointers::~camera_pointers()
{
}
