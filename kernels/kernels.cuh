/**
 * @file kernels.cuh
 * @brief This file contains the declaration of the kernels used in the three approaches
*/

#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <opencv2/opencv.hpp>


/**
 * @brief Kernel that computes the cost cube for one image
 * 
 * @param d_width width of the image
 * @param d_height height of the image
 * @param d_Y_cam camera image
 * @param d_Y_ref reference image
 * @param d_K intrinsic matrix
 * @param d_R rotation matrix
 * @param d_t translation vector
 * @param d_K_inv inverse of the intrinsic matrix
 * @param d_R_inv inverse of the rotation matrix
 * @param d_t_inv inverse of the translation vector
 * @param d_cost_cube cost cube
 * 
 * @note The kernel is launched with a 3D grid and 3D blocks
 * 
*/
__global__ void first_approach(int * d_width,
		int * d_height,
		u_int8_t * d_Y_cam,
		u_int8_t * d_Y_ref,
		double * d_K,
		double * d_R,
		double * d_t,
		double * d_K_inv,
		double * d_R_inv,
		double * d_t_inv,
		float * d_cost_cube);


/**
 * @brief Kernel that computes the cost cube for all the images
 * 
 * @param d_width width of the image
 * @param d_height height of the image
 * @param d_Y_cam camera image
 * @param d_Y_ref reference image
 * @param d_K intrinsic matrix
 * @param d_R rotation matrix
 * @param d_t translation vector
 * @param d_K_inv inverse of the intrinsic matrix
 * @param d_R_inv inverse of the rotation matrix
 * @param d_t_inv inverse of the translation vector
 * @param d_cost_cube cost cube
 * 
 * @note The kernel is launched with a 3D grid and 3D blocks
*/
__global__ void single_kernel_approach(int * d_width,
		int * d_height,
		u_int8_t * d_Y_cam_1,
		u_int8_t * d_Y_cam_2,
		u_int8_t * d_Y_cam_3,
		u_int8_t * d_Y_ref,
		double * d_K_1,
		double * d_R_1,
		double * d_t_1,
		double * d_K_2,
		double * d_R_2,
		double * d_t_2,
		double * d_K_3,
		double * d_R_3,
		double * d_t_3,
		double * d_K_inv,
		double * d_R_inv,
		double * d_t_inv,
		float * d_cost_cube);

/**
 * @brief Parent kernel that computes the cost cube
 * 
 * @param d_width width of the image
 * @param d_height height of the image
 * @param d_Y_cam camera image
 * @param d_Y_ref reference image
 * @param d_K intrinsic matrix
 * @param d_R rotation matrix
 * @param d_t translation vector
 * @param d_K_inv inverse of the intrinsic matrix
 * @param d_R_inv inverse of the rotation matrix
 * @param d_t_inv inverse of the translation vector
 * @param d_cost_cube cost cube
 * 
 * @note The kernel is launched with a 3D grid and 3D blocks
 * 
*/
__global__ void parent_kernel_approach(int * d_width,
		int * d_height,
		u_int8_t * d_Y_cam,
		u_int8_t * d_Y_ref,
		double * d_K,
		double * d_R,
		double * d_t,
		double * d_K_inv,
		double * d_R_inv,
		double * d_t_inv,
		float * d_cost_cube);


/**
 * @brief Child kernel that computes the cost and the cross correlation for a given pixel
 * 
 * @param x x coordinate of the pixel
 * @param y y coordinate of the pixel
 * @param d_width width of the image
 * @param d_height height of the image
 * @param d_Y_ref reference image
 * @param d_Y_cam camera image
 * @param cost cost of the pixel
 * @param x_proj x coordinate of the projected pixel
 * @param y_proj y coordinate of the projected pixel
 * @param cc cross correlation of the pixel
*/
__global__ void child_kernel(int * x, int * y,int * d_width, int * d_height, u_int8_t * d_Y_ref, u_int8_t * d_Y_cam,
                                float * cost, double * x_proj, double * y_proj, float * cc);