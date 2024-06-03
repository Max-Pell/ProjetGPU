/**
 * @file child_kernel.cu
 * @brief This file contains the implementation of the child kernel
*/

#include "kernels.cuh"
#include <vector>
#include <opencv2/opencv.hpp>
#include "../src/constants.hpp"


#define ZNear 0.3f
#define ZFar 1.1f
#define ZPlanes 256
#define window 9


__global__ void child_kernel(int * x, int * y,int * d_width, int * d_height, u_int8_t * d_Y_ref, u_int8_t * d_Y_cam,
                                float * cost, double * x_proj, double * y_proj, float * cc) {
	// compute k and l knowing that the grid is 2D and blocks are window x window
	
	int k = threadIdx.x - window / 2;
	int l = threadIdx.y - window / 2;
	// print the values of k and l
	
	// Check if the pixel is inside the frame else skip
	if (*x + l < 0 || *x + l >= *d_width)
		return;
	if (*y + k < 0 || *y + k >= *d_height)
		return;
	if (*x_proj + l < 0 || *x_proj + l >= *d_width)
		return;
	if (*y_proj + k < 0 || *y_proj + k >= *d_height)
		return;

	// Y
	*cost += fabsf(d_Y_ref[(*y + k) * *d_width + (*x + l)] - d_Y_cam[((int)*y_proj + k) * *d_width + ((int)*x_proj + l)]);

	*cc += 1.0f;
}