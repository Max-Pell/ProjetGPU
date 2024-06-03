/**
 * @file single_kernel_approach.cu
 * @brief This file contains the implementation of the single kernel approach
 */
#include "kernels.cuh"
#include <vector>
#include <opencv2/opencv.hpp>
#include "../src/constants.hpp"


#define ZNear 0.3f
#define ZFar 1.1f
#define ZPlanes 256
#define window 9


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
		float * d_cost_cube) {
			
			
			// compute x, y, zi knowing that the grid is 3D and blocks are 8x8x8
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;
			int zi = blockIdx.z * blockDim.z + threadIdx.z;


			// check if the pixel is inside the frame
			if (x >= *d_width || y >= *d_height || zi >= ZPlanes)
				return;


			// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
			double z = ZNear * ZFar / (ZNear + (((double)zi / (double)ZPlanes) * (ZFar - ZNear)));

			// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
			double X_ref = (d_K_inv[0] * x + d_K_inv[1] * y + d_K_inv[2]) * z;
			double Y_ref = (d_K_inv[3] * x + d_K_inv[4] * y + d_K_inv[5]) * z;
			double Z_ref = (d_K_inv[6] * x + d_K_inv[7] * y + d_K_inv[8]) * z;

			// 3D in ref camera coordinates to 3D world
			double X = d_R_inv[0] * X_ref + d_R_inv[1] * Y_ref + d_R_inv[2] * Z_ref - d_t_inv[0];
			double Y = d_R_inv[3] * X_ref + d_R_inv[4] * Y_ref + d_R_inv[5] * Z_ref - d_t_inv[1];
			double Z = d_R_inv[6] * X_ref + d_R_inv[7] * Y_ref + d_R_inv[8] * Z_ref - d_t_inv[2];

			// 3D world to projected camera 3D coordinates for cam 1 2 and 3
			double X_proj_1 = d_R_1[0] * X + d_R_1[1] * Y + d_R_1[2] * Z - d_t_1[0];
			double Y_proj_1 = d_R_1[3] * X + d_R_1[4] * Y + d_R_1[5] * Z - d_t_1[1];
			double Z_proj_1 = d_R_1[6] * X + d_R_1[7] * Y + d_R_1[8] * Z - d_t_1[2];

			double X_proj_2 = d_R_2[0] * X + d_R_2[1] * Y + d_R_2[2] * Z - d_t_2[0];
			double Y_proj_2 = d_R_2[3] * X + d_R_2[4] * Y + d_R_2[5] * Z - d_t_2[1];
			double Z_proj_2 = d_R_2[6] * X + d_R_2[7] * Y + d_R_2[8] * Z - d_t_2[2];

			double X_proj_3 = d_R_3[0] * X + d_R_3[1] * Y + d_R_3[2] * Z - d_t_3[0];
			double Y_proj_3 = d_R_3[3] * X + d_R_3[4] * Y + d_R_3[5] * Z - d_t_3[1];
			double Z_proj_3 = d_R_3[6] * X + d_R_3[7] * Y + d_R_3[8] * Z - d_t_3[2];
			

			// Projected camera 3D coordinates to projected camera 2D coordinates for cam 1 2 and 3
			double x_proj_1 = (d_K_1[0] * X_proj_1 / Z_proj_1 + d_K_1[1] * Y_proj_1 / Z_proj_1 + d_K_1[2]);
			double y_proj_1 = (d_K_1[3] * X_proj_1 / Z_proj_1 + d_K_1[4] * Y_proj_1 / Z_proj_1 + d_K_1[5]);

			double x_proj_2 = (d_K_2[0] * X_proj_2 / Z_proj_2 + d_K_2[1] * Y_proj_2 / Z_proj_2 + d_K_2[2]);
			double y_proj_2 = (d_K_2[3] * X_proj_2 / Z_proj_2 + d_K_2[4] * Y_proj_2 / Z_proj_2 + d_K_2[5]);

			double x_proj_3 = (d_K_3[0] * X_proj_3 / Z_proj_3 + d_K_3[1] * Y_proj_3 / Z_proj_3 + d_K_3[2]);
			double y_proj_3 = (d_K_3[3] * X_proj_3 / Z_proj_3 + d_K_3[4] * Y_proj_3 / Z_proj_3 + d_K_3[5]);
			//double z_proj = Z_proj;


			// Check if the projected point is inside the camera frame
			x_proj_1 = x_proj_1 < 0 || x_proj_1 >= *d_width ? 0 : roundf(x_proj_1);
			y_proj_1 = y_proj_1 < 0 || y_proj_1 >= *d_height ? 0 : roundf(y_proj_1);

			x_proj_2 = x_proj_2 < 0 || x_proj_2 >= *d_width ? 0 : roundf(x_proj_2);
			y_proj_2 = y_proj_2 < 0 || y_proj_2 >= *d_height ? 0 : roundf(y_proj_2);

			x_proj_3 = x_proj_3 < 0 || x_proj_3 >= *d_width ? 0 : roundf(x_proj_3);
			y_proj_3 = y_proj_3 < 0 || y_proj_3 >= *d_height ? 0 : roundf(y_proj_3);


			// (ii) calculate cost against reference
			// Calculating cost in a window
			float cost_1 = 0.0f;
			float cost_2 = 0.0f;
			float cost_3 = 0.0f;
			float cc_1 = 0.0f;
			float cc_2 = 0.0f;
			float cc_3 = 0.0f;
			for (int k = -window / 2; k <= window / 2; k++)
			{
				for (int l = -window / 2; l <= window / 2; l++)
				{	
					// Check if the pixel is inside the frame else skip
					if (x + l < 0 || x + l >= *d_width)
						continue;
					if (y + k < 0 || y + k >= *d_height)
						continue;
					if (x_proj_1 + l < 0 || x_proj_1 + l >= *d_width)
						goto cam_2;
					if (y_proj_1 + k < 0 || y_proj_1 + k >= *d_height)
						goto cam_2;

					// Y
					cost_1 += fabsf(d_Y_ref[(y + k) * *d_width + (x + l)] - d_Y_cam_1[((int)y_proj_1 + k) * *d_width + ((int)x_proj_1 + l)]);
					cc_1 += 1.0f;

					cam_2:

					// Check if the pixel is inside the frame else skip
					if (x_proj_2 + l < 0 || x_proj_2 + l >= *d_width)
						goto cam_3;
					if (y_proj_2 + k < 0 || y_proj_2 + k >= *d_height)
						goto cam_3;

					// Y
					cost_2 += fabsf(d_Y_ref[(y + k) * *d_width + (x + l)] - d_Y_cam_2[((int)y_proj_2 + k) * *d_width + ((int)x_proj_2 + l)]);
					cc_2 += 1.0f;

					cam_3:

					// Check if the pixel is inside the frame else skip
					if (x_proj_3 + l < 0 || x_proj_3 + l >= *d_width)
						continue;
					if (y_proj_3 + k < 0 || y_proj_3 + k >= *d_height)
						continue;
					
					// Y
					cost_3 += fabsf(d_Y_ref[(y + k) * *d_width + (x + l)] - d_Y_cam_3[((int)y_proj_3 + k) * *d_width + ((int)x_proj_3 + l)]);
					cc_3 += 1.0f;
					
				}
			}
			cost_1 /= cc_1;
			cost_2 /= cc_2;
			cost_3 /= cc_3;
			

			//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
			// only the minimum cost for all the cameras is stored
			d_cost_cube[zi * (*d_height * *d_width) + y * *d_width + x] = fminf(d_cost_cube[zi * (*d_height * *d_width) + y * *d_width + x], fminf(cost_1, fminf(cost_2, cost_3)));
}