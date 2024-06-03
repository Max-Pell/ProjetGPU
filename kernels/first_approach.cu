/**
 * @file first_approach.cu
 * @brief This file contains the implementation of the first approach
*/

#include "kernels.cuh"
#include <vector>
#include <opencv2/opencv.hpp>
#include "../src/constants.hpp"


#define ZNear 0.3f
#define ZFar 1.1f
#define ZPlanes 256
#define window 9


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

			// 3D world to projected camera 3D coordinates
			double X_proj = d_R[0] * X + d_R[1] * Y + d_R[2] * Z - d_t[0];
			double Y_proj = d_R[3] * X + d_R[4] * Y + d_R[5] * Z - d_t[1];
			double Z_proj = d_R[6] * X + d_R[7] * Y + d_R[8] * Z - d_t[2];

			// Projected camera 3D coordinates to projected camera 2D coordinates
			double x_proj = (d_K[0] * X_proj / Z_proj + d_K[1] * Y_proj / Z_proj + d_K[2]);
			double y_proj = (d_K[3] * X_proj / Z_proj + d_K[4] * Y_proj / Z_proj + d_K[5]);
			//double z_proj = Z_proj;


			// Check if the projected point is inside the camera frame
			x_proj = x_proj < 0 || x_proj >= *d_width ? 0 : roundf(x_proj);
			y_proj = y_proj < 0 || y_proj >= *d_height ? 0 : roundf(y_proj);


			// (ii) calculate cost against reference
			// Calculating cost in a window
			float cost = 0.0f;
			float cc = 0.0f;
			for (int k = -window / 2; k <= window / 2; k++)
			{
				for (int l = -window / 2; l <= window / 2; l++)
				{	
					// Check if the pixel is inside the frame else skip
					if (x + l < 0 || x + l >= *d_width)
						continue;
					if (y + k < 0 || y + k >= *d_height)
						continue;
					if (x_proj + l < 0 || x_proj + l >= *d_width)
						continue;
					if (y_proj + k < 0 || y_proj + k >= *d_height)
						continue;

					// Y
					cost += fabsf(d_Y_ref[(y + k) * *d_width + (x + l)] - d_Y_cam[((int)y_proj + k) * *d_width + ((int)x_proj + l)]);
					// print the value added to the cost
				
					// U
					// cost += fabs(ref.YUV[1].at<uint8_t >(y + k, x + l) - cam.YUV[1].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
					// V
					// cost += fabs(ref.YUV[2].at<uint8_t >(y + k, x + l) - cam.YUV[2].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
					cc += 1.0f;
				}
			}
			cost /= cc;
			

			//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
			// only the minimum cost for all the cameras is stored
			d_cost_cube[zi * (*d_height * *d_width) + y * *d_width + x] = fminf(d_cost_cube[zi * (*d_height * *d_width) + y * *d_width + x], cost);
}