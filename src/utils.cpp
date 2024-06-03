/**
 * @file utils.cpp
 * @brief This file contains the implementation of the utility functions used in the project
*/
#include "utils.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "cuda_runtime.h"


// Get the results from the GPU
void get_results_from_GPU(float *d_cost_cube, float *h_cost_cube, std::vector<cv::Mat> &cost_cube) {
	// copy the data from the GPU to the CPU
	cudaMemcpy(h_cost_cube, d_cost_cube, cost_cube.size() * cost_cube[0].rows * cost_cube[0].cols * cost_cube[0].channels() * sizeof(float), cudaMemcpyDeviceToHost);
	// copy the data from the array to the vector
	convert_array_of_2D_arrays_to_vector_of_2D_mat<float>(h_cost_cube, cost_cube, cost_cube[0].rows, cost_cube[0].cols);
}

void allocate_memory(camera_pointers * ref_p, camera_pointers * cam_p, const cam & ref) {
	convert_vector_to_array(ref.p.K_inv, (*ref_p).h_K);
	convert_vector_to_array(ref.p.R_inv, (*ref_p).h_R);
	convert_vector_to_array(ref.p.t_inv, (*ref_p).h_t);
	convert_mat_to_array<u_int8_t>(ref.YUV[0], (*ref_p).h_Y);
	cudaMalloc(&(*ref_p).d_Y, ref.YUV[0].rows * ref.YUV[0].cols * sizeof(u_int8_t));
	cudaMalloc(&(*ref_p).d_K, ref.p.K_inv.size() * sizeof(double));
	cudaMalloc(&(*ref_p).d_R, ref.p.R_inv.size() * sizeof(double));
	cudaMalloc(&(*ref_p).d_t, ref.p.t_inv.size() * sizeof(double));
	cudaMemcpy((*ref_p).d_Y, (*ref_p).h_Y, ref.YUV[0].rows * ref.YUV[0].cols * sizeof(u_int8_t), cudaMemcpyHostToDevice);
	cudaMemcpy((*ref_p).d_K, (*ref_p).h_K, ref.p.K_inv.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((*ref_p).d_R, (*ref_p).h_R, ref.p.R_inv.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((*ref_p).d_t, (*ref_p).h_t, ref.p.t_inv.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&(*cam_p).d_Y, ref.YUV[0].rows * ref.YUV[0].cols * sizeof(u_int8_t));
	cudaMalloc(&(*cam_p).d_K, ref.p.K.size() * sizeof(double));
	cudaMalloc(&(*cam_p).d_R, ref.p.R.size() * sizeof(double));
	cudaMalloc(&(*cam_p).d_t, ref.p.t.size() * sizeof(double));
}

void allocate_memory_3_cams(camera_pointers * ref_p, camera_pointers * cam_p_1, camera_pointers * cam_p_2, camera_pointers * cam_p_3, const cam & ref, const cam & cam_1, const cam & cam_2, const cam & cam_3) {
	// Allocate ref cam
	convert_vector_to_array(ref.p.K_inv, (*ref_p).h_K);
	convert_vector_to_array(ref.p.R_inv, (*ref_p).h_R);
	convert_vector_to_array(ref.p.t_inv, (*ref_p).h_t);
	convert_mat_to_array<u_int8_t>(ref.YUV[0], (*ref_p).h_Y);
	cudaMalloc(&(*ref_p).d_Y, ref.YUV[0].rows * ref.YUV[0].cols * sizeof(u_int8_t));
	cudaMalloc(&(*ref_p).d_K, ref.p.K_inv.size() * sizeof(double));
	cudaMalloc(&(*ref_p).d_R, ref.p.R_inv.size() * sizeof(double));
	cudaMalloc(&(*ref_p).d_t, ref.p.t_inv.size() * sizeof(double));
	cudaMemcpy((*ref_p).d_Y, (*ref_p).h_Y, ref.YUV[0].rows * ref.YUV[0].cols * sizeof(u_int8_t), cudaMemcpyHostToDevice);
	cudaMemcpy((*ref_p).d_K, (*ref_p).h_K, ref.p.K_inv.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((*ref_p).d_R, (*ref_p).h_R, ref.p.R_inv.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((*ref_p).d_t, (*ref_p).h_t, ref.p.t_inv.size() * sizeof(double), cudaMemcpyHostToDevice);

	// Allocate cam 1
	convert_vector_to_array(cam_1.p.K, (*cam_p_1).h_K);
	convert_vector_to_array(cam_1.p.R, (*cam_p_1).h_R);
	convert_vector_to_array(cam_1.p.t, (*cam_p_1).h_t);
	convert_mat_to_array<u_int8_t>(cam_1.YUV[0], (*cam_p_1).h_Y);
	cudaMalloc(&(*cam_p_1).d_Y, cam_1.YUV[0].rows * cam_1.YUV[0].cols * sizeof(u_int8_t));
	cudaMalloc(&(*cam_p_1).d_K, cam_1.p.K.size() * sizeof(double));
	cudaMalloc(&(*cam_p_1).d_R, cam_1.p.R.size() * sizeof(double));
	cudaMalloc(&(*cam_p_1).d_t, cam_1.p.t.size() * sizeof(double));
	cudaMemcpy((*cam_p_1).d_Y, (*cam_p_1).h_Y, cam_1.YUV[0].rows * cam_1.YUV[0].cols * sizeof(u_int8_t), cudaMemcpyHostToDevice);
	cudaMemcpy((*cam_p_1).d_K, (*cam_p_1).h_K, cam_1.p.K.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((*cam_p_1).d_R, (*cam_p_1).h_R, cam_1.p.R.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((*cam_p_1).d_t, (*cam_p_1).h_t, cam_1.p.t.size() * sizeof(double), cudaMemcpyHostToDevice);

	// Allocate cam 2
	convert_vector_to_array(cam_2.p.K, (*cam_p_2).h_K);
	convert_vector_to_array(cam_2.p.R, (*cam_p_2).h_R);
	convert_vector_to_array(cam_2.p.t, (*cam_p_2).h_t);
	convert_mat_to_array<u_int8_t>(cam_2.YUV[0], (*cam_p_2).h_Y);
	cudaMalloc(&(*cam_p_2).d_Y, cam_2.YUV[0].rows * cam_2.YUV[0].cols * sizeof(u_int8_t));
	cudaMalloc(&(*cam_p_2).d_K, cam_2.p.K.size() * sizeof(double));
	cudaMalloc(&(*cam_p_2).d_R, cam_2.p.R.size() * sizeof(double));
	cudaMalloc(&(*cam_p_2).d_t, cam_2.p.t.size() * sizeof(double));
	cudaMemcpy((*cam_p_2).d_Y, (*cam_p_2).h_Y, cam_2.YUV[0].rows * cam_2.YUV[0].cols * sizeof(u_int8_t), cudaMemcpyHostToDevice);
	cudaMemcpy((*cam_p_2).d_K, (*cam_p_2).h_K, cam_2.p.K.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((*cam_p_2).d_R, (*cam_p_2).h_R, cam_2.p.R.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((*cam_p_2).d_t, (*cam_p_2).h_t, cam_2.p.t.size() * sizeof(double), cudaMemcpyHostToDevice);

	// Allocate cam 3
	convert_vector_to_array(cam_3.p.K, (*cam_p_3).h_K);
	convert_vector_to_array(cam_3.p.R, (*cam_p_3).h_R);
	convert_vector_to_array(cam_3.p.t, (*cam_p_3).h_t);
	convert_mat_to_array<u_int8_t>(cam_3.YUV[0], (*cam_p_3).h_Y);
	cudaMalloc(&(*cam_p_3).d_Y, cam_3.YUV[0].rows * cam_3.YUV[0].cols * sizeof(u_int8_t));
	cudaMalloc(&(*cam_p_3).d_K, cam_3.p.K.size() * sizeof(double));
	cudaMalloc(&(*cam_p_3).d_R, cam_3.p.R.size() * sizeof(double));
	cudaMalloc(&(*cam_p_3).d_t, cam_3.p.t.size() * sizeof(double));
	cudaMemcpy((*cam_p_3).d_Y, (*cam_p_3).h_Y, cam_3.YUV[0].rows * cam_3.YUV[0].cols * sizeof(u_int8_t), cudaMemcpyHostToDevice);
	cudaMemcpy((*cam_p_3).d_K, (*cam_p_3).h_K, cam_3.p.K.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((*cam_p_3).d_R, (*cam_p_3).h_R, cam_3.p.R.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((*cam_p_3).d_t, (*cam_p_3).h_t, cam_3.p.t.size() * sizeof(double), cudaMemcpyHostToDevice);

}

void free_memory(camera_pointers * ref_p, camera_pointers * cam_p) {
	cudaFree((*cam_p).d_Y);
	cudaFree((*cam_p).d_K);
	cudaFree((*cam_p).d_R);
	cudaFree((*cam_p).d_t);
	delete[] (*cam_p).h_K;
	delete[] (*cam_p).h_R;
	delete[] (*cam_p).h_t;
	delete[] (*cam_p).h_Y;
	cudaFree((*ref_p).d_K);
	cudaFree((*ref_p).d_R);
	cudaFree((*ref_p).d_t);
	cudaFree((*cam_p).d_Y);
	delete[] (*ref_p).h_K;
	delete[] (*ref_p).h_R;
	delete[] (*ref_p).h_t;
	delete[] (*ref_p).h_Y;
}

void free_memory_3_cams(camera_pointers * ref_p, camera_pointers * cam_p_1, camera_pointers * cam_p_2, camera_pointers * cam_p_3) {
	// Free ref cam
	cudaFree((*ref_p).d_Y);
	cudaFree((*ref_p).d_K);
	cudaFree((*ref_p).d_R);
	cudaFree((*ref_p).d_t);
	delete[] (*ref_p).h_K;
	delete[] (*ref_p).h_R;
	delete[] (*ref_p).h_t;
	delete[] (*ref_p).h_Y;

	// Free cam 1
	cudaFree((*cam_p_1).d_Y);
	cudaFree((*cam_p_1).d_K);
	cudaFree((*cam_p_1).d_R);
	cudaFree((*cam_p_1).d_t);
	delete[] (*cam_p_1).h_K;
	delete[] (*cam_p_1).h_R;
	delete[] (*cam_p_1).h_t;
	delete[] (*cam_p_1).h_Y;

	// Free cam 2
	cudaFree((*cam_p_2).d_Y);
	cudaFree((*cam_p_2).d_K);
	cudaFree((*cam_p_2).d_R);
	cudaFree((*cam_p_2).d_t);
	delete[] (*cam_p_2).h_K;
	delete[] (*cam_p_2).h_R;
	delete[] (*cam_p_2).h_t;
	delete[] (*cam_p_2).h_Y;

}
