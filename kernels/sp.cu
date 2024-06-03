/**
 * @file sp.cu
 * @brief Sweeping plane implementation for the three main approaches described in the report
*/
#include "sp.cuh"
#include "../src/utils.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "../src/constants.hpp"
#include "kernels.cuh"
#include "../src/camera_pointers.hpp"

#define BLOCK_DIM 8


std::vector<cv::Mat> sweeping_plane_CPU(cam const ref, std::vector<cam> const &cam_vector, int window)
{   
    std::cout << "Sweeping plane CPU" << std::endl;
	// Initialization to MAX value
	// std::vector<float> cost_cube(ref.width * ref.height * ZPlanes, 255.f);
	std::vector<cv::Mat> cost_cube(ZPlanes);
	for (int i = 0; i < cost_cube.size(); ++i)
	{
		cost_cube[i] = cv::Mat(ref.height, ref.width, CV_32FC1, 255.);
	}

	// For each camera in the setup (reference is skipped)
	for (auto &cam : cam_vector)
	{
		if (cam.name == ref.name)
			continue;

		std::cout << "Cam: " << cam.name << std::endl;
		// For each pixel and candidate: (i) calculate projection index, (ii) calculate cost against reference, (iii) store minimum cost
		for (int zi = 0; zi < ZPlanes; zi++)
		{
			for (int y = 0; y < ref.height; y++)
			{
				for (int x = 0; x < ref.width; x++)
				{
					// (i) calculate projection index

					// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
					double z = ZNear * ZFar / (ZNear + (((double)zi / (double)ZPlanes) * (ZFar - ZNear)));

					// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
					double X_ref = (ref.p.K_inv[0] * x + ref.p.K_inv[1] * y + ref.p.K_inv[2]) * z;
					double Y_ref = (ref.p.K_inv[3] * x + ref.p.K_inv[4] * y + ref.p.K_inv[5]) * z;
					double Z_ref = (ref.p.K_inv[6] * x + ref.p.K_inv[7] * y + ref.p.K_inv[8]) * z;

					// 3D in ref camera coordinates to 3D world
					double X = ref.p.R_inv[0] * X_ref + ref.p.R_inv[1] * Y_ref + ref.p.R_inv[2] * Z_ref - ref.p.t_inv[0];
					double Y = ref.p.R_inv[3] * X_ref + ref.p.R_inv[4] * Y_ref + ref.p.R_inv[5] * Z_ref - ref.p.t_inv[1];
					double Z = ref.p.R_inv[6] * X_ref + ref.p.R_inv[7] * Y_ref + ref.p.R_inv[8] * Z_ref - ref.p.t_inv[2];

					// 3D world to projected camera 3D coordinates
					double X_proj = cam.p.R[0] * X + cam.p.R[1] * Y + cam.p.R[2] * Z - cam.p.t[0];
					double Y_proj = cam.p.R[3] * X + cam.p.R[4] * Y + cam.p.R[5] * Z - cam.p.t[1];
					double Z_proj = cam.p.R[6] * X + cam.p.R[7] * Y + cam.p.R[8] * Z - cam.p.t[2];

					// Projected camera 3D coordinates to projected camera 2D coordinates
					double x_proj = (cam.p.K[0] * X_proj / Z_proj + cam.p.K[1] * Y_proj / Z_proj + cam.p.K[2]);
					double y_proj = (cam.p.K[3] * X_proj / Z_proj + cam.p.K[4] * Y_proj / Z_proj + cam.p.K[5]);
					double z_proj = Z_proj;

					// Check if the projected point is inside the camera frame
					x_proj = x_proj < 0 || x_proj >= cam.width ? 0 : roundf(x_proj);
					y_proj = y_proj < 0 || y_proj >= cam.height ? 0 : roundf(y_proj);

					// (ii) calculate cost against reference
					// Calculating cost in a window
					float cost = 0.0f;
					float cc = 0.0f;
					for (int k = -window / 2; k <= window / 2; k++)
					{
						for (int l = -window / 2; l <= window / 2; l++)
						{	
							// Check if the pixel is inside the frame else skip
							if (x + l < 0 || x + l >= ref.width)
								continue;
							if (y + k < 0 || y + k >= ref.height)
								continue;
							if (x_proj + l < 0 || x_proj + l >= cam.width)
								continue;
							if (y_proj + k < 0 || y_proj + k >= cam.height)
								continue;

							// Y
							cost += fabs(ref.YUV[0].at<uint8_t>(y + k, x + l) - cam.YUV[0].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
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
					cost_cube[zi].at<float>(y, x) = fminf(cost_cube[zi].at<float>(y, x), cost);
				}
			}
		}
	}

	// Visualize costs
	// for (int zi = 0; zi < ZPlanes; zi++)
	// {
	// 	std::cout << "plane " << zi << std::endl;
	// 	cv::namedWindow("Cost", cv::WINDOW_NORMAL);
	// 	cv::imshow("Cost", cost_cube.at(zi) / 255.f);
	// 	cv::waitKey(0);
	// }
	return cost_cube;
}


std::vector<cv::Mat> sweeping_plane_GPU1(cam const ref, std::vector<cam> const &cam_vector, int window)
{   
	
    std::cout << "Sweeping plane GPU1" << std::endl;
	// Initialization to MAX value
	// std::vector<float> cost_cube(ref.width * ref.height * ZPlanes, 255.f);
	std::vector<cv::Mat> cost_cube(ZPlanes);
	for (int i = 0; i < cost_cube.size(); ++i)
	{
		cost_cube[i] = cv::Mat(ref.height, ref.width, CV_32FC1, 255.);
	}
	std::cout << "Initialized cost cube" << std::endl;
	// Create a 3D grid of 3D blocks of size BLOCK_DIM x BLOCK_DIM x BLOCK_DIM
	dim3 BLOCK_SIZE(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
	dim3 GRID_SIZE((ref.width + BLOCK_DIM - 1) / BLOCK_DIM, (ref.height + BLOCK_DIM - 1) / BLOCK_DIM, ZPlanes);

	// convert cost cube to an array of 3D arrays and send it in global memory
	float * h_cost_cube = new float[cost_cube.size() * cost_cube[0].rows * cost_cube[0].cols];
	convert_vector_of_2D_mat_to_array_of_2D_arrays<float>(cost_cube, h_cost_cube);
	float * d_cost_cube;
	cudaMalloc(&d_cost_cube, cost_cube.size() * cost_cube[0].rows * cost_cube[0].cols * cost_cube[0].channels() * sizeof(float));
	cudaMemcpy(d_cost_cube, h_cost_cube, cost_cube.size() * cost_cube[0].rows * 
				cost_cube[0].cols * cost_cube[0].channels() * sizeof(float), cudaMemcpyHostToDevice);

	int * d_width;
	int * d_height;
	cudaMalloc(&d_width, sizeof(int));
	cudaMalloc(&d_height, sizeof(int));
	cudaMemcpy(d_width, &ref.width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_height, &ref.height, sizeof(int), cudaMemcpyHostToDevice);

	// Prepare the pointers for the reference camera
	camera_pointers ref_p = camera_pointers(ref);

	// Prepare the pointers for the other cameras
	camera_pointers cam_p_1 = camera_pointers(ref);

	// allocate memory for the non changing data
	allocate_memory(&ref_p, &cam_p_1, ref);

	std::cout << "Allocated memory" << std::endl;
	// For each camera in the setup (reference is skipped)
	for (auto &cam : cam_vector)
	{
		if (cam.name == ref.name)
			continue;

		std::cout << "Cam: " << cam.name << std::endl;

		// Fill the pointers with the data in the cpu
		convert_vector_to_array(cam.p.K, cam_p_1.h_K);
		convert_vector_to_array(cam.p.R, cam_p_1.h_R);
		convert_vector_to_array(cam.p.t, cam_p_1.h_t);
		convert_mat_to_array<u_int8_t>(cam.YUV[0], cam_p_1.h_Y);


		// Copy data to the GPU
		cudaMemcpy(cam_p_1.d_Y, cam_p_1.h_Y, cam.YUV[0].rows * cam.YUV[0].cols * sizeof(u_int8_t), cudaMemcpyHostToDevice);
		cudaMemcpy(cam_p_1.d_K, cam_p_1.h_K, cam.p.K.size() * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(cam_p_1.d_R, cam_p_1.h_R, cam.p.R.size() * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(cam_p_1.d_t, cam_p_1.h_t, cam.p.t.size() * sizeof(double), cudaMemcpyHostToDevice);

		// Set the L1 cache size
		cudaFuncSetAttribute(first_approach, cudaFuncAttributePreferredSharedMemoryCarveout, 0);

		// Call the kernel and compute the time taken with the cuda events
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		first_approach<<<GRID_SIZE, BLOCK_SIZE>>>(d_width, d_height, cam_p_1.d_Y, ref_p.d_Y, cam_p_1.d_K, cam_p_1.d_R, cam_p_1.d_t, 
													ref_p.d_K, ref_p.d_R, ref_p.d_t, d_cost_cube);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Time taken: " << milliseconds << " ms" << std::endl;

		// Copy the data back to the CPU
		cudaMemcpy(h_cost_cube, d_cost_cube, cost_cube.size() * cost_cube[0].rows * cost_cube[0].cols * 
					cost_cube[0].channels() * sizeof(float), cudaMemcpyDeviceToHost);

	}

	// convert the array of 3D arrays to cost cube
	get_results_from_GPU(d_cost_cube, h_cost_cube, cost_cube);
	// Free the memory
	free_memory(&ref_p, &cam_p_1);
	cudaFree(d_cost_cube);
	delete[] h_cost_cube;

	return cost_cube;

}



std::vector<cv::Mat> sweeping_plane_GPU2(cam const ref, std::vector<cam> const &cam_vector, int window)
{   
	
    std::cout << "Sweeping plane GPU2" << std::endl;
	// Initialization to MAX value
	// std::vector<float> cost_cube(ref.width * ref.height * ZPlanes, 255.f);
	std::vector<cv::Mat> cost_cube(ZPlanes);
	for (int i = 0; i < cost_cube.size(); ++i)
	{
		cost_cube[i] = cv::Mat(ref.height, ref.width, CV_32FC1, 255.);
	}
	std::cout << "Initialized cost cube" << std::endl;
	// Create a 3D grid of 3D blocks of size BLOCK_DIM x BLOCK_DIM x BLOCK_DIM
	dim3 BLOCK_SIZE(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
	dim3 GRID_SIZE((ref.width + BLOCK_DIM - 1) / BLOCK_DIM, (ref.height + BLOCK_DIM - 1) / BLOCK_DIM, ZPlanes);

	// convert cost cube to an array of 3D arrays and send it in global memory
	float * h_cost_cube = new float[cost_cube.size() * cost_cube[0].rows * cost_cube[0].cols];
	convert_vector_of_2D_mat_to_array_of_2D_arrays<float>(cost_cube, h_cost_cube);
	float * d_cost_cube;
	cudaMalloc(&d_cost_cube, cost_cube.size() * cost_cube[0].rows * cost_cube[0].cols * cost_cube[0].channels() * sizeof(float));
	cudaMemcpy(d_cost_cube, h_cost_cube, cost_cube.size() * cost_cube[0].rows * cost_cube[0].cols * cost_cube[0].channels() * sizeof(float), cudaMemcpyHostToDevice);

	int * d_width;
	int * d_height;
	cudaMalloc(&d_width, sizeof(int));
	cudaMalloc(&d_height, sizeof(int));
	cudaMemcpy(d_width, &ref.width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_height, &ref.height, sizeof(int), cudaMemcpyHostToDevice);

	// Prepare the pointers for the reference camera
	camera_pointers ref_p = camera_pointers(ref);

	// Prepare the pointers for the other cameras
	camera_pointers cam_p_1 = camera_pointers(ref);
	camera_pointers cam_p_2 = camera_pointers(ref);
	camera_pointers cam_p_3 = camera_pointers(ref);


	std::cout << "Allocated memory" << std::endl;
	allocate_memory_3_cams(&ref_p, &cam_p_1, &cam_p_2, &cam_p_3, ref, cam_vector[1], cam_vector[2], cam_vector[3]);
	
	// Set the L1 cache size
	cudaFuncSetAttribute(first_approach, cudaFuncAttributePreferredSharedMemoryCarveout, 0);

	// Call the kernel and compute the time taken with the cuda events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	single_kernel_approach<<<GRID_SIZE, BLOCK_SIZE>>>(d_width, d_height, cam_p_1.d_Y, cam_p_2.d_Y, cam_p_3.d_Y, ref_p.d_Y, 
														cam_p_1.d_K, cam_p_1.d_R, cam_p_1.d_t, cam_p_2.d_K, cam_p_2.d_R, 
														cam_p_2.d_t, cam_p_3.d_K, cam_p_3.d_R, cam_p_3.d_t, ref_p.d_K, ref_p.d_R, 
														ref_p.d_t, d_cost_cube);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Time taken: " << milliseconds << " ms" << std::endl;

	// Copy the data back to the CPU
	cudaMemcpy(h_cost_cube, d_cost_cube, cost_cube.size() * cost_cube[0].rows * cost_cube[0].cols * cost_cube[0].channels() * sizeof(float), cudaMemcpyDeviceToHost);


	// convert the array of 3D arrays to cost cube
	get_results_from_GPU(d_cost_cube, h_cost_cube, cost_cube);
	// Free the memory
	free_memory_3_cams(&ref_p, &cam_p_1, &cam_p_2, &cam_p_3);

	cudaFree(d_cost_cube);
	delete[] h_cost_cube;

	return cost_cube;

}


std::vector<cv::Mat> sweeping_plane_GPU3(cam const ref, std::vector<cam> const &cam_vector, int window)
{   
	
    std::cout << "Sweeping plane GPU3" << std::endl;
	// Initialization to MAX value
	// std::vector<float> cost_cube(ref.width * ref.height * ZPlanes, 255.f);
	std::vector<cv::Mat> cost_cube(ZPlanes);
	for (int i = 0; i < cost_cube.size(); ++i)
	{
		cost_cube[i] = cv::Mat(ref.height, ref.width, CV_32FC1, 255.);
	}
	std::cout << "Initialized cost cube" << std::endl;
	// Create a 3D grid of 3D blocks of size BLOCK_DIM x BLOCK_DIM x BLOCK_DIM
	dim3 BLOCK_SIZE(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
	dim3 GRID_SIZE((ref.width + BLOCK_DIM - 1) / BLOCK_DIM, (ref.height + BLOCK_DIM - 1) / BLOCK_DIM, ZPlanes);

	// convert cost cube to an array of 3D arrays and send it in global memory
	float * h_cost_cube = new float[cost_cube.size() * cost_cube[0].rows * cost_cube[0].cols];
	convert_vector_of_2D_mat_to_array_of_2D_arrays<float>(cost_cube, h_cost_cube);
	float * d_cost_cube;
	cudaMalloc(&d_cost_cube, cost_cube.size() * cost_cube[0].rows * cost_cube[0].cols * cost_cube[0].channels() * sizeof(float));
	cudaMemcpy(d_cost_cube, h_cost_cube, cost_cube.size() * cost_cube[0].rows * 
				cost_cube[0].cols * cost_cube[0].channels() * sizeof(float), cudaMemcpyHostToDevice);

	int * d_width;
	int * d_height;
	cudaMalloc(&d_width, sizeof(int));
	cudaMalloc(&d_height, sizeof(int));
	cudaMemcpy(d_width, &ref.width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_height, &ref.height, sizeof(int), cudaMemcpyHostToDevice);

	// Prepare the pointers for the reference camera
	camera_pointers ref_p = camera_pointers(ref);

	// Prepare the pointers for the other cameras
	camera_pointers cam_p_1 = camera_pointers(ref);

	// allocate memory for the non changing data
	allocate_memory(&ref_p, &cam_p_1, ref);

	std::cout << "Allocated memory" << std::endl;
	// For each camera in the setup (reference is skipped)
	for (auto &cam : cam_vector)
	{
		if (cam.name == ref.name)
			continue;

		std::cout << "Cam: " << cam.name << std::endl;

		// Fill the pointers with the data in the cpu
		convert_vector_to_array(cam.p.K, cam_p_1.h_K);
		convert_vector_to_array(cam.p.R, cam_p_1.h_R);
		convert_vector_to_array(cam.p.t, cam_p_1.h_t);
		convert_mat_to_array<u_int8_t>(cam.YUV[0], cam_p_1.h_Y);


		// Copy data to the GPU
		cudaMemcpy(cam_p_1.d_Y, cam_p_1.h_Y, cam.YUV[0].rows * cam.YUV[0].cols * sizeof(u_int8_t), cudaMemcpyHostToDevice);
		cudaMemcpy(cam_p_1.d_K, cam_p_1.h_K, cam.p.K.size() * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(cam_p_1.d_R, cam_p_1.h_R, cam.p.R.size() * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(cam_p_1.d_t, cam_p_1.h_t, cam.p.t.size() * sizeof(double), cudaMemcpyHostToDevice);

		// Set the L1 cache size
		cudaFuncSetAttribute(first_approach, cudaFuncAttributePreferredSharedMemoryCarveout, 0);

		// Call the kernel and compute the time taken with the cuda events
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		parent_kernel_approach<<<GRID_SIZE, BLOCK_SIZE>>>(d_width, d_height, cam_p_1.d_Y, ref_p.d_Y, cam_p_1.d_K, cam_p_1.d_R, cam_p_1.d_t, 
													ref_p.d_K, ref_p.d_R, ref_p.d_t, d_cost_cube);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Time taken: " << milliseconds << " ms" << std::endl;

		// Copy the data back to the CPU
		cudaMemcpy(h_cost_cube, d_cost_cube, cost_cube.size() * cost_cube[0].rows * cost_cube[0].cols * 
					cost_cube[0].channels() * sizeof(float), cudaMemcpyDeviceToHost);

	}

	// convert the array of 3D arrays to cost cube
	get_results_from_GPU(d_cost_cube, h_cost_cube, cost_cube);
	// Free the memory
	free_memory(&ref_p, &cam_p_1);
	cudaFree(d_cost_cube);
	delete[] h_cost_cube;

	return cost_cube;

}



// Define the type of the function pointer
using SweepFunctionType = std::vector<cv::Mat> (*)(cam const, std::vector<cam> const&, int);
// Define the map of the functions
std::map<std::string, SweepFunctionType, std::less<std::string>> sweep_functions = {
    {"CPU", sweeping_plane_CPU},
    {"GPU1", sweeping_plane_GPU1},
	{"GPU2", sweeping_plane_GPU2},
	{"GPU3", sweeping_plane_GPU3}
};