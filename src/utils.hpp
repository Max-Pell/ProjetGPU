/**
 * @file utils.hpp
 * @brief This file contains the declaration of the utility functions used in the project
*/
#include <opencv2/opencv.hpp>
#include <vector>
#include "camera_pointers.hpp"

#ifndef UTILS_HPP
#define UTILS_HPP



/**
 * @brief This function copy the cost cube from the GPU to the CPU
 * 
 * @param d_cost_cube // Pointer to the cost cube on the GPU
 * @param h_cost_cube // Pointer to the cost cube on the CPU
*/
void get_results_from_GPU(float *d_cost_cube, float *h_cost_cube, std::vector<cv::Mat> &cost_cube);

// Convert an array of 2D arrays to a vector of 2D matrices
template <typename T>
void convert_array_of_2D_arrays_to_vector_of_2D_mat(T *array, std::vector<cv::Mat> &mat_vector, int rows, int cols) {
	for (int i = 0; i < mat_vector.size(); i++) {
		mat_vector[i] = cv::Mat(rows, cols, CV_32FC1);
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < cols; column++) {
				mat_vector[i].at<T>(row, column) = array[i * rows * cols + row * cols + column];
			}
		}
	}
}


/**
 * @brief This converts a vector of 2D matrices to an array of 2D arrays
 * 
 * @param mat_vector // Vector of 2D matrices
 * @param array // Array of 2D arrays
*/
template <typename T>
void convert_vector_of_2D_mat_to_array_of_2D_arrays(std::vector<cv::Mat> const &mat_vector, T *array) {
	for (int mat = 0; mat < mat_vector.size(); mat++) {
		for (int row = 0; row < mat_vector[mat].rows; row++) {
			for (int column = 0; column < mat_vector[mat].cols; column++) {
				array[mat * mat_vector[mat].rows * mat_vector[mat].cols + row * mat_vector[mat].cols + column] = mat_vector[mat].at<T>(row, column);
			}
		}
	}
}


/**
 * @brief This function converts a 2D matrix to an array
 * 
 * @param mat // 2D matrix
 * @param array // Array
*/
template <typename T>
void convert_mat_to_array(cv::Mat const &mat, T *array) {
	for (int row = 0; row < mat.rows; row++) {
		for (int column = 0; column < mat.cols; column++) {
			array[row * mat.cols + column] = mat.at<T>(row, column);
		}
	}
}

/**
 * @brief This function converts an array to a 2D matrix
 * 
 * @param array // Array
 * @param mat // 2D matrix
 * @param rows // Number of rows
*/
template <typename T>
void convert_vector_to_array(std::vector<T> const &vec, T * array) {
	for (int i = 0; i < vec.size(); i++) {
		array[i] = vec[i];
	}
}

/**
 * @brief This function allocates memory for the camera and the reference camera
 * 
 * @param ref_p // Pointer to the reference camera
 * @param cam_p // Pointer to the camera
 * @param ref // Reference camera
*/
void allocate_memory(camera_pointers * ref_p, camera_pointers * cam_p, const cam & ref);

/**
 * @brief This function allocates memory for the cameras and the reference camera
 * 
 * @param ref_p // Pointer to the reference camera
 * @param cam_p_1 // Pointer to the first camera
 * @param cam_p_2 // Pointer to the second camera
 * @param ref // Reference camera
 * @param cam_1 // First camera
 * @param cam_2 // Second camera
*/
void allocate_memory_3_cams(camera_pointers * ref_p, camera_pointers * cam_p_1, camera_pointers * cam_p_2, camera_pointers * cam_p_3, const cam & ref, const cam & cam_1, const cam & cam_2, const cam & cam_3);

/**
 * @brief This function frees the memory allocated for the camera and the reference camera
 * 
 * @param ref_p // Pointer to the reference camera
 * @param cam_p // Pointer to the camera
*/
void free_memory(camera_pointers * ref_p, camera_pointers * cam_p);


/**
 * @brief This function frees the memory allocated for the cameras and the reference camera
 * 
 * @param ref_p // Pointer to the reference camera
 * @param cam_p_1 // Pointer to the first camera
 * @param cam_p_2 // Pointer to the second camera
*/
void free_memory_3_cams(camera_pointers * ref_p, camera_pointers * cam_p_1, camera_pointers * cam_p_2, camera_pointers * cam_p_3);

#endif // UTILS_HPP