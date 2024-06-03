#include "cam_params.hpp"
#include "constants.hpp"
#include "graph.h"
#include "../kernels/sp.cuh"

#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <string>


// rtx 2070 GPU capability 7.5
// Max threads per block: 1024
// Max shared memory per SM: 64kb (can be used as L1 cache)
// Max shared memory per block: 48kb
// Max threads per SM: 2048
// Max blocks per SM: 16
// Number of SMs: 36
// cmake -DCMAKE_BUILD_TYPE=Debug .. to build in debug mode

#define SHRT_MAX 32767


/**
 * @brief This function reads the camera parameters from the cam_params.hpp file
 * 
 * @param folder 
 * @return std::vector<cam> 
 */
std::vector<cam> read_cams(std::string const &folder)
{	
	// Init parameters
	std::vector<params<double>> cam_params_vector = get_cam_params();

	// Init cameras
	std::vector<cam> cam_array(cam_params_vector.size());
	for (int i = 0; i < cam_params_vector.size(); i++)
	{
		// Name
		std::string name = folder + "/v" + std::to_string(i) + ".png";

		// Read PNG file
		cv::Mat im_rgb = cv::imread(name);
		cv::Mat im_yuv;
		const int width = im_rgb.cols;
		const int height = im_rgb.rows;

		// Convert to YUV420
		cv::cvtColor(im_rgb, im_yuv, cv::COLOR_BGR2YUV_I420);
		const int size = width * height * 1.5; // YUV 420

		std::vector<cv::Mat> YUV;
		cv::split(im_yuv, YUV);

		// Params
		cam_array.at(i) = cam(name, width, height, size, YUV, cam_params_vector.at(i));
	}

	return cam_array;

	// cv::Mat U(height / 2, width / 2, CV_8UC1, cam_array.at(0).image.data() + (int)(width * height * 1.25));
	// cv::namedWindow("im", cv::WINDOW_NORMAL);
	// cv::imshow("im", U);
	// cv::waitKey(0);
}


cv::Mat find_min(std::vector<cv::Mat> const &cost_cube)
{
	const int zPlanes = cost_cube.size();
	const int height = cost_cube[0].size().height;
	const int width = cost_cube[0].size().width;

	cv::Mat ret(height, width, CV_32FC1, 255.);
	cv::Mat depth(height, width, CV_8U, 255);

	for (int zi = 0; zi < zPlanes; zi++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				if (cost_cube[zi].at<float>(y, x) < ret.at<float>(y, x))
				{
					ret.at<float>(y, x) = cost_cube[zi].at<float>(y, x);
					depth.at<u_int8_t>(y, x) = zi;
				}
			}
		}
	}

	return depth;
}


/*The next two function are used to perform the grpah cut on the results
DO NOT MODIFY THOSE FUNCTIONS - DO NOT TRY TO IMPLEMENT THEM ON THE GPU*/
void depth_estimation_by_graph_cut_sWeight_add_nodes(Graph& g, std::vector<Graph::node_id>& nodes, cv::Size destPixel, cv::Size sourcePixel, cv::Size imgSize, std::vector<double> m_aiEdgeCost, cv::Mat1w labels, int label, double cost_cur) {
	const int idxSourcePixel = sourcePixel.height * imgSize.width + sourcePixel.width;
	const int idxDestPixel = destPixel.height * imgSize.width + destPixel.width;
	const double cost_cur_temp = cost_cur;

	if (labels(sourcePixel.height, sourcePixel.width) != labels(destPixel.height, destPixel.width)) {
		//add a new node and add edge between it and the adjacent nodes
		Graph::node_id tmp_node = g.add_node();
		const double cost_temp = m_aiEdgeCost[std::abs(labels(destPixel.height, destPixel.width) - label)];
		g.set_tweights(tmp_node, 0, m_aiEdgeCost[std::abs(labels(sourcePixel.height, sourcePixel.width) - labels(destPixel.height, destPixel.width))]);
		g.add_edge(nodes[idxSourcePixel], tmp_node, cost_cur_temp, cost_cur_temp);
		g.add_edge(tmp_node, nodes[idxDestPixel], cost_temp, cost_temp);
	}
	else //only add an edge between two nodes
		g.add_edge(nodes[idxSourcePixel], nodes[idxDestPixel], cost_cur_temp, cost_cur_temp);
}

cv::Mat depth_estimation_by_graph_cut_sWeight(std::vector<cv::Mat> const& cost_cube) {
	//DO NOT TRY TO IMPLEMENT THIS FUNCTION ON THE GPU

	const int zPlanes = cost_cube.size();
	const int height = cost_cube[0].size().height;
	const int width = cost_cube[0].size().width;

	//To store the depth values assigned to each pixels, start with 0
	cv::Mat1w labels = cv::Mat::zeros(height, width, CV_16U); 
	//store the cost for a label
	std::vector<double> m_aiEdgeCost;
	double smoothing_lambda = 1.0;
	m_aiEdgeCost.resize(zPlanes);
	for (int i = 0; i < zPlanes; ++i)
		m_aiEdgeCost[i] = smoothing_lambda * i;

	for (int source = 0; source < zPlanes; ++source) {
		//printf("depth layer %i \n", source);
		Graph g;
		std::vector<Graph::node_id> nodes(height * width, nullptr);

		//Putting the weights for the connection to the source and the sink for each nodes
		for (int r = 0; r < height; ++r) {
			for (int c = 0; c < width; ++c) {
				//indice global du pixel
				const int pp = r * width + c;
				nodes[pp] = g.add_node();
				const ushort label = labels(r, c);
				if (label == source)
					g.set_tweights(nodes[pp], cost_cube[source].at<float>(r, c), SHRT_MAX);
				else
					g.set_tweights(nodes[pp], cost_cube[source].at<float>(r, c), cost_cube[label].at<float>(r, c));
			}
		}

		
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				const double cost_curr = m_aiEdgeCost[std::abs(labels(j, i) - source)];

				//create an edge between the adjacent nodes, may add an additional node on this edge if the previously calculated labels are different
				if (i != width - 1) {
					depth_estimation_by_graph_cut_sWeight_add_nodes(g, nodes, cv::Size(i + 1, j), cv::Size(i, j), cv::Size(width, height), m_aiEdgeCost, labels, source, cost_curr);
				}
				if (j != height - 1) {
					depth_estimation_by_graph_cut_sWeight_add_nodes(g, nodes, cv::Size(i, j + 1), cv::Size(i, j), cv::Size(width, height), m_aiEdgeCost, labels, source, cost_curr);
				}
			}
		}
		//printf("nodes and egde set \n");

		//resolve the maximum flow/minimum cut problem
		g.maxflow();

		//update the depth labels, nodes that are still connected to the source will receive a new depth label
		for (int r = 0; r < height; ++r) {
			for (int c = 0; c < width; ++c) {
				const int pp = r * width + c;
				if (g.what_segment(nodes[pp]) != Graph::SOURCE)
					labels(r, c) = ushort(source);
			}
		}
		nodes.clear();
		
		/*
		cv::namedWindow("labels", cv::WINDOW_NORMAL);
		cv::imshow("labels", labels);
		cv::waitKey(0);
		*/

	}

	cv::Mat depth;
	labels.convertTo(depth, CV_8U, 1.0);

	return depth;
}


int main(int argc, char** argv)
{	
	// If argument is missing or it is not a key in the sweep_functions map, print usage and exit
	if (argc < 2 || sweep_functions.find(argv[1]) == sweep_functions.end()) {
		goto usage;
	} else {
		std::cout << "Sweep function: " << argv[1] << std::endl;
	}

	goto program;

	// Print usage and exit
	usage: {
		std::cout << "Usage: " << argv[0] << "<sweep_function_name>" << std::endl;
		std::cout << "Available sweep functions: " << std::endl;
		std::cout << "CPU" << std::endl;
		std::cout << "GPU1" << std::endl;
		std::cout << "GPU2" << std::endl;
		std::cout << "GPU3 is not working" << std::endl;
		return 1;
	}

	program: {
		std::cout << "Program started" << std::endl;
	}

	// Read cams
	std::vector<cam> cam_vector = read_cams("data");
	
	// start the clock
	std::cout << "Starting the clock" << std::endl;
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	// Sweeping algorithm with camera 0 as reference
	// The arg is used as a key to the sweep_functions map and launch the corresponding function
	std::vector<cv::Mat> cost_cube = sweep_functions[argv[1]](cam_vector[0], cam_vector, window);
	// end the clock
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time taken = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

	// Use graph cut to generate depth map 
	// Cleaner results, long compute time
	cv::Mat depth = depth_estimation_by_graph_cut_sWeight(cost_cube);

	// Find min cost and generate depth map
	// Faster result, low quality
	//cv::Mat depth = find_min(cost_cube);

	cv::namedWindow("Depth", cv::WINDOW_NORMAL);
	cv::imshow("Depth", depth);
	cv::waitKey(0);

	cv::imwrite("./depth_map.png", depth);

	return 0;
}