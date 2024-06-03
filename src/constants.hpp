#include <opencv2/core/mat.hpp>
#include "cam_params.hpp"

#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP


extern const float ZNear;
extern const float ZFar;
extern const int ZPlanes;
extern const int window;

typedef unsigned char u_int8_t;

struct cam
{
    std::string name;
    int width;
    int height;
    int size;
    std::vector<cv::Mat> YUV;
    params<double> p;
    cam() : name(""), width(-1), height(-1), size(-1), YUV(), p(){};
    cam(std::string _name, int _width, int _height, int _size, std::vector<cv::Mat> &_YUV, params<double> &_p)
        : name(_name), width(_width), height(_height), size(_size), YUV(_YUV), p(_p){};
};

#endif // CONSTANTS_HPP