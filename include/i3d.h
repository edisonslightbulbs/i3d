#ifndef INTACT_UTILS_H
#define INTACT_UTILS_H

#include <string>
#include <thread>
#include <torch/script.h>

#include "intact.h"
#include "io.h"

namespace i3d {

void configTorch(
    std::vector<std::string>& classNames, torch::jit::script::Module& module);

void adapt(const int& index, Point& point, const int16_t* ptr_pcl,
    const uint8_t* ptr_img);

bool invalid(const int& index, const int16_t* ptr_pcl, const uint8_t* ptr_img);

void addPoint(const int& index, int16_t* ptr_pcl);

void addPoint(
    const int& index, int16_t* ptr_pclDest, const int16_t* ptr_pclSrc);

void addPixel_CV(
    const int& index, uint8_t* ptr_imgDest, const uint8_t* ptr_imgSrc);

void addPixel_GL(
    const int& index, uint8_t* ptr_imgDest, const uint8_t* ptr_imgSrc);

void addPixel_CV(const int& index, uint8_t* ptr_img);

void addPixel_GL(const int& index, uint8_t* ptr_img);

bool inSegment(
    const int& index, const short* ptr_pcl, const Point& min, const Point& max);

void stitch(const int& index, Point& point, int16_t* ptr_pcl,
    uint8_t* ptr_img_GL, uint8_t* ptr_img_CV);

std::pair<Point, Point> queryBoundary(std::vector<Point>& points);
}
#endif /*INTACT_UTILS_H*/
