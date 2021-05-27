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

bool invalid(const int& index, const int16_t* ptr_pCloudData,
    const uint8_t* ptr_imgData);

void addPoint(const int& index, int16_t* ptr_pcl);

bool inSegment(const int& index, const short* ptr_pclData,
    const Point& minPoint, const Point& maxPoint);

void stitch(const int& index, Point& point, int16_t* ptr_pcl,
    uint8_t* ptr_img_GL, uint8_t* ptr_img_CV);

std::pair<Point, Point> queryBoundary(std::vector<Point>& points);

bool invalid(const int& index, const k4a_float2_t* ptr_xyTable,
    const uint16_t* ptr_depth);

void addXYZ(const int& index, std::vector<int16_t>& pCloudFrame,
    const int16_t* ptr_pCloudData);

void addPixel_GL(const int& index, std::vector<uint8_t>& imgFrame_GL,
    const uint8_t* ptr_imgData);

void addPixel_CV(const int& index, std::vector<uint8_t>& imgFrame_CV,
    const uint8_t* ptr_imgData);

void adapt(const int& index, Point& point,
    const std::vector<int16_t>& pCloudFrame,
    const std::vector<uint8_t>& imgFrame);

void addXYZ(const int& index, std::vector<int16_t>& pCloudFrame);

void addPixel_GL(const int& index, std::vector<uint8_t>& imgFrame_GL);

void addPixel_GL(const int& index, std::vector<uint8_t>& imgFrame_GL,
    const std::vector<uint8_t>& imgFrame);

void addXYZ(const int& index, std::vector<int16_t>& pCloudFrame,
    const std::vector<int16_t>& otherPCloudFrame);

bool inSegment(const int& index, const std::vector<int16_t>& ptr_pclData,
    const Point& minPoint, const Point& maxPoint);

void addPixel_CV(const int& index, std::vector<uint8_t>& imgFrame_CV);

bool null(const int& index, std::vector<int16_t>& pCloudFrame,
    std::vector<uint8_t>& imgFrame);
}
#endif /*INTACT_UTILS_H*/
