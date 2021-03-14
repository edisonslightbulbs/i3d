#ifndef IQR_H
#define IQR_H

#include "point.h"
#include "io.h"
#include <vector>

namespace iqr {

/** inter quartile range **/
const float IQR = 1.5;

std::vector<Point> denoise(std::vector<Point> points)
{
    float sd = 0;
    float sum = 0;
    float mean = 0;
    float varsum = 0;
    float variance = 0;

    for (auto& point : points) {
        sum += point.m_distance.second;
    }
    mean = sum / (float)points.size();

    for (auto& point : points) {
        varsum += (float)pow((point.m_distance.second - mean), 2);
    }

    variance = varsum / (float)points.size();
    sd = std::sqrt(variance);

    auto upperBound = mean + (IQR * sd);
    int reject = 0;
    for (auto& point : points) {
        if (point.m_distance.second > upperBound) {
            reject++;
        }
    }
    // IO::writeDist(m_points, "./build/noisy.csv");
    points.resize(points.size() - reject);
    // IO::writeDist(m_points, "./build/conditioned.csv");
    return points;
}
}
#endif /* IQR_H */
