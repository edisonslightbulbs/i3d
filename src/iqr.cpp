#include "iqr.h"
#include "timer.h"
#include "logger.h"

const float IQR = 1.5;

/** remove outliers using the inter quartile range of normal distribution **/
std::vector<Point> iqr::denoise(std::vector<Point> points)
{
    Timer timer;
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
    points.resize(points.size() - reject);
    LOG(INFO) << timer.getDuration() << " ms: outlier removal runtime";

    return points;
}
