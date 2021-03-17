#include <vector>

#include "lda.h"
#include "point.h"
#include "logger.h"
#include "timer.h"
#include "iqr.h"

std::vector<Point> lda::analyze(std::vector<Point>& points)
{
    Timer timer;
    /** find within-point variance using the centroid */
    Point centroid = Point::centroid(points);
    for (auto& point : points) {
        float distance = centroid.distance(point);
        point.m_distance.second = distance;
    }
    Point::sort(points);

    /** remove outliers */
    std::vector<Point> denoisedPoints = iqr::denoise(points);
    LOG(INFO) << timer.getDuration() << " ms: dimension reduction runtime";

    return denoisedPoints;
}
