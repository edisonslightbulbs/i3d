#ifndef LDA_H
#define LDA_H

#include <vector>

#include "point.h"

namespace lda {

    std::vector<Point> reduce(std::vector<Point>& points)
    {
        /** find within-point variance using the centroid */
        Point centroid = Point::centroid(points);
        for (auto& point : points) {
            float distance = centroid.distance(point);
            point.m_distance.second = distance;
        }
        Point::sort(points);
        return points;
    }
}
#endif /* LDA_H */
