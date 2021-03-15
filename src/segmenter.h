#ifndef SEGMENTER_H
#define SEGMENTER_H

#include "logger.h"
#include "point.h"
#include "timer.h"

#include <vector>

namespace segmenter {

std::vector<Point> segment(std::pair<std::vector<Point>, int> points)
{
    std::vector<std::vector<Point>> clusters(points.second);

    for (int i = 0; i <= points.second; ++i) {
        for (auto& point : points.first) {
            if (i == point.m_cluster) {
                clusters[i].push_back(point);
            }
        }
    }

    std::vector<Point> tabletopContext;
    for (auto& cluster : clusters) {
        if (cluster.size() > tabletopContext.size()) {
            tabletopContext.clear();
            tabletopContext = cluster;
        }
    }
    return tabletopContext;
}

}
#endif /* SEGMENTER_H */
