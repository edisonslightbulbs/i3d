#include "edge.h"

    const float VANISHING_RANGE = 0.004;
    std::vector<Point> edge::detect(std::vector<Point>& points)
    {
        std::vector<Point> proposal;
        Point centroid = Point::centroid(points);

        for (auto& point : points) {
            if (point.m_z < centroid.m_z + points.size() * VANISHING_RANGE) {
                proposal.push_back(point);
            }
        }
        return proposal;
    }
