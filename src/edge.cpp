#include "edge.h"

/** factor tabletop unevenness */
const float UPPER_VANISHING_RANGE = 20;

std::vector<Point> edge::detect(std::vector<Point>& points)
{
    std::vector<Point> proposal;
    Point centroid = Point::centroid(points);

    for (auto& point : points) {
        if (point.m_z < centroid.m_z + UPPER_VANISHING_RANGE) {
            proposal.push_back(point);
        }
    }
    return proposal;
}
