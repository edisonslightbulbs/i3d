#include <vector>

#include "polygon.h"
#include "point.h"

bool inside(std::vector<Point>& points, const Point& point, const float& xMax,
    const float& xMin, const float& yMax, const float& yMin)
{
    // https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
    bool inside = false;
    for (int i = 0, j = (int)points.size() - 1; i < points.size(); j = i++) {
        if ((points[i].m_y > point.m_y) != (points[j].m_y > point.m_x)
            && point.m_x < (points[j].m_x - points[i].m_x)
                        * (point.m_y - points[i].m_y)
                        / (points[j].m_y - points[i].m_y)
                    + points[i].m_y) {
            inside = !inside;
        }
    }
    return inside;
}

std::vector<Point> polygon::fit(std::vector<Point>& points)
{
    std::vector<float> x;
    std::vector<float> y;

    for (auto& other : points) {
        x.push_back(other.m_x);
        y.push_back(other.m_y);
    }
    float xMax = *std::max_element(x.begin(), x.end());
    float xMin = *std::min_element(x.begin(), x.end());

    float yMax = *std::max_element(y.begin(), y.end());
    float yMin = *std::min_element(y.begin(), y.end());

    std::vector<Point> proposal;
    for (auto& point : points) {
        if (point.m_x < xMin || point.m_x > xMax || point.m_y < yMin
            || point.m_y > yMax) {
            continue;
        }

        proposal.push_back(point);
        // if(!inside(points, point, xMax, xMin, yMax, yMin)){
        // } else {
        //     proposal.push_back(point);
        // }
    }
    return proposal;
}
