#ifndef SEED_H
#define SEED_H

#include <algorithm>
#include <cmath>
#include <vector>

#include "point.h"

namespace seed {

std::vector<Point> sample(std::vector<Point> points)
{
    const int SAMPLE_SIZE = 1000;

    /** find centroid */
    Point center = Point::centroid(points);

    /** compute distances from centroid for each point */
    for (auto& point : points) {
        float distance = point.distance(center);
        point.m_distance.second = distance;
    }

    /** sort points */
    Point::sort(points);

    //   /** re-assign id's for traces
    //    * *********************************************************************/
    //   int index = 0; // <-- used for traces
    //   for (auto& point : points) {
    //       point.m_id = index; // <-- used for traces
    //       index++;            // <-- used for traces
    //   }
    //   /** re-assign id's for traces
    //    * *********************************************************************/

    /** sample edge of cluster points */
    std::vector<Point> sample
        = std::vector<Point>(points.end() - SAMPLE_SIZE, points.end());

    /** visual check of sample size */
    std::cout << "sampled " << sample.size() << " points" << std::endl;

    return sample;
}
}
#endif /* SEED_H */
