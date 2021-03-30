#ifndef INTACT_H
#define INTACT_H

#include <vector>
#include <memory>
#include <mutex>

#include "point.h"
#include "kinect.h"

namespace intact {
/**
 * cut
 *   Identifies and isolates tabletop interaction context segment.
 *
 * @param points
 *   Unorganized 3D point-cloud representation of tabletop environment.
 *
 * @retval
 *    Segment of tabletop interaction context
 */
    void segment(std::mutex& m, Kinect& kinect, const int& numPoints,
                         std::shared_ptr<std::vector<float>>& sptr_points, std::shared_ptr<std::pair<Point, Point>>& sptr_threshold);

    void render(std::mutex& m, Kinect kinect, int numPoints,
                std::shared_ptr<std::vector<float>>& sptr_points, std::shared_ptr<std::pair<Point, Point>>& sptr_threshold);
}
#endif /* INTACT_H */