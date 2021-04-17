#ifndef INTACT_H
#define INTACT_H

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "context.h"
#include "kinect.h"
#include "point.h"

namespace intact {

/**
 * segmentContext
 *   Initiates pipeline operation for segmenting context.
 *
 * @param sptr_kinect
 *   Kinect device.
 */
void segmentContext(std::shared_ptr<Kinect>& sptr_kinect);

/**
 * render
 *   Renders point cloud in real-time.
 *
 * @param sptr_kinect
 *   Kinect device.
 */
void render(std::shared_ptr<Kinect>& sptr_kinect);

/**
 * queryContextBoundary
 *   Queries the min max point boundaries of segmented context.
 *
 * @param context
 *   Segmented interaction context.
 *
 * @retval
 *    { Point_min, Point_max }
 */
std::pair<Point, Point> queryContextBoundary(std::vector<Point>& context);

/**
 * parsePcl
 *   Parses point cloud from std::vector<float> to std::vector<Point>
 *
 * @param sptr_kinect
 *   Kinect device.
 *
 * @retval
 *    Point cloud points.
 */
std::vector<Point> parsePcl(std::shared_ptr<Kinect>& sptr_kinect);

Context getContext();
}
#endif /* INTACT_H */
