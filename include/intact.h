#ifndef INTACT_H
#define INTACT_H

#include <memory>
#include <mutex>
#include <vector>

#include "kinect.h"
#include "point.h"

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
void segment(std::shared_ptr<Kinect>& sptr_kinect);

void render(std::shared_ptr<Kinect>& sptr_kinect);
}
#endif /* INTACT_H */
